import { MessageRole } from '$lib/enums';
import type {
	ChatContextBlock,
	DatabaseArchiveChunk,
	DatabaseMessage,
	DatabaseRetrievalHitUsage,
	DatabaseRetrievalTrace,
	MemoryRetrievalHit,
	MemoryRetrievalResponse,
	RetrievalTraceHit
} from '$lib/types';
import { uuid } from '$lib/utils';
import { memoryDebug } from '$lib/utils/memory-debug';
import { DatabaseService } from './database.service';
import { SpominService } from './spomin.service';

interface RetrievalSettings {
	spominEnabled: boolean;
	spominBaseUrl: string;
	spominApiToken?: string;
	spominProject?: string;
	spominCandidateLimit: number;
	spominResultLimit: number;
	spominTokenBudget: number;
	spominTimeoutMs: number;
	embeddingBaseUrl: string;
	embeddingModel: string;
	embeddingTimeoutMs: number;
	localResultLimit: number;
	localTokenBudget: number;
	totalTokenBudget: number;
	semanticThreshold: number;
	lexicalThreshold: number;
}

interface Candidate {
	id: string;
	text: string;
	source: 'conversation-recall' | 'long-term-memory';
	score: number;
	lexicalScore?: number;
	semanticScore?: number;
	providerScore?: number;
	provenance: Record<string, unknown>;
}

interface CandidateEvaluation {
	candidate: Candidate;
	accepted: boolean;
	reason?: string;
}

interface RetrievalQuery {
	value: string;
	latestUser: string;
	terms: string[];
	strategy: 'latest-only' | 'context-expanded';
	supportingCharacterCount: number;
}

interface ActiveCompactionScope {
	id: string;
	generation: number;
	sourceMessageIds: string[];
}

interface LocalCandidateResult {
	candidates: Candidate[];
	evaluations: CandidateEvaluation[];
	detail: string;
}

export interface RetrievalPreparation {
	blocks: ChatContextBlock[];
	trace: DatabaseRetrievalTrace;
	usage: DatabaseRetrievalHitUsage[];
	spominFeedback?: {
		baseUrl: string;
		apiToken?: string;
		timeoutMs: number;
	};
}

const WORD_RE = /[\p{L}\p{N}_-]+/gu;
const STOP_WORDS = new Set([
	'a',
	'an',
	'and',
	'are',
	'did',
	'do',
	'for',
	'how',
	'i',
	'in',
	'is',
	'it',
	'of',
	'on',
	'or',
	'that',
	'the',
	'to',
	'was',
	'we',
	'what',
	'which',
	'with',
	'you'
]);
const CONTEXT_REFERENCE_RE =
	/\b(that one|this|those|these|the same|same one|what about|as before|earlier one|previous one)\b/i;
const MAX_SUPPORTING_CONTEXT_CHARACTERS = 1200;
const MAX_TRACE_EVALUATIONS = 20;

function terms(text: string): string[] {
	return Array.from(
		new Set((text.toLowerCase().match(WORD_RE) ?? []).filter((term) => !STOP_WORDS.has(term)))
	);
}

function cosine(left: number[], right: number[]): number {
	if (!left.length || left.length !== right.length) return 0;
	let dot = 0;
	let leftNorm = 0;
	let rightNorm = 0;
	for (let index = 0; index < left.length; index++) {
		dot += left[index] * right[index];
		leftNorm += left[index] ** 2;
		rightNorm += right[index] ** 2;
	}
	return leftNorm && rightNorm ? dot / Math.sqrt(leftNorm * rightNorm) : 0;
}

function similarity(left: string[], right: string[]): number {
	const a = new Set(left);
	const b = new Set(right);
	const union = new Set([...a, ...b]);
	if (!union.size) return 1;
	return [...a].filter((term) => b.has(term)).length / union.size;
}

function tokenEstimate(text: string): number {
	return Math.max(1, Math.ceil((text.match(WORD_RE)?.length ?? 0) * 1.35));
}

export class RetrievalService {
	private static buildRetrievalQuery(messages: DatabaseMessage[]): RetrievalQuery {
		const conversational = messages.filter(
			(message) =>
				(message.role === MessageRole.USER || message.role === MessageRole.ASSISTANT) &&
				message.content.trim()
		);
		const latestUserIndex = conversational.findLastIndex(
			(message) => message.role === MessageRole.USER
		);
		const latestUser = latestUserIndex >= 0 ? conversational[latestUserIndex].content.trim() : '';
		const queryTerms = terms(latestUser);
		const contextDependent = queryTerms.length < 4 || CONTEXT_REFERENCE_RE.test(latestUser);
		const supportingContext = contextDependent
			? conversational
					.slice(0, latestUserIndex)
					.slice(-4)
					.map((message) => `${message.role}: ${message.content}`)
					.join('\n')
					.slice(-MAX_SUPPORTING_CONTEXT_CHARACTERS)
			: '';
		return {
			value: supportingContext
				? `Current user request:\n${latestUser}\n\nRecent context:\n${supportingContext}`
				: `Current user request:\n${latestUser}`,
			latestUser,
			terms: queryTerms,
			strategy: supportingContext ? 'context-expanded' : 'latest-only',
			supportingCharacterCount: supportingContext.length
		};
	}

	static buildQuery(messages: DatabaseMessage[]): string {
		return RetrievalService.buildRetrievalQuery(messages).value;
	}

	static async fingerprint(value: string): Promise<string> {
		const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
		return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join(
			''
		);
	}

	private static async embeddings(
		input: string[],
		settings: RetrievalSettings
	): Promise<number[][]> {
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), settings.embeddingTimeoutMs);
		const startedAt = performance.now();
		memoryDebug('retrieval.embedding.start', {
			inputCount: input.length,
			model: settings.embeddingModel,
			timeoutMs: settings.embeddingTimeoutMs
		});
		try {
			const response = await fetch(`${settings.embeddingBaseUrl.replace(/\/+$/, '')}/embeddings`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ model: settings.embeddingModel, input }),
				signal: controller.signal
			});
			memoryDebug('retrieval.embedding.response', {
				status: response.status,
				ok: response.ok,
				durationMs: Math.round(performance.now() - startedAt)
			});
			if (!response.ok) throw new Error(`Embedding request failed (${response.status})`);
			const body = (await response.json()) as {
				data: Array<{ index: number; embedding: number[] }>;
			};
			const embeddings = body.data
				.sort((left, right) => left.index - right.index)
				.map((item) => item.embedding);
			memoryDebug('retrieval.embedding.complete', {
				resultCount: embeddings.length,
				dimensions: embeddings[0]?.length ?? 0,
				durationMs: Math.round(performance.now() - startedAt)
			});
			return embeddings;
		} catch (error) {
			memoryDebug('retrieval.embedding.error', {
				durationMs: Math.round(performance.now() - startedAt),
				aborted: controller.signal.aborted,
				error
			});
			throw error;
		} finally {
			clearTimeout(timeout);
		}
	}

	private static lexicalScore(queryTerms: string[], chunk: DatabaseArchiveChunk): number {
		if (!queryTerms.length) return 0;
		const chunkTerms = new Set(chunk.terms);
		return queryTerms.filter((term) => chunkTerms.has(term)).length / queryTerms.length;
	}

	private static async localCandidates(
		conversationId: string,
		activeCompaction: ActiveCompactionScope,
		query: string,
		queryTerms: string[],
		settings: RetrievalSettings
	): Promise<LocalCandidateResult> {
		const archived = await DatabaseService.getConversationArchiveChunks(conversationId);
		const activeMessageIds = new Set(activeCompaction.sourceMessageIds);
		const matching = archived.filter(
			(chunk) =>
				chunk.sourceMessageIds.length > 0 &&
				chunk.sourceMessageIds.every((messageId) => activeMessageIds.has(messageId))
		);
		const unique = new Map<string, DatabaseArchiveChunk>();
		for (const chunk of matching) {
			const key = `${chunk.sourceMessageIds.join('\u0000')}\u0000${chunk.text}`;
			const existing = unique.get(key);
			if (!existing || chunk.createdAt > existing.createdAt) unique.set(key, chunk);
		}
		const chunks = Array.from(unique.values());
		memoryDebug('retrieval.local.archive-loaded', {
			conversationId,
			compactionId: activeCompaction.id,
			compactionGeneration: activeCompaction.generation,
			compactedMessageCount: activeMessageIds.size,
			archiveCount: archived.length,
			chunkCount: chunks.length,
			ignoredArchiveCount: archived.length - matching.length,
			duplicateArchiveCount: matching.length - chunks.length,
			pendingEmbeddingCount: chunks.filter((chunk) => chunk.embeddingStatus === 'pending').length,
			readyEmbeddingCount: chunks.filter((chunk) => chunk.embeddingStatus === 'ready').length
		});
		if (!chunks.length) {
			return { candidates: [], evaluations: [], detail: 'no archived fragments' };
		}
		let queryEmbedding: number[] | undefined;
		let detail = 'lexical+semantic';
		try {
			const pending = chunks.filter((chunk) => chunk.embeddingStatus === 'pending').slice(0, 15);
			const values = await RetrievalService.embeddings(
				[query, ...pending.map((chunk) => chunk.text)],
				settings
			);
			queryEmbedding = values[0];
			await Promise.all(
				pending.map((chunk, index) =>
					DatabaseService.updateArchiveChunk(chunk.id, {
						embedding: values[index + 1],
						embeddingModel: settings.embeddingModel,
						embeddingStatus: 'ready',
						embeddingError: undefined
					})
				)
			);
			pending.forEach((chunk, index) => {
				chunk.embedding = values[index + 1];
				chunk.embeddingStatus = 'ready';
			});
		} catch (error) {
			memoryDebug('retrieval.local.semantic-fallback', {
				conversationId,
				compactionId: activeCompaction.id,
				error
			});
			queryEmbedding = undefined;
			detail =
				error instanceof DOMException && error.name === 'AbortError'
					? 'lexical-only - embedding timed out'
					: 'lexical-only - embedding unavailable';
		}

		const ranked = chunks
			.map((chunk) => {
				const lexical = RetrievalService.lexicalScore(queryTerms, chunk);
				const semantic =
					queryEmbedding && chunk.embedding ? cosine(queryEmbedding, chunk.embedding) : 0;
				return {
					id: `local:${chunk.id}`,
					text: chunk.text,
					source: 'conversation-recall' as const,
					score: Math.max(lexical, semantic),
					lexicalScore: lexical,
					semanticScore: semantic,
					provenance: {
						compactionId: chunk.compactionId,
						sourceMessageIds: chunk.sourceMessageIds,
						generation: chunk.generation
					}
				};
			})
			.sort((left, right) => right.score - left.score);
		let acceptedCount = 0;
		const evaluations = ranked.slice(0, MAX_TRACE_EVALUATIONS).map((candidate) => {
			const aboveThreshold =
				(candidate.lexicalScore ?? 0) >= settings.lexicalThreshold ||
				(candidate.semanticScore ?? 0) >= settings.semanticThreshold;
			if (!aboveThreshold) {
				return { candidate, accepted: false, reason: 'below-threshold' };
			}
			acceptedCount++;
			return acceptedCount <= settings.localResultLimit
				? { candidate, accepted: true }
				: { candidate, accepted: false, reason: 'result-limit' };
		});
		const candidates = evaluations
			.filter((evaluation) => evaluation.accepted)
			.map((evaluation) => evaluation.candidate);
		memoryDebug('retrieval.local.candidates', {
			conversationId,
			evaluatedCount: evaluations.length,
			candidateCount: candidates.length,
			semanticAvailable: Boolean(queryEmbedding),
			candidates: evaluations.map(({ candidate, accepted, reason }) => ({
				id: candidate.id,
				score: candidate.score,
				lexicalScore: candidate.lexicalScore,
				semanticScore: candidate.semanticScore,
				accepted,
				reason
			}))
		});
		return { candidates, evaluations, detail };
	}

	private static evaluateSpominCandidates(
		results: MemoryRetrievalHit[],
		settings: RetrievalSettings,
		profile?: MemoryRetrievalResponse['profile']
	): CandidateEvaluation[] {
		return results.map((result) => {
			const candidate: Candidate = {
				id: `spomin:${result.memory_ids?.[0] ?? result.id}`,
				text: result.text,
				source: 'long-term-memory',
				score: Math.max(result.semantic_score ?? 0, result.lexical_coverage ?? 0),
				lexicalScore: result.lexical_coverage ?? undefined,
				semanticScore: result.semantic_score ?? undefined,
				provenance: {
					chunkId: result.id,
					memoryIds: result.memory_ids,
					project: result.project,
					tier: result.tier,
					createdAt: result.created_at,
					embeddingProfileId: profile?.id,
					embeddingModelId: profile?.model_id
				}
			};
			const accepted =
				(result.lexical_coverage ?? 0) >= settings.lexicalThreshold ||
				(result.semantic_score ?? 0) >= settings.semanticThreshold;
			return {
				candidate,
				accepted,
				reason: accepted ? undefined : 'below-threshold'
			};
		});
	}

	private static mergeSpominChannels(response: MemoryRetrievalResponse): MemoryRetrievalHit[] {
		const merged = new Map<string, MemoryRetrievalHit>();
		for (const result of response.semantic.results) {
			merged.set(result.id, { ...result });
		}
		for (const result of response.keyword.results) {
			const current = merged.get(result.id);
			merged.set(result.id, current ? { ...current, ...result } : { ...result });
		}
		return [...merged.values()];
	}

	static async prepare(input: {
		conversationId: string;
		anchorMessageId: string;
		responseMessageId?: string;
		messages: DatabaseMessage[];
		activeCompaction?: ActiveCompactionScope;
		settings: RetrievalSettings;
	}): Promise<RetrievalPreparation> {
		const retrievalQuery = RetrievalService.buildRetrievalQuery(input.messages);
		const query = retrievalQuery.value;
		const latestUser = retrievalQuery.latestUser;
		const queryTerms = retrievalQuery.terms;
		const queryFingerprint = await RetrievalService.fingerprint(query);
		const userTurn = input.messages.filter((message) => message.role === MessageRole.USER).length;
		const explicitRecall = /\b(remember|recall|previously|before|earlier|last time)\b/i.test(
			latestUser || ''
		);
		const providers: DatabaseRetrievalTrace['providers'] = {};
		memoryDebug('retrieval.prepare.start', {
			conversationId: input.conversationId,
			anchorMessageId: input.anchorMessageId,
			messageCount: input.messages.length,
			userTurn,
			compactionId: input.activeCompaction?.id,
			compactionGeneration: input.activeCompaction?.generation ?? 0,
			compactedMessageCount: input.activeCompaction?.sourceMessageIds.length ?? 0,
			explicitRecall,
			spominEnabled: input.settings.spominEnabled,
			queryStrategy: retrievalQuery.strategy,
			currentMessageCharacterCount: latestUser.length,
			supportingContextCharacterCount: retrievalQuery.supportingCharacterCount,
			queryTermCount: queryTerms.length,
			fingerprint: queryFingerprint,
			budgets: {
				local: input.settings.localTokenBudget,
				spomin: input.settings.spominTokenBudget,
				total: input.settings.totalTokenBudget
			},
			thresholds: {
				semantic: input.settings.semanticThreshold,
				lexical: input.settings.lexicalThreshold
			}
		});

		const localPromise =
			input.activeCompaction && input.activeCompaction.sourceMessageIds.length > 0
				? RetrievalService.localCandidates(
						input.conversationId,
						input.activeCompaction,
						query,
						queryTerms,
						input.settings
					)
				: Promise.resolve(null);
		const spominPromise = input.settings.spominEnabled
			? new SpominService({
					baseUrl: input.settings.spominBaseUrl,
					apiToken: input.settings.spominApiToken,
					timeoutMs: input.settings.spominTimeoutMs
				}).retrieve({
					query,
					candidateLimit: input.settings.spominCandidateLimit,
					project: input.settings.spominProject,
					excludeConversationId: input.conversationId
				})
			: Promise.resolve(null);

		const [localResult, spominResult] = await Promise.allSettled([localPromise, spominPromise]);
		let local: Candidate[] = [];
		let remote: Candidate[] = [];
		let localEvaluations: CandidateEvaluation[] = [];
		let remoteEvaluations: CandidateEvaluation[] = [];
		if (localResult.status === 'fulfilled' && localResult.value) {
			local = localResult.value.candidates;
			localEvaluations = localResult.value.evaluations;
			providers.local = {
				status: 'ok',
				detail: localResult.value.detail
			};
		} else if (localResult.status === 'fulfilled') {
			providers.local = {
				status: 'not-applicable',
				detail: 'conversation is not compacted'
			};
		} else {
			providers.local = { status: 'error', detail: String(localResult.reason) };
		}
		if (!input.settings.spominEnabled) {
			providers.spomin = { status: 'disabled' };
		} else if (spominResult.status === 'fulfilled' && spominResult.value) {
			const mergedResults = RetrievalService.mergeSpominChannels(spominResult.value);
			remoteEvaluations = RetrievalService.evaluateSpominCandidates(
				mergedResults,
				input.settings,
				spominResult.value.profile
			);
			remote = remoteEvaluations
				.filter((evaluation) => evaluation.accepted)
				.map((evaluation) => evaluation.candidate);
			const detail = !mergedResults.length
				? 'no-results'
				: !remote.length
					? 'all-below-threshold'
					: spominResult.value.semantic.status === 'complete' &&
						  spominResult.value.keyword.status === 'complete'
						? 'raw-semantic+keyword'
						: spominResult.value.semantic.status === 'complete'
							? 'semantic-only'
							: 'keyword-only';
			const channelSummary = [
				detail,
				`semantic=${spominResult.value.semantic.results.length}`,
				`keyword=${spominResult.value.keyword.results.length}`,
				`merged=${mergedResults.length}`,
				spominResult.value.profile ? `profile=${spominResult.value.profile.id.slice(0, 12)}` : null
			]
				.filter(Boolean)
				.join('; ');
			providers.spomin = {
				status: 'ok',
				detail:
					spominResult.value.semantic.status === 'unavailable'
						? `${channelSummary}; ${spominResult.value.semantic.error ?? 'semantic unavailable'}`
						: channelSummary
			};
		} else {
			const reason = spominResult.status === 'rejected' ? spominResult.reason : 'unavailable';
			providers.spomin = {
				status:
					reason instanceof DOMException && reason.name === 'AbortError' ? 'timeout' : 'error',
				detail: String(reason)
			};
		}
		memoryDebug('retrieval.providers.complete', {
			conversationId: input.conversationId,
			providers,
			localCandidateCount: local.length,
			localEvaluatedCount: localEvaluations.length,
			spominReturnedCount: remoteEvaluations.length,
			spominCandidateCount: remote.length,
			spominSemanticCount:
				spominResult.status === 'fulfilled' && spominResult.value
					? spominResult.value.semantic.results.length
					: 0,
			spominKeywordCount:
				spominResult.status === 'fulfilled' && spominResult.value
					? spominResult.value.keyword.results.length
					: 0
		});

		const previous = await DatabaseService.getRetrievalHitUsage(input.conversationId);
		const previousById = new Map(
			previous.map((record) => [`${record.source}:${record.hitId}`, record])
		);
		const evaluations = [...localEvaluations, ...remoteEvaluations];
		const traceHits: RetrievalTraceHit[] = evaluations.map(({ candidate, accepted, reason }) => ({
			id: candidate.id,
			source: candidate.source,
			score: candidate.score,
			lexicalScore: candidate.lexicalScore,
			semanticScore: candidate.semanticScore,
			providerScore: candidate.providerScore,
			selected: accepted,
			reason,
			tokenCount: tokenEstimate(candidate.text),
			provenance: candidate.provenance
		}));
		const eligible = [...local, ...remote].filter((candidate) => {
			const prior = previousById.get(`${candidate.source}:${candidate.id}`);
			let reason: string | undefined;
			if (prior && userTurn - prior.lastInjectedUserTurn < 3) {
				const topicChanged = similarity(queryTerms, prior.queryTerms ?? []) < 0.55;
				const generationChanged =
					(input.activeCompaction?.generation ?? 0) !== prior.compactionGeneration;
				const scoreImproved = candidate.score >= prior.score + 0.15;
				if (!topicChanged && !generationChanged && !scoreImproved && !explicitRecall) {
					reason = userTurn - prior.lastInjectedUserTurn <= 1 ? 'consecutive-request' : 'cooldown';
				}
			}
			const hit = traceHits.find(
				(item) => item.id === candidate.id && item.source === candidate.source
			);
			if (hit && reason) {
				hit.selected = false;
				hit.reason = reason;
			}
			return !reason;
		});

		const selected: Candidate[] = [];
		let selectedSpomin = 0;
		const budgets = new Map([
			['conversation-recall', input.settings.localTokenBudget],
			['long-term-memory', input.settings.spominTokenBudget]
		]);
		let totalTokens = 0;
		for (const candidate of eligible.sort((left, right) => right.score - left.score)) {
			const count = tokenEstimate(candidate.text);
			const sourceBudget = budgets.get(candidate.source) ?? 0;
			const overResultLimit =
				candidate.source === 'long-term-memory' &&
				selectedSpomin >= input.settings.spominResultLimit;
			if (
				overResultLimit ||
				count > sourceBudget ||
				totalTokens + count > input.settings.totalTokenBudget
			) {
				const hit = traceHits.find(
					(item) => item.id === candidate.id && item.source === candidate.source
				);
				if (hit) {
					hit.selected = false;
					hit.reason = overResultLimit ? 'result-limit' : 'token-budget';
				}
				continue;
			}
			selected.push(candidate);
			if (candidate.source === 'long-term-memory') selectedSpomin += 1;
			budgets.set(candidate.source, sourceBudget - count);
			totalTokens += count;
			const hit = traceHits.find(
				(item) => item.id === candidate.id && item.source === candidate.source
			);
			if (hit) hit.tokenCount = count;
		}
		memoryDebug('retrieval.selection.complete', {
			conversationId: input.conversationId,
			eligibleCount: eligible.length,
			selectedCount: selected.length,
			selectedTokenEstimate: totalTokens,
			selected: selected.map((candidate) => ({
				id: candidate.id,
				source: candidate.source,
				score: candidate.score,
				lexicalScore: candidate.lexicalScore,
				semanticScore: candidate.semanticScore,
				providerScore: candidate.providerScore
			})),
			skipped: traceHits
				.filter((hit) => !hit.selected)
				.map((hit) => ({
					id: hit.id,
					source: hit.source,
					reason: hit.reason,
					score: hit.score,
					lexicalScore: hit.lexicalScore,
					semanticScore: hit.semanticScore,
					providerScore: hit.providerScore
				}))
		});

		const trace: DatabaseRetrievalTrace = {
			id: uuid(),
			conversationId: input.conversationId,
			anchorMessageId: input.anchorMessageId,
			responseMessageId: input.responseMessageId,
			createdAt: Date.now(),
			query,
			queryFingerprint,
			queryTerms,
			compactionGeneration: input.activeCompaction?.generation ?? 0,
			providers,
			hits: traceHits,
			injectedHitIds: [],
			injectedTokenCount: 0
		};
		return {
			blocks: selected.map((candidate) => ({
				id: candidate.id,
				source: candidate.source,
				content: candidate.text,
				provenance: { ...candidate.provenance, score: candidate.score }
			})),
			trace,
			usage: selected.map((candidate) => ({
				id: `${input.conversationId}:${candidate.source}:${candidate.id}`,
				conversationId: input.conversationId,
				hitId: candidate.id,
				source: candidate.source,
				lastInjectedUserTurn: userTurn,
				queryFingerprint,
				queryTerms,
				score: candidate.score,
				compactionGeneration: input.activeCompaction?.generation ?? 0,
				updatedAt: Date.now()
			})),
			spominFeedback: input.settings.spominEnabled
				? {
						baseUrl: input.settings.spominBaseUrl,
						apiToken: input.settings.spominApiToken,
						timeoutMs: input.settings.spominTimeoutMs
					}
				: undefined
		};
	}

	static async finalize(
		preparation: RetrievalPreparation,
		selectedBlockIds: string[],
		finalPromptTokenCount?: number
	): Promise<void> {
		const selected = new Set(selectedBlockIds);
		const selectedBlocks = new Map(
			preparation.blocks.filter((block) => selected.has(block.id)).map((block) => [block.id, block])
		);
		preparation.trace.injectedHitIds = selectedBlockIds;
		preparation.trace.injectedTokenCount = preparation.trace.hits
			.filter((hit) => selected.has(hit.id))
			.reduce((sum, hit) => sum + (hit.tokenCount ?? 0), 0);
		preparation.trace.finalPromptTokenCount = finalPromptTokenCount;
		for (const hit of preparation.trace.hits) {
			const block = selectedBlocks.get(hit.id);
			if (block) {
				hit.contentSnapshot = block.content;
				hit.provenance = block.provenance;
			}
			if (hit.selected && !selected.has(hit.id)) {
				hit.selected = false;
				hit.reason = 'exact-prompt-budget';
			}
		}
		memoryDebug('retrieval.finalize', {
			conversationId: preparation.trace.conversationId,
			anchorMessageId: preparation.trace.anchorMessageId,
			selectedBlockIds,
			injectedTokenCount: preparation.trace.injectedTokenCount,
			finalPromptTokenCount,
			droppedByExactPromptBudget: preparation.trace.hits
				.filter((hit) => hit.reason === 'exact-prompt-budget')
				.map((hit) => hit.id)
		});
		const spominChunkIds = [...selectedBlocks.values()]
			.filter((block) => block.source === 'long-term-memory')
			.map((block) => block.provenance?.chunkId)
			.filter((id): id is string => typeof id === 'string');
		const feedback = preparation.spominFeedback;
		await Promise.all([
			DatabaseService.addRetrievalTrace(preparation.trace),
			DatabaseService.putRetrievalHitUsage(
				preparation.usage.filter((record) => selected.has(record.hitId))
			),
			feedback && spominChunkIds.length
				? new SpominService(feedback)
						.recordAccess({
							eventId: `${preparation.trace.id}:injected`,
							chunkIds: spominChunkIds,
							contextId: preparation.trace.conversationId
						})
						.catch((error) => {
							memoryDebug('spomin.access.error', {
								traceId: preparation.trace.id,
								chunkCount: spominChunkIds.length,
								error
							});
							return false;
						})
				: Promise.resolve(false)
		]);
	}
}
