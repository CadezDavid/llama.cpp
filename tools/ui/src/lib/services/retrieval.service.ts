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
import { sha256 } from '$lib/utils/sha256';
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
}

interface Candidate {
	id: string;
	text: string;
	source: 'conversation-recall' | 'long-term-memory';
	score: number;
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

export interface RecallSnapshot {
	traceId?: string;
	blocks: ChatContextBlock[];
}

const WORD_RE = /[\p{L}\p{N}_-]+/gu;
const CONTEXT_REFERENCE_RE =
	/\b(that one|this|those|these|the same|same one|what about|as before|earlier one|previous one)\b/i;
const MAX_SUPPORTING_CONTEXT_CHARACTERS = 1200;
const MAX_TRACE_EVALUATIONS = 20;

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

function tokenEstimate(text: string): number {
	return Math.max(1, Math.ceil((text.match(WORD_RE)?.length ?? 0) * 1.35));
}

export class RetrievalService {
	static snapshotFromTrace(trace: DatabaseRetrievalTrace): RecallSnapshot | null {
		const hits = new Map(trace.hits.map((hit) => [hit.id, hit]));
		const blocks: ChatContextBlock[] = [];
		for (const id of trace.injectedHitIds) {
			const hit = hits.get(id);
			if (!hit || typeof hit.contentSnapshot !== 'string') return null;
			blocks.push({
				id: hit.id,
				source: hit.source,
				content: hit.contentSnapshot,
				provenance: hit.provenance ? { ...hit.provenance } : undefined
			});
		}
		return {
			traceId: trace.reusedFromTraceId ?? trace.id,
			blocks
		};
	}

	static buildReusedTrace(
		sourceTrace: DatabaseRetrievalTrace,
		responseMessageId: string,
		selectedBlockIds: string[],
		finalPromptTokenCount?: number
	): DatabaseRetrievalTrace {
		const selected = new Set(selectedBlockIds);
		const previouslyInjected = new Set(sourceTrace.injectedHitIds);
		const hits = sourceTrace.hits.map((hit) => {
			const isSelected = selected.has(hit.id);
			return {
				...hit,
				selected: isSelected,
				reason: isSelected
					? undefined
					: previouslyInjected.has(hit.id)
						? 'exact-prompt-budget'
						: hit.reason,
				provenance: hit.provenance ? { ...hit.provenance } : undefined
			};
		});
		return {
			...sourceTrace,
			id: uuid(),
			responseMessageId,
			reusedFromTraceId: sourceTrace.reusedFromTraceId ?? sourceTrace.id,
			createdAt: Date.now(),
			providers: Object.fromEntries(
				Object.entries(sourceTrace.providers).map(([name, provider]) => [name, { ...provider }])
			),
			hits,
			injectedHitIds: [...selectedBlockIds],
			injectedTokenCount: hits
				.filter((hit) => selected.has(hit.id))
				.reduce((sum, hit) => sum + (hit.tokenCount ?? 0), 0),
			finalPromptTokenCount
		};
	}

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
		const contextDependent =
			(latestUser.match(WORD_RE)?.length ?? 0) < 4 || CONTEXT_REFERENCE_RE.test(latestUser);
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
			strategy: supportingContext ? 'context-expanded' : 'latest-only',
			supportingCharacterCount: supportingContext.length
		};
	}

	static buildQuery(messages: DatabaseMessage[]): string {
		return RetrievalService.buildRetrievalQuery(messages).value;
	}

	static async fingerprint(value: string): Promise<string> {
		return await sha256(value);
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
			if (
				embeddings.length !== input.length ||
				embeddings.some(
					(embedding) =>
						!embedding.length ||
						embedding.length !== embeddings[0].length ||
						embedding.some((value) => !Number.isFinite(value))
				)
			) {
				throw new Error(
					`Embedding response shape mismatch: expected ${input.length} valid vectors, received ${embeddings.length}`
				);
			}
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

	private static async localCandidates(
		conversationId: string,
		activeCompaction: ActiveCompactionScope,
		query: string,
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
		const pending = chunks.filter((chunk) => chunk.embeddingStatus === 'pending').slice(0, 15);
		const values = await RetrievalService.embeddings(
			[query, ...pending.map((chunk) => chunk.text)],
			settings
		);
		const queryEmbedding = values[0];
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

		const ranked = chunks
			.map((chunk) => {
				const semantic = chunk.embedding ? cosine(queryEmbedding, chunk.embedding) : 0;
				return {
					id: `local:${chunk.id}`,
					text: chunk.text,
					source: 'conversation-recall' as const,
					score: semantic,
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
			const aboveThreshold = (candidate.semanticScore ?? 0) >= settings.semanticThreshold;
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
				semanticScore: candidate.semanticScore,
				accepted,
				reason
			}))
		});
		return { candidates, evaluations, detail: 'semantic' };
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
				score: result.semantic_score ?? 0,
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
			const accepted = (result.semantic_score ?? 0) >= settings.semanticThreshold;
			return {
				candidate,
				accepted,
				reason: accepted ? undefined : 'below-threshold'
			};
		});
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
			fingerprint: queryFingerprint,
			budgets: {
				local: input.settings.localTokenBudget,
				spomin: input.settings.spominTokenBudget,
				total: input.settings.totalTokenBudget
			},
			thresholds: {
				semantic: input.settings.semanticThreshold
			}
		});

		const localPromise =
			input.activeCompaction && input.activeCompaction.sourceMessageIds.length > 0
				? RetrievalService.localCandidates(
						input.conversationId,
						input.activeCompaction,
						query,
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
			providers.local = {
				status:
					localResult.reason instanceof DOMException &&
					localResult.reason.name === 'AbortError'
						? 'timeout'
						: 'error',
				detail: String(localResult.reason)
			};
		}
		if (!input.settings.spominEnabled) {
			providers.spomin = { status: 'disabled' };
		} else if (spominResult.status === 'fulfilled' && spominResult.value) {
			remoteEvaluations = RetrievalService.evaluateSpominCandidates(
				spominResult.value.semantic.results,
				input.settings,
				spominResult.value.profile
			);
			remote = remoteEvaluations
				.filter((evaluation) => evaluation.accepted)
				.map((evaluation) => evaluation.candidate);
			const detail = !spominResult.value.semantic.results.length
				? 'no-results'
				: !remote.length
					? 'all-below-threshold'
					: 'semantic';
			const channelSummary = [
				detail,
				`semantic=${spominResult.value.semantic.results.length}`,
				spominResult.value.profile ? `profile=${spominResult.value.profile.id.slice(0, 12)}` : null
			]
				.filter(Boolean)
				.join('; ');
			providers.spomin = {
				status: 'ok',
				detail: channelSummary
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
				const topicChanged = queryFingerprint !== prior.queryFingerprint;
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
