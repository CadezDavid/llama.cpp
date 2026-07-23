import { MessageRole } from '$lib/enums';
import type {
	ChatContextBlock,
	DatabaseArchiveChunk,
	DatabaseMessage,
	DatabaseRetrievalHitUsage,
	DatabaseRetrievalTrace,
	MemoryRetrievalHit,
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

export interface RetrievalPreparation {
	blocks: ChatContextBlock[];
	trace: DatabaseRetrievalTrace;
	usage: DatabaseRetrievalHitUsage[];
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
		query: string,
		queryTerms: string[],
		settings: RetrievalSettings
	): Promise<{
		candidates: Candidate[];
		evaluations: CandidateEvaluation[];
		semantic: boolean;
	}> {
		const chunks = await DatabaseService.getConversationArchiveChunks(conversationId);
		memoryDebug('retrieval.local.archive-loaded', {
			conversationId,
			chunkCount: chunks.length,
			pendingEmbeddingCount: chunks.filter((chunk) => chunk.embeddingStatus === 'pending').length,
			readyEmbeddingCount: chunks.filter((chunk) => chunk.embeddingStatus === 'ready').length
		});
		if (!chunks.length) return { candidates: [], evaluations: [], semantic: false };
		let queryEmbedding: number[] | undefined;
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
				error
			});
			queryEmbedding = undefined;
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
		return { candidates, evaluations, semantic: !!queryEmbedding };
	}

	private static evaluateSpominCandidates(
		results: MemoryRetrievalHit[],
		settings: RetrievalSettings
	): CandidateEvaluation[] {
		return results.map((result) => {
			const candidate: Candidate = {
				id: `spomin:${result.memory_ids?.[0] ?? result.id}`,
				text: result.text,
				source: 'long-term-memory',
				score: Math.max(result.semantic_score ?? 0, result.lexical_coverage ?? 0, result.score),
				lexicalScore: result.lexical_coverage ?? undefined,
				semanticScore: result.semantic_score ?? undefined,
				providerScore: result.score,
				provenance: {
					chunkId: result.id,
					memoryIds: result.memory_ids,
					project: result.project,
					tier: result.tier,
					createdAt: result.created_at
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

	static async prepare(input: {
		conversationId: string;
		anchorMessageId: string;
		responseMessageId?: string;
		messages: DatabaseMessage[];
		compactionGeneration: number;
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
			compactionGeneration: input.compactionGeneration,
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

		const localPromise = RetrievalService.localCandidates(
			input.conversationId,
			query,
			queryTerms,
			input.settings
		);
		const spominPromise = input.settings.spominEnabled
			? new SpominService({
					baseUrl: input.settings.spominBaseUrl,
					apiToken: input.settings.spominApiToken,
					timeoutMs: input.settings.spominTimeoutMs
				}).retrieve({
					query,
					limit: input.settings.spominResultLimit,
					project: input.settings.spominProject,
					excludeConversationId: input.conversationId
				})
			: Promise.resolve(null);

		const [localResult, spominResult] = await Promise.allSettled([localPromise, spominPromise]);
		let local: Candidate[] = [];
		let remote: Candidate[] = [];
		let localEvaluations: CandidateEvaluation[] = [];
		let remoteEvaluations: CandidateEvaluation[] = [];
		if (localResult.status === 'fulfilled') {
			local = localResult.value.candidates;
			localEvaluations = localResult.value.evaluations;
			providers.local = {
				status: 'ok',
				detail: localResult.value.semantic ? 'lexical+semantic' : 'lexical-only'
			};
		} else {
			providers.local = { status: 'error', detail: String(localResult.reason) };
		}
		if (!input.settings.spominEnabled) {
			providers.spomin = { status: 'disabled' };
		} else if (spominResult.status === 'fulfilled' && spominResult.value) {
			remoteEvaluations = RetrievalService.evaluateSpominCandidates(
				spominResult.value.results,
				input.settings
			);
			remote = remoteEvaluations
				.filter((evaluation) => evaluation.accepted)
				.map((evaluation) => evaluation.candidate);
			const detail = !spominResult.value.results.length
				? 'no-results'
				: !remote.length
					? 'all-below-threshold'
					: spominResult.value.semantic_available
						? 'hybrid'
						: 'keyword-only';
			providers.spomin = {
				status: 'ok',
				detail: spominResult.value.degraded_reason
					? `${detail}: ${spominResult.value.degraded_reason}`
					: detail
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
			spominCandidateCount: remote.length
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
				const generationChanged = input.compactionGeneration !== prior.compactionGeneration;
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
		const budgets = new Map([
			['conversation-recall', input.settings.localTokenBudget],
			['long-term-memory', input.settings.spominTokenBudget]
		]);
		let totalTokens = 0;
		for (const candidate of eligible.sort((left, right) => right.score - left.score)) {
			const count = tokenEstimate(candidate.text);
			const sourceBudget = budgets.get(candidate.source) ?? 0;
			if (count > sourceBudget || totalTokens + count > input.settings.totalTokenBudget) {
				const hit = traceHits.find(
					(item) => item.id === candidate.id && item.source === candidate.source
				);
				if (hit) {
					hit.selected = false;
					hit.reason = 'token-budget';
				}
				continue;
			}
			selected.push(candidate);
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
			compactionGeneration: input.compactionGeneration,
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
				compactionGeneration: input.compactionGeneration,
				updatedAt: Date.now()
			}))
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
		await Promise.all([
			DatabaseService.addRetrievalTrace(preparation.trace),
			DatabaseService.putRetrievalHitUsage(
				preparation.usage.filter((record) => selected.has(record.hitId))
			)
		]);
	}
}
