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
	provenance: Record<string, unknown>;
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
	static buildQuery(messages: DatabaseMessage[]): string {
		const conversational = messages.filter(
			(message) =>
				(message.role === MessageRole.USER || message.role === MessageRole.ASSISTANT) &&
				message.content.trim()
		);
		return conversational
			.slice(-6)
			.map((message) => `${message.role}: ${message.content}`)
			.join('\n')
			.slice(-6000);
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
		try {
			const response = await fetch(`${settings.embeddingBaseUrl.replace(/\/+$/, '')}/embeddings`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ model: settings.embeddingModel, input }),
				signal: controller.signal
			});
			if (!response.ok) throw new Error(`Embedding request failed (${response.status})`);
			const body = (await response.json()) as {
				data: Array<{ index: number; embedding: number[] }>;
			};
			return body.data
				.sort((left, right) => left.index - right.index)
				.map((item) => item.embedding);
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
	): Promise<{ candidates: Candidate[]; semantic: boolean }> {
		const chunks = await DatabaseService.getConversationArchiveChunks(conversationId);
		if (!chunks.length) return { candidates: [], semantic: false };
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
		} catch {
			queryEmbedding = undefined;
		}

		const candidates = chunks
			.map((chunk) => {
				const lexical = RetrievalService.lexicalScore(queryTerms, chunk);
				const semantic = queryEmbedding && chunk.embedding ? cosine(queryEmbedding, chunk.embedding) : 0;
				return {
					id: `local:${chunk.id}`,
					text: chunk.text,
					source: 'conversation-recall' as const,
					score: Math.max(lexical, semantic),
					lexical,
					semantic,
					provenance: {
						compactionId: chunk.compactionId,
						sourceMessageIds: chunk.sourceMessageIds,
						generation: chunk.generation
					}
				};
			})
			.filter(
				(candidate) =>
					candidate.lexical >= settings.lexicalThreshold ||
					candidate.semantic >= settings.semanticThreshold
			)
			.sort((left, right) => right.score - left.score)
			.slice(0, settings.localResultLimit);
		return { candidates, semantic: !!queryEmbedding };
	}

	private static spominCandidates(results: MemoryRetrievalHit[]): Candidate[] {
		return results.map((result) => ({
			id: `spomin:${result.memory_ids?.[0] ?? result.id}`,
			text: result.text,
			source: 'long-term-memory',
			score: Math.max(result.semantic_score ?? 0, result.lexical_coverage ?? 0, result.score),
			provenance: {
				chunkId: result.id,
				memoryIds: result.memory_ids,
				project: result.project,
				tier: result.tier,
				createdAt: result.created_at
			}
		}));
	}

	static async prepare(input: {
		conversationId: string;
		anchorMessageId: string;
		messages: DatabaseMessage[];
		compactionGeneration: number;
		settings: RetrievalSettings;
	}): Promise<RetrievalPreparation> {
		const query = RetrievalService.buildQuery(input.messages);
		const latestUser = [...input.messages]
			.reverse()
			.find((message) => message.role === MessageRole.USER)?.content;
		const queryTerms = terms(latestUser || query);
		const queryFingerprint = await RetrievalService.fingerprint(query);
		const userTurn = input.messages.filter((message) => message.role === MessageRole.USER).length;
		const explicitRecall = /\b(remember|recall|previously|before|earlier|last time)\b/i.test(
			latestUser || ''
		);
		const providers: DatabaseRetrievalTrace['providers'] = {};

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
		if (localResult.status === 'fulfilled') {
			local = localResult.value.candidates;
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
			remote = RetrievalService.spominCandidates(spominResult.value.results).filter(
				(candidate) =>
					candidate.score >=
					Math.min(input.settings.semanticThreshold, input.settings.lexicalThreshold)
			);
			providers.spomin = {
				status: 'ok',
				detail: spominResult.value.semantic_available ? 'hybrid' : 'keyword-only'
			};
		} else {
			const reason = spominResult.status === 'rejected' ? spominResult.reason : 'unavailable';
			providers.spomin = {
				status: reason instanceof DOMException && reason.name === 'AbortError' ? 'timeout' : 'error',
				detail: String(reason)
			};
		}

		const previous = await DatabaseService.getRetrievalHitUsage(input.conversationId);
		const previousById = new Map(previous.map((record) => [`${record.source}:${record.hitId}`, record]));
		const traceHits: RetrievalTraceHit[] = [];
		const eligible = [...local, ...remote].filter((candidate) => {
			const prior = previousById.get(`${candidate.source}:${candidate.id}`);
			let reason: string | undefined;
			if (prior && userTurn - prior.lastInjectedUserTurn < 3) {
				const topicChanged = similarity(queryTerms, prior.queryTerms ?? []) < 0.55;
				const generationChanged = input.compactionGeneration !== prior.compactionGeneration;
				const scoreImproved = candidate.score >= prior.score + 0.15;
				if (!topicChanged && !generationChanged && !scoreImproved && !explicitRecall) {
					reason =
						userTurn - prior.lastInjectedUserTurn <= 1 ? 'consecutive-request' : 'cooldown';
				}
			}
			traceHits.push({
				id: candidate.id,
				source: candidate.source,
				score: candidate.score,
				selected: !reason,
				reason
			});
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

		const trace: DatabaseRetrievalTrace = {
			id: uuid(),
			conversationId: input.conversationId,
			anchorMessageId: input.anchorMessageId,
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
		preparation.trace.injectedHitIds = selectedBlockIds;
		preparation.trace.injectedTokenCount = preparation.trace.hits
			.filter((hit) => selected.has(hit.id))
			.reduce((sum, hit) => sum + (hit.tokenCount ?? 0), 0);
		preparation.trace.finalPromptTokenCount = finalPromptTokenCount;
		for (const hit of preparation.trace.hits) {
			if (hit.selected && !selected.has(hit.id)) {
				hit.selected = false;
				hit.reason = 'exact-prompt-budget';
			}
		}
		await Promise.all([
			DatabaseService.addRetrievalTrace(preparation.trace),
			DatabaseService.putRetrievalHitUsage(
				preparation.usage.filter((record) => selected.has(record.hitId))
			)
		]);
	}
}
