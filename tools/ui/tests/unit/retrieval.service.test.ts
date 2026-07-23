import { afterEach, describe, expect, it, vi } from 'vitest';
import { MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { RetrievalService } from '$lib/services/retrieval.service';
import type { DatabaseMessage } from '$lib/types';

const settings = {
	spominEnabled: false,
	spominBaseUrl: 'http://127.0.0.1:8084',
	spominResultLimit: 3,
	spominTokenBudget: 1000,
	spominTimeoutMs: 2000,
	embeddingBaseUrl: 'http://127.0.0.1:8081/v1',
	embeddingModel: 'embedding-test',
	embeddingTimeoutMs: 1200,
	localResultLimit: 5,
	localTokenBudget: 1500,
	totalTokenBudget: 2500,
	semanticThreshold: 0.62,
	lexicalThreshold: 0.34
};

function message(id: string, role: MessageRole, content: string): DatabaseMessage {
	return {
		id,
		convId: 'chat-1',
		type: MessageType.TEXT,
		timestamp: Date.now(),
		role,
		content,
		parent: null,
		children: []
	};
}

describe('RetrievalService', () => {
	afterEach(() => vi.restoreAllMocks());

	it('uses recent conversation context and returns matching archive blocks', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([
			{
				id: 'chunk-1',
				conversationId: 'chat-1',
				compactionId: 'compact-1',
				generation: 1,
				sourceMessageIds: ['old-1'],
				text: 'user: The database choice was SQLite for local storage.',
				terms: ['the', 'database', 'choice', 'was', 'sqlite', 'for', 'local', 'storage'],
				createdAt: Date.now(),
				embeddingStatus: 'ready'
			}
		]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'updateArchiveChunk').mockResolvedValue();

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-2',
			messages: [
				message('user-1', MessageRole.USER, 'We discussed local persistence.'),
				message('assistant-1', MessageRole.ASSISTANT, 'Yes, there was a database decision.'),
				message('user-2', MessageRole.USER, 'Which database did we choose?')
			],
			compactionGeneration: 1,
			settings
		});

		expect(result.blocks).toHaveLength(1);
		expect(result.blocks[0]).toMatchObject({
			id: 'local:chunk-1',
			source: 'conversation-recall'
		});
		expect(result.trace.query).toContain('We discussed local persistence');
		expect(result.trace.providers.spomin.status).toBe('disabled');
	});

	it('uses a specific latest request without carrying forward an unrelated topic', () => {
		const query = RetrievalService.buildQuery([
			message('user-1', MessageRole.USER, 'Tell me how my Bending Spoons interview went.'),
			message(
				'assistant-1',
				MessageRole.ASSISTANT,
				'We discussed the interview exercises and your performance at length.'
			),
			message(
				'user-2',
				MessageRole.USER,
				'Please recall my salary, debts, and the financial plan for June.'
			)
		]);

		expect(query).toContain('salary, debts, and the financial plan for June');
		expect(query).not.toContain('Bending Spoons');
		expect(query).not.toContain('interview exercises');
		expect(query).not.toContain('Recent context:');
	});

	it('expands a vague follow-up with a small recent context window', () => {
		const query = RetrievalService.buildQuery([
			message('user-1', MessageRole.USER, 'We compared SQLite and PostgreSQL.'),
			message('assistant-1', MessageRole.ASSISTANT, 'SQLite is simpler for the local application.'),
			message('user-2', MessageRole.USER, 'What about that one?')
		]);

		expect(query).toContain('Current user request:\nWhat about that one?');
		expect(query).toContain('Recent context:');
		expect(query).toContain('SQLite is simpler');
	});

	it('records only blocks that survive final prompt fitting', async () => {
		const preparation = {
			blocks: [],
			trace: {
				id: 'trace-1',
				conversationId: 'chat-1',
				anchorMessageId: 'user-2',
				createdAt: Date.now(),
				query: 'query',
				queryFingerprint: 'hash',
				queryTerms: ['query'],
				compactionGeneration: 1,
				providers: {},
				hits: [
					{
						id: 'local:chunk-1',
						source: 'conversation-recall' as const,
						score: 0.8,
						selected: true,
						tokenCount: 20
					}
				],
				injectedHitIds: [],
				injectedTokenCount: 0
			},
			usage: []
		};
		const addTrace = vi.spyOn(DatabaseService, 'addRetrievalTrace').mockResolvedValue();
		vi.spyOn(DatabaseService, 'putRetrievalHitUsage').mockResolvedValue();

		await RetrievalService.finalize(preparation, [], 1200);

		expect(addTrace.mock.calls[0][0].hits[0]).toMatchObject({
			selected: false,
			reason: 'exact-prompt-budget'
		});
	});

	it('records consecutive-request suppression for a recently injected local hit', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([
			{
				id: 'chunk-1',
				conversationId: 'chat-1',
				compactionId: 'compact-1',
				generation: 1,
				sourceMessageIds: ['old-1'],
				text: 'The local database choice was SQLite.',
				terms: ['local', 'database', 'choice', 'was', 'sqlite'],
				createdAt: Date.now(),
				embeddingStatus: 'ready'
			}
		]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([
			{
				id: 'usage-1',
				conversationId: 'chat-1',
				hitId: 'local:chunk-1',
				source: 'conversation-recall',
				lastInjectedUserTurn: 1,
				queryFingerprint: 'old-hash',
				queryTerms: ['database', 'choice', 'applies', 'locally'],
				score: 0.75,
				compactionGeneration: 1,
				updatedAt: Date.now()
			}
		]);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-2',
			messages: [
				message('user-1', MessageRole.USER, 'Which database choice applies locally?'),
				message('user-2', MessageRole.USER, 'Which database choice applies locally?')
			],
			compactionGeneration: 1,
			settings
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.hits[0]).toMatchObject({
			id: 'local:chunk-1',
			selected: false,
			reason: 'consecutive-request'
		});
	});

	it('records token-budget rejection for an otherwise eligible hit', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([
			{
				id: 'chunk-1',
				conversationId: 'chat-1',
				compactionId: 'compact-1',
				generation: 1,
				sourceMessageIds: ['old-1'],
				text: 'The local database choice was SQLite.',
				terms: ['local', 'database', 'choice', 'was', 'sqlite'],
				createdAt: Date.now(),
				embeddingStatus: 'ready'
			}
		]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'Which database choice applies locally?')],
			compactionGeneration: 1,
			settings: { ...settings, localTokenBudget: 1 }
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.hits[0]).toMatchObject({
			id: 'local:chunk-1',
			selected: false,
			reason: 'token-budget'
		});
	});

	it('applies semantic and lexical thresholds independently to Spomin results', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(
				JSON.stringify({
					query: 'database',
					semantic_available: true,
					results: [
						{
							id: 'below-thresholds',
							text: 'Do not include this result.',
							source: 'test',
							tier: 'archive',
							created_at: new Date().toISOString(),
							score: 0.9,
							semantic_score: 0.5,
							lexical_coverage: 0.1
						},
						{
							id: 'lexical-match',
							text: 'SQLite was selected for the local database.',
							source: 'test',
							tier: 'archive',
							created_at: new Date().toISOString(),
							score: 0.4,
							semantic_score: null,
							lexical_coverage: 0.4
						}
					]
				}),
				{ status: 200, headers: { 'content-type': 'application/json' } }
			)
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'Which database did we choose?')],
			compactionGeneration: 0,
			settings: { ...settings, spominEnabled: true }
		});

		expect(result.blocks).toHaveLength(1);
		expect(result.blocks[0].id).toBe('spomin:lexical-match');
		expect(result.trace.providers.spomin.detail).toBe('hybrid');
		expect(result.trace.hits).toContainEqual(
			expect.objectContaining({
				id: 'spomin:below-thresholds',
				selected: false,
				reason: 'below-threshold',
				semanticScore: 0.5,
				lexicalScore: 0.1,
				providerScore: 0.9
			})
		);
	});

	it('distinguishes a successful empty Spomin response', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(
				JSON.stringify({
					query: 'salary debts June',
					semantic_available: true,
					results: []
				}),
				{ status: 200, headers: { 'content-type': 'application/json' } }
			)
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [
				message(
					'user-1',
					MessageRole.USER,
					'Please recall my salary debts and financial plan for June.'
				)
			],
			compactionGeneration: 0,
			settings: { ...settings, spominEnabled: true }
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.providers.spomin).toEqual({ status: 'ok', detail: 'no-results' });
	});

	it('distinguishes Spomin results that all fail the configured thresholds', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(
				JSON.stringify({
					query: 'salary debts June',
					semantic_available: true,
					results: [
						{
							id: 'weak',
							text: 'An unrelated memory.',
							source: 'test',
							tier: 'archive',
							created_at: new Date().toISOString(),
							score: 0.2,
							semantic_score: 0.2,
							lexical_coverage: 0
						}
					]
				}),
				{ status: 200, headers: { 'content-type': 'application/json' } }
			)
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [
				message(
					'user-1',
					MessageRole.USER,
					'Please recall my salary debts and financial plan for June.'
				)
			],
			compactionGeneration: 0,
			settings: { ...settings, spominEnabled: true }
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.providers.spomin).toEqual({
			status: 'ok',
			detail: 'all-below-threshold'
		});
		expect(result.trace.hits[0]).toMatchObject({
			id: 'spomin:weak',
			selected: false,
			reason: 'below-threshold'
		});
	});

	it('records a Spomin timeout without blocking local recall', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockRejectedValue(
			new DOMException('The operation was aborted.', 'AbortError')
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'Recall the exact database choice.')],
			compactionGeneration: 0,
			settings: { ...settings, spominEnabled: true }
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.providers.spomin.status).toBe('timeout');
	});

	it('stores text snapshots only for blocks actually injected', async () => {
		const preparation = {
			blocks: [
				{
					id: 'spomin:memory-1',
					source: 'long-term-memory' as const,
					content: 'Injected durable memory',
					provenance: { memoryIds: ['memory-1'], project: 'llama', score: 0.9 }
				},
				{
					id: 'local:chunk-2',
					source: 'conversation-recall' as const,
					content: 'Dropped compacted excerpt',
					provenance: { chunkId: 'chunk-2', score: 0.7 }
				}
			],
			trace: {
				id: 'trace-snapshot',
				conversationId: 'chat-1',
				anchorMessageId: 'user-2',
				responseMessageId: 'assistant-2',
				createdAt: Date.now(),
				query: 'query',
				queryFingerprint: 'hash',
				queryTerms: ['query'],
				compactionGeneration: 1,
				providers: {},
				hits: [
					{
						id: 'spomin:memory-1',
						source: 'long-term-memory' as const,
						score: 0.9,
						selected: true,
						tokenCount: 4
					},
					{
						id: 'local:chunk-2',
						source: 'conversation-recall' as const,
						score: 0.7,
						selected: true,
						tokenCount: 4
					}
				],
				injectedHitIds: [],
				injectedTokenCount: 0
			},
			usage: []
		};
		const addTrace = vi.spyOn(DatabaseService, 'addRetrievalTrace').mockResolvedValue();
		vi.spyOn(DatabaseService, 'putRetrievalHitUsage').mockResolvedValue();

		await RetrievalService.finalize(preparation, ['spomin:memory-1'], 1200);

		const saved = addTrace.mock.calls[0][0];
		expect(saved.responseMessageId).toBe('assistant-2');
		expect(saved.hits[0]).toMatchObject({
			contentSnapshot: 'Injected durable memory',
			provenance: { memoryIds: ['memory-1'], project: 'llama' }
		});
		expect(saved.hits[1].contentSnapshot).toBeUndefined();
		expect(saved.hits[1].reason).toBe('exact-prompt-budget');
	});
});
