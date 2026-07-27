import { afterEach, describe, expect, it, vi } from 'vitest';
import { MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { RetrievalService } from '$lib/services/retrieval.service';
import type { DatabaseMessage, DatabaseRetrievalTrace } from '$lib/types';

const settings = {
	spominEnabled: false,
	spominBaseUrl: 'http://127.0.0.1:8084',
	spominCandidateLimit: 20,
	spominResultLimit: 3,
	spominTokenBudget: 1000,
	spominTimeoutMs: 2000,
	embeddingBaseUrl: 'http://127.0.0.1:8081/v1',
	embeddingModel: 'embedding-test',
	embeddingTimeoutMs: 1200,
	localResultLimit: 5,
	localTokenBudget: 1500,
	totalTokenBudget: 2500,
	semanticThreshold: 0.62
};

const activeCompaction = {
	id: 'compact-1',
	generation: 1,
	sourceMessageIds: ['old-1']
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

function retrievalTrace(overrides: Partial<DatabaseRetrievalTrace> = {}): DatabaseRetrievalTrace {
	return {
		id: 'trace-1',
		conversationId: 'chat-1',
		anchorMessageId: 'user-1',
		responseMessageId: 'assistant-1',
		createdAt: 1,
		query: 'current request',
		queryFingerprint: 'fingerprint',
		compactionGeneration: 1,
		providers: {
			local: { status: 'ok', detail: 'semantic' },
			spomin: { status: 'ok', detail: 'semantic' }
		},
		hits: [],
		injectedHitIds: [],
		injectedTokenCount: 0,
		...overrides
	};
}

describe('RetrievalService', () => {
	afterEach(() => vi.restoreAllMocks());

	it('reconstructs an immutable recall snapshot in injected order', () => {
		const trace = retrievalTrace({
			reusedFromTraceId: 'root-trace',
			hits: [
				{
					id: 'memory-1',
					source: 'long-term-memory',
					score: 0.9,
					selected: true,
					contentSnapshot: 'First memory',
					provenance: { chunkId: 'chunk-1' }
				},
				{
					id: 'memory-2',
					source: 'conversation-recall',
					score: 0.8,
					selected: true,
					contentSnapshot: 'Second memory'
				}
			],
			injectedHitIds: ['memory-2', 'memory-1']
		});

		expect(RetrievalService.snapshotFromTrace(trace)).toEqual({
			traceId: 'root-trace',
			blocks: [
				{
					id: 'memory-2',
					source: 'conversation-recall',
					content: 'Second memory',
					provenance: undefined
				},
				{
					id: 'memory-1',
					source: 'long-term-memory',
					content: 'First memory',
					provenance: { chunkId: 'chunk-1' }
				}
			]
		});
	});

	it('reuses an empty result but rejects legacy injected hits without snapshots', () => {
		expect(RetrievalService.snapshotFromTrace(retrievalTrace())).toEqual({
			traceId: 'trace-1',
			blocks: []
		});
		expect(
			RetrievalService.snapshotFromTrace(
				retrievalTrace({
					hits: [
						{
							id: 'legacy',
							source: 'conversation-recall',
							score: 0.8,
							selected: true
						}
					],
					injectedHitIds: ['legacy']
				})
			)
		).toBeNull();
	});

	it('builds a diagnostic trace without recording a new provider selection', () => {
		const source = retrievalTrace({
			hits: [
				{
					id: 'memory-1',
					source: 'long-term-memory',
					score: 0.9,
					selected: true,
					tokenCount: 20,
					contentSnapshot: 'First memory'
				},
				{
					id: 'memory-2',
					source: 'conversation-recall',
					score: 0.8,
					selected: true,
					tokenCount: 30,
					contentSnapshot: 'Second memory'
				},
				{
					id: 'skipped',
					source: 'conversation-recall',
					score: 0.2,
					selected: false,
					reason: 'below-threshold'
				}
			],
			injectedHitIds: ['memory-1', 'memory-2'],
			injectedTokenCount: 50
		});

		const reused = RetrievalService.buildReusedTrace(source, 'assistant-2', ['memory-1'], 1234);

		expect(reused).toMatchObject({
			conversationId: 'chat-1',
			responseMessageId: 'assistant-2',
			reusedFromTraceId: 'trace-1',
			injectedHitIds: ['memory-1'],
			injectedTokenCount: 20,
			finalPromptTokenCount: 1234
		});
		expect(reused.id).not.toBe(source.id);
		expect(reused.hits).toEqual([
			expect.objectContaining({ id: 'memory-1', selected: true, reason: undefined }),
			expect.objectContaining({
				id: 'memory-2',
				selected: false,
				reason: 'exact-prompt-budget'
			}),
			expect.objectContaining({ id: 'skipped', selected: false, reason: 'below-threshold' })
		]);
	});

	it('does not inspect retained archives or call embeddings without an active compaction', async () => {
		const archiveLookup = vi
			.spyOn(DatabaseService, 'getConversationArchiveChunks')
			.mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		const fetchRequest = vi.spyOn(globalThis, 'fetch');

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'What did we decide earlier?')],
			settings
		});

		expect(archiveLookup).not.toHaveBeenCalled();
		expect(fetchRequest).not.toHaveBeenCalled();
		expect(result.trace.providers.local).toEqual({
			status: 'not-applicable',
			detail: 'conversation is not compacted'
		});
	});

	it('reports an active compaction with no archived fragments without calling embeddings', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		const fetchRequest = vi.spyOn(globalThis, 'fetch');

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'What did we decide earlier?')],
			activeCompaction,
			settings
		});

		expect(fetchRequest).not.toHaveBeenCalled();
		expect(result.trace.providers.local).toEqual({
			status: 'ok',
			detail: 'no archived fragments'
		});
	});

	it('reports an embedding timeout without injecting local recall', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([
			{
				id: 'chunk-1',
				conversationId: 'chat-1',
				compactionId: 'compact-1',
				generation: 1,
				sourceMessageIds: ['old-1'],
				text: 'user: The database choice was SQLite.',
				terms: ['the', 'database', 'choice', 'was', 'sqlite'],
				createdAt: 1,
				embeddingStatus: 'ready'
			}
		]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockRejectedValue(
			new DOMException('The operation was aborted.', 'AbortError')
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'Which database did we choose?')],
			activeCompaction,
			settings
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.providers.local).toEqual({
			status: 'timeout',
			detail: 'AbortError: The operation was aborted.'
		});
	});

	it('searches the complete active compacted range across generations and ignores stale archives', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([
			{
				id: 'old-duplicate',
				conversationId: 'chat-1',
				compactionId: 'compact-1',
				generation: 1,
				sourceMessageIds: ['old-1'],
				text: 'user: The database choice was SQLite.',
				terms: ['the', 'database', 'choice', 'was', 'sqlite'],
				createdAt: 1,
				embedding: [1, 0],
				embeddingStatus: 'ready'
			},
			{
				id: 'old-current',
				conversationId: 'chat-1',
				compactionId: 'compact-2',
				generation: 2,
				sourceMessageIds: ['old-1'],
				text: 'user: The database choice was SQLite.',
				terms: ['the', 'database', 'choice', 'was', 'sqlite'],
				createdAt: 2,
				embedding: [1, 0],
				embeddingStatus: 'ready'
			},
			{
				id: 'new-delta',
				conversationId: 'chat-1',
				compactionId: 'compact-2',
				generation: 2,
				sourceMessageIds: ['old-2'],
				text: 'assistant: SQLite keeps local persistence simple.',
				terms: ['sqlite', 'keeps', 'local', 'persistence', 'simple'],
				createdAt: 2,
				embedding: [1, 0],
				embeddingStatus: 'ready'
			},
			{
				id: 'restored-stale',
				conversationId: 'chat-1',
				compactionId: 'compact-old',
				generation: 1,
				sourceMessageIds: ['restored-1'],
				text: 'user: This restored section must never be recalled.',
				terms: ['this', 'restored', 'section', 'must', 'never', 'be', 'recalled'],
				createdAt: 3,
				embedding: [1, 0],
				embeddingStatus: 'ready'
			}
		]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(
				JSON.stringify({
					data: [{ index: 0, embedding: [1, 0] }]
				}),
				{ status: 200, headers: { 'content-type': 'application/json' } }
			)
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'Which SQLite choice did we make?')],
			activeCompaction: {
				id: 'compact-2',
				generation: 2,
				sourceMessageIds: ['old-1', 'old-2']
			},
			settings
		});

		expect(result.trace.hits.map((hit) => hit.id)).toEqual(
			expect.arrayContaining(['local:old-current', 'local:new-delta'])
		);
		expect(result.trace.hits.map((hit) => hit.id)).not.toContain('local:old-duplicate');
		expect(result.trace.hits.map((hit) => hit.id)).not.toContain('local:restored-stale');
		expect(result.trace.providers.local).toEqual({
			status: 'ok',
			detail: 'semantic'
		});
	});

	it('returns archive blocks from semantic similarity', async () => {
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
				embedding: [1, 0],
				embeddingStatus: 'ready'
			}
		]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'updateArchiveChunk').mockResolvedValue();
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 0] }] }), {
				status: 200,
				headers: { 'content-type': 'application/json' }
			})
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-2',
			messages: [
				message('user-1', MessageRole.USER, 'We discussed local persistence.'),
				message('assistant-1', MessageRole.ASSISTANT, 'Yes, there was a database decision.'),
				message('user-2', MessageRole.USER, 'Which database did we choose?')
			],
			activeCompaction,
			settings
		});

		expect(result.blocks).toHaveLength(1);
		expect(result.blocks[0]).toMatchObject({
			id: 'local:chunk-1',
			source: 'conversation-recall'
		});
		expect(result.trace.query).toBe('Current user request:\nWhich database did we choose?');
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
				embedding: [1, 0],
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
				queryFingerprint: await RetrievalService.fingerprint(
					'Current user request:\nWhich database choice applies locally?'
				),
				score: 0.9,
				compactionGeneration: 1,
				updatedAt: Date.now()
			}
		]);
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 0] }] }), {
				status: 200,
				headers: { 'content-type': 'application/json' }
			})
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-2',
			messages: [
				message('user-1', MessageRole.USER, 'Which database choice applies locally?'),
				message('user-2', MessageRole.USER, 'Which database choice applies locally?')
			],
			activeCompaction,
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
				embedding: [1, 0],
				embeddingStatus: 'ready'
			}
		]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 0] }] }), {
				status: 200,
				headers: { 'content-type': 'application/json' }
			})
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'Which database choice applies locally?')],
			activeCompaction,
			settings: { ...settings, localTokenBudget: 1 }
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.hits[0]).toMatchObject({
			id: 'local:chunk-1',
			selected: false,
			reason: 'token-budget'
		});
	});

	it('applies the semantic threshold to Spomin results', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(
				JSON.stringify({
					query: 'database',
					profile: { id: 'profile-1', model_id: 'embedding-test' },
					semantic: {
						status: 'complete',
						results: [
							{
								id: 'below-thresholds',
								text: 'Do not include this result.',
								source: 'test',
								tier: 'archive',
								created_at: new Date().toISOString(),
								semantic_score: 0.5,
								memory_ids: ['below-thresholds']
							},
							{
								id: 'semantic-match',
								text: 'SQLite was selected for the local database.',
								source: 'test',
								tier: 'archive',
								created_at: new Date().toISOString(),
								semantic_score: 0.8,
								memory_ids: ['semantic-match']
							}
						]
					}
				}),
				{ status: 200, headers: { 'content-type': 'application/json' } }
			)
		);

		const result = await RetrievalService.prepare({
			conversationId: 'chat-1',
			anchorMessageId: 'user-1',
			messages: [message('user-1', MessageRole.USER, 'Which database did we choose?')],
			settings: { ...settings, spominEnabled: true }
		});

		expect(result.blocks).toHaveLength(1);
		expect(result.blocks[0].id).toBe('spomin:semantic-match');
		expect(result.trace.providers.spomin.detail).toBe(
			'semantic; semantic=2; profile=profile-1'
		);
		expect(result.trace.hits).toContainEqual(
			expect.objectContaining({
				id: 'spomin:below-thresholds',
				selected: false,
				reason: 'below-threshold',
				semanticScore: 0.5
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
					profile: null,
					semantic: { status: 'complete', results: [] }
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
			settings: { ...settings, spominEnabled: true }
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.providers.spomin).toEqual({
			status: 'ok',
			detail: 'no-results; semantic=0'
		});
	});

	it('distinguishes Spomin results that all fail the configured thresholds', async () => {
		vi.spyOn(DatabaseService, 'getConversationArchiveChunks').mockResolvedValue([]);
		vi.spyOn(DatabaseService, 'getRetrievalHitUsage').mockResolvedValue([]);
		vi.spyOn(globalThis, 'fetch').mockResolvedValue(
			new Response(
				JSON.stringify({
					query: 'salary debts June',
					profile: null,
					semantic: {
						status: 'complete',
						results: [
							{
								id: 'weak',
								text: 'An unrelated memory.',
								source: 'test',
								tier: 'archive',
								created_at: new Date().toISOString(),
								semantic_score: 0.2,
								memory_ids: ['weak']
							}
						]
					}
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
			settings: { ...settings, spominEnabled: true }
		});

		expect(result.blocks).toEqual([]);
		expect(result.trace.providers.spomin).toEqual({
			status: 'ok',
			detail: 'all-below-threshold; semantic=1'
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
					provenance: {
						chunkId: 'chunk-1',
						memoryIds: ['memory-1'],
						project: 'llama',
						score: 0.9
					}
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
			usage: [],
			spominFeedback: {
				baseUrl: 'http://127.0.0.1:8084',
				timeoutMs: 2000
			}
		};
		const addTrace = vi.spyOn(DatabaseService, 'addRetrievalTrace').mockResolvedValue();
		vi.spyOn(DatabaseService, 'putRetrievalHitUsage').mockResolvedValue();
		const fetchRequest = vi
			.spyOn(globalThis, 'fetch')
			.mockResolvedValue(new Response(JSON.stringify({ recorded: true }), { status: 200 }));

		await RetrievalService.finalize(preparation, ['spomin:memory-1'], 1200);

		const saved = addTrace.mock.calls[0][0];
		expect(saved.responseMessageId).toBe('assistant-2');
		expect(saved.hits[0]).toMatchObject({
			contentSnapshot: 'Injected durable memory',
			provenance: { memoryIds: ['memory-1'], project: 'llama' }
		});
		expect(saved.hits[1].contentSnapshot).toBeUndefined();
		expect(saved.hits[1].reason).toBe('exact-prompt-budget');
		expect(fetchRequest).toHaveBeenCalledWith(
			'http://127.0.0.1:8084/v2/memories/access',
			expect.objectContaining({
				method: 'POST',
				body: JSON.stringify({
					event_id: 'trace-snapshot:injected',
					client: 'llama.cpp-webui',
					event_type: 'injected',
					context_id: 'chat-1',
					chunk_ids: ['chunk-1']
				})
			})
		);
	});
});
