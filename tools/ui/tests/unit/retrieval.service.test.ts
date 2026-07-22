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
	spominTimeoutMs: 750,
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
});
