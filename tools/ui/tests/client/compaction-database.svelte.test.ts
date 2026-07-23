import { afterEach, describe, expect, it, vi } from 'vitest';
import { MessageRole, MessageType } from '$lib/enums';
import { ChatContextService } from '$lib/services/chat-context.service';
import { DatabaseService } from '$lib/services/database.service';

describe('compaction database persistence', () => {
	let conversationId: string | null = null;

	afterEach(async () => {
		if (conversationId) {
			await DatabaseService.deleteConversation(conversationId, { deleteWithForks: true });
		}
		conversationId = null;
		vi.restoreAllMocks();
	});

	it('stores plain copies of reactive source message arrays', async () => {
		const conversation = await DatabaseService.createConversation('Reactive source test');
		conversationId = conversation.id;
		const sourceMessageIds = new Proxy(['message-1'], {});
		const deltaSourceMessageIds = new Proxy(['message-1'], {});

		const pending = await DatabaseService.createPendingCompaction({
			conversationId: conversation.id,
			sourceMessageIds,
			deltaSourceMessageIds,
			sourceFingerprint: 'fingerprint',
			summary: '',
			summarySchemaVersion: 1,
			promptVersion: 1,
			sourceTokenCount: 1000,
			beforeTokenCount: 2000,
			projectedTokenCount: 0,
			generation: 1
		});

		expect(pending.sourceMessageIds).toEqual(['message-1']);
		expect(pending.deltaSourceMessageIds).toEqual(['message-1']);
		expect(await DatabaseService.getConversationCompactions(conversation.id)).toHaveLength(1);
	});

	it('keeps pending records inactive and commits ready state with its projection event', async () => {
		const conversation = await DatabaseService.createConversation('Compaction test');
		conversationId = conversation.id;
		const rootId = await DatabaseService.createRootMessage(conversation.id);
		const user = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now(),
				role: MessageRole.USER,
				content: 'Old question',
				parent: rootId,
				children: []
			},
			rootId
		);
		const assistant = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now() + 1,
				role: MessageRole.ASSISTANT,
				content: 'Old answer',
				parent: user.id,
				children: []
			},
			user.id
		);
		const pending = await DatabaseService.createPendingCompaction({
			conversationId: conversation.id,
			sourceMessageIds: [user.id, assistant.id],
			deltaSourceMessageIds: [user.id, assistant.id],
			sourceFingerprint: await ChatContextService.fingerprintMessages([user, assistant]),
			summary: '',
			summarySchemaVersion: 1,
			promptVersion: 1,
			sourceTokenCount: 1000,
			beforeTokenCount: 2000,
			projectedTokenCount: 0,
			generation: 1
		});

		expect(
			await DatabaseService.getConversationCompactionProjectionEvents(conversation.id)
		).toEqual([]);

		const event = await DatabaseService.activateCompaction(
			pending.id,
			{
				summary: 'ready',
				projectedTokenCount: 800,
				activationMode: 'automatic',
				attemptCount: 2,
				strictRetryUsed: true
			},
			assistant.id
		);
		const records = await DatabaseService.getConversationCompactions(conversation.id);

		expect(records).toHaveLength(1);
		expect(records[0]).toMatchObject({
			status: 'ready',
			summary: 'ready',
			activationMode: 'automatic',
			attemptCount: 2,
			strictRetryUsed: true
		});
		expect(event).toMatchObject({ action: 'apply', compactionId: pending.id });
		const archive = await DatabaseService.getConversationArchiveChunks(conversation.id);
		expect(archive).toHaveLength(2);
		expect(new Set(archive.map((chunk) => chunk.sourceMessageIds[0]))).toEqual(
			new Set([user.id, assistant.id])
		);
		expect(archive.every((chunk) => chunk.embeddingStatus === 'pending')).toBe(true);
	});

	it('keeps activation alive while Web Crypto fingerprints the source', async () => {
		const conversation = await DatabaseService.createConversation('Delayed fingerprint test');
		conversationId = conversation.id;
		const rootId = await DatabaseService.createRootMessage(conversation.id);
		const user = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now(),
				role: MessageRole.USER,
				content: 'Old question',
				parent: rootId,
				children: []
			},
			rootId
		);
		const assistant = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now() + 1,
				role: MessageRole.ASSISTANT,
				content: 'Old answer',
				parent: user.id,
				children: []
			},
			user.id
		);
		const sourceFingerprint = await ChatContextService.fingerprintMessages([user, assistant]);
		const pending = await DatabaseService.createPendingCompaction({
			conversationId: conversation.id,
			sourceMessageIds: [user.id, assistant.id],
			deltaSourceMessageIds: [user.id, assistant.id],
			sourceFingerprint,
			summary: '',
			summarySchemaVersion: 1,
			promptVersion: 1,
			sourceTokenCount: 1000,
			beforeTokenCount: 2000,
			projectedTokenCount: 0,
			generation: 1
		});
		const fingerprintMessages = ChatContextService.fingerprintMessages.bind(ChatContextService);
		vi.spyOn(ChatContextService, 'fingerprintMessages').mockImplementation(async (messages) => {
			await new Promise((resolve) => setTimeout(resolve, 25));
			return await fingerprintMessages(messages);
		});

		await expect(
			DatabaseService.activateCompaction(
				pending.id,
				{ summary: 'ready', projectedTokenCount: 800 },
				assistant.id
			)
		).resolves.toMatchObject({ action: 'apply', compactionId: pending.id });
		const records = await DatabaseService.getConversationCompactions(conversation.id);
		expect(records[0]).toMatchObject({ status: 'ready', summary: 'ready' });
	});

	it('records restore as a later path projection event', async () => {
		const conversation = await DatabaseService.createConversation('Restore test');
		conversationId = conversation.id;
		const rootId = await DatabaseService.createRootMessage(conversation.id);

		const event = await DatabaseService.createCompactionRestoreEvent(conversation.id, rootId);

		expect(event).toMatchObject({ action: 'restore', anchorMessageId: rootId });
	});

	it('does not activate a pending record when its anchor is invalid', async () => {
		const conversation = await DatabaseService.createConversation('Invalid activation test');
		conversationId = conversation.id;
		const pending = await DatabaseService.createPendingCompaction({
			conversationId: conversation.id,
			sourceMessageIds: [],
			deltaSourceMessageIds: [],
			sourceFingerprint: 'fingerprint',
			summary: '',
			summarySchemaVersion: 1,
			promptVersion: 1,
			sourceTokenCount: 1000,
			beforeTokenCount: 2000,
			projectedTokenCount: 0,
			generation: 1
		});

		await expect(
			DatabaseService.activateCompaction(
				pending.id,
				{ summary: 'must not apply', projectedTokenCount: 800 },
				'missing-anchor'
			)
		).rejects.toThrow('is not in the conversation');

		const records = await DatabaseService.getConversationCompactions(conversation.id);
		expect(records[0].status).toBe('pending');
		expect(
			await DatabaseService.getConversationCompactionProjectionEvents(conversation.id)
		).toEqual([]);
	});

	it('does not activate a pending record after its source messages change', async () => {
		const conversation = await DatabaseService.createConversation('Stale source test');
		conversationId = conversation.id;
		const rootId = await DatabaseService.createRootMessage(conversation.id);
		const user = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now(),
				role: MessageRole.USER,
				content: 'Original question',
				parent: rootId,
				children: []
			},
			rootId
		);
		const assistant = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now() + 1,
				role: MessageRole.ASSISTANT,
				content: 'Original answer',
				parent: user.id,
				children: []
			},
			user.id
		);
		const pending = await DatabaseService.createPendingCompaction({
			conversationId: conversation.id,
			sourceMessageIds: [user.id, assistant.id],
			deltaSourceMessageIds: [user.id, assistant.id],
			sourceFingerprint: await ChatContextService.fingerprintMessages([user, assistant]),
			summary: '',
			summarySchemaVersion: 1,
			promptVersion: 1,
			sourceTokenCount: 1000,
			beforeTokenCount: 2000,
			projectedTokenCount: 0,
			generation: 1
		});

		await DatabaseService.updateMessage(user.id, { content: 'Edited question' });
		await expect(
			DatabaseService.activateCompaction(
				pending.id,
				{ summary: 'must not apply', projectedTokenCount: 800 },
				assistant.id
			)
		).rejects.toThrow('source messages changed before activation');

		const records = await DatabaseService.getConversationCompactions(conversation.id);
		expect(records[0].status).toBe('pending');
		expect(
			await DatabaseService.getConversationCompactionProjectionEvents(conversation.id)
		).toEqual([]);
	});

	it('copies applicable ready compactions when a conversation is forked', async () => {
		const conversation = await DatabaseService.createConversation('Fork source');
		conversationId = conversation.id;
		const rootId = await DatabaseService.createRootMessage(conversation.id);
		const user = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now(),
				role: MessageRole.USER,
				content: 'Old question',
				parent: rootId,
				children: []
			},
			rootId
		);
		const assistant = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now() + 1,
				role: MessageRole.ASSISTANT,
				content: 'Old answer',
				parent: user.id,
				children: []
			},
			user.id
		);
		const pending = await DatabaseService.createPendingCompaction({
			conversationId: conversation.id,
			sourceMessageIds: [user.id, assistant.id],
			deltaSourceMessageIds: [user.id, assistant.id],
			sourceFingerprint: await ChatContextService.fingerprintMessages([user, assistant]),
			summary: '',
			summarySchemaVersion: 1,
			promptVersion: 1,
			sourceTokenCount: 1000,
			beforeTokenCount: 2000,
			projectedTokenCount: 0,
			generation: 1
		});
		await DatabaseService.activateCompaction(
			pending.id,
			{ summary: 'ready', projectedTokenCount: 800 },
			assistant.id
		);
		const sourceChunk = (await DatabaseService.getConversationArchiveChunks(conversation.id))[0];
		await DatabaseService.addRetrievalTrace({
			id: 'source-trace',
			conversationId: conversation.id,
			anchorMessageId: user.id,
			responseMessageId: assistant.id,
			createdAt: Date.now(),
			query: 'old question',
			queryFingerprint: 'fingerprint',
			queryTerms: ['old', 'question'],
			compactionGeneration: 1,
			providers: { local: { status: 'ok' } },
			hits: [
				{
					id: `local:${sourceChunk.id}`,
					source: 'conversation-recall',
					score: 0.9,
					selected: true,
					tokenCount: 4,
					contentSnapshot: sourceChunk.text,
					provenance: {
						chunkId: sourceChunk.id,
						compactionId: pending.id,
						sourceMessageIds: sourceChunk.sourceMessageIds
					}
				}
			],
			injectedHitIds: [`local:${sourceChunk.id}`],
			injectedTokenCount: 4
		});

		const fork = await DatabaseService.forkConversation(conversation.id, assistant.id, {
			name: 'Fork target',
			includeAttachments: true
		});
		const forkMessages = await DatabaseService.getConversationMessages(fork.id);
		const forkCompactions = await DatabaseService.getConversationCompactions(fork.id);
		const forkEvents = await DatabaseService.getConversationCompactionProjectionEvents(fork.id);
		const forkTraces = await DatabaseService.getRetrievalTraces(fork.id);

		expect(forkCompactions).toHaveLength(1);
		expect(forkCompactions[0]).toMatchObject({
			conversationId: fork.id,
			status: 'ready',
			summary: 'ready'
		});
		expect(
			forkCompactions[0].sourceMessageIds.every((id) => forkMessages.some((m) => m.id === id))
		).toBe(true);
		expect(forkEvents).toHaveLength(1);
		expect(forkEvents[0]).toMatchObject({
			conversationId: fork.id,
			action: 'apply',
			compactionId: forkCompactions[0].id
		});
		expect(forkTraces).toHaveLength(1);
		expect(forkTraces[0]).toMatchObject({
			conversationId: fork.id,
			hits: [
				expect.objectContaining({
					contentSnapshot: sourceChunk.text,
					provenance: expect.objectContaining({ compactionId: forkCompactions[0].id })
				})
			]
		});
		expect(forkMessages.some((message) => message.id === forkTraces[0].anchorMessageId)).toBe(true);
		expect(forkMessages.some((message) => message.id === forkTraces[0].responseMessageId)).toBe(
			true
		);
		expect(forkTraces[0].hits[0].id).not.toBe(`local:${sourceChunk.id}`);
	});

	it('deletes request traces with their response branch', async () => {
		const conversation = await DatabaseService.createConversation('Trace deletion');
		conversationId = conversation.id;
		const rootId = await DatabaseService.createRootMessage(conversation.id);
		const user = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now(),
				role: MessageRole.USER,
				content: 'Question',
				parent: rootId,
				children: []
			},
			rootId
		);
		const assistant = await DatabaseService.createMessageBranch(
			{
				convId: conversation.id,
				type: MessageType.TEXT,
				timestamp: Date.now() + 1,
				role: MessageRole.ASSISTANT,
				content: 'Answer',
				parent: user.id,
				children: []
			},
			user.id
		);
		await DatabaseService.addRetrievalTrace({
			id: 'trace-delete',
			conversationId: conversation.id,
			anchorMessageId: user.id,
			responseMessageId: assistant.id,
			createdAt: Date.now(),
			query: 'question',
			queryFingerprint: 'fingerprint',
			queryTerms: ['question'],
			compactionGeneration: 0,
			providers: { local: { status: 'ok' } },
			hits: [],
			injectedHitIds: [],
			injectedTokenCount: 0
		});

		await DatabaseService.deleteMessageCascading(conversation.id, assistant.id);

		expect(await DatabaseService.getRetrievalTraces(conversation.id)).toEqual([]);
	});
});
