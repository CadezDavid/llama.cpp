import { afterEach, describe, expect, it, vi } from 'vitest';
import { MessageRole, MessageType } from '$lib/enums';
import { CompactionService } from '$lib/services/compaction.service';
import { DatabaseService } from '$lib/services/database.service';
import {
	RetrievalService,
	type RecallSnapshot,
	type RetrievalPreparation
} from '$lib/services/retrieval.service';
import { chatStore } from '$lib/stores/chat.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import type { ChatContextBlock, DatabaseMessage, PreparedChatContext } from '$lib/types';

type PrepareConversationContext = (
	conversationId: string,
	messages: DatabaseMessage[],
	responseMessageId?: string,
	model?: string | null,
	excludeReasoning?: boolean,
	measurementOptions?: undefined,
	includeRecall?: boolean,
	recallSnapshot?: RecallSnapshot,
	agenticTurn?: number
) => Promise<PreparedChatContext & { recallSnapshot: RecallSnapshot }>;

function message(
	id: string,
	role: MessageRole,
	content: string,
	overrides: Partial<DatabaseMessage> = {}
): DatabaseMessage {
	return {
		id,
		convId: 'conversation-1',
		type: MessageType.TEXT,
		timestamp: Number(id.replace(/\D/g, '')) || 1,
		role,
		content,
		parent: null,
		children: [],
		...overrides
	};
}

function preparation(blocks: ChatContextBlock[]): RetrievalPreparation {
	return {
		blocks,
		trace: {
			id: 'trace-1',
			conversationId: 'conversation-1',
			anchorMessageId: 'user-1',
			responseMessageId: 'assistant-1',
			createdAt: Date.now(),
			query: 'Which database does the project use?',
			queryFingerprint: 'fingerprint',
			queryTerms: ['database', 'project'],
			compactionGeneration: 0,
			providers: {
				local: { status: 'not-applicable' },
				spomin: { status: 'ok' }
			},
			hits: [],
			injectedHitIds: [],
			injectedTokenCount: 0
		},
		usage: []
	};
}

describe('chat recall snapshot', () => {
	const originalSpominEnabled = settingsStore.config.spominEnabled;

	afterEach(() => {
		settingsStore.updateConfig('spominEnabled', originalSpominEnabled);
		vi.restoreAllMocks();
	});

	it('retrieves once and reuses the same blocks with later tool results', async () => {
		settingsStore.updateConfig('spominEnabled', true);
		vi.spyOn(CompactionService, 'resolveActiveCompaction').mockResolvedValue(null);
		vi.spyOn(DatabaseService, 'getConversation').mockResolvedValue(undefined);
		const blocks: ChatContextBlock[] = [
			{
				id: 'spomin:memory-1',
				source: 'long-term-memory',
				content: 'The project uses SQLite.'
			}
		];
		const retrieve = vi.spyOn(RetrievalService, 'prepare').mockResolvedValue(preparation(blocks));
		const finalize = vi.spyOn(RetrievalService, 'finalize').mockResolvedValue();
		const prepareContext = Reflect.get(chatStore, 'prepareConversationContext').bind(
			chatStore
		) as PrepareConversationContext;
		const firstMessages = [
			message('user-1', MessageRole.USER, 'Which database does the project use?')
		];

		const first = await prepareContext('conversation-1', firstMessages, 'assistant-1');
		const later = await prepareContext(
			'conversation-1',
			[
				...firstMessages,
				message('assistant-1', MessageRole.ASSISTANT, '', {
					toolCalls: JSON.stringify([
						{
							id: 'call-1',
							type: 'function',
							function: { name: 'inspect_project', arguments: '{}' }
						}
					])
				}),
				message('tool-1', MessageRole.TOOL, 'The dependency is present.', {
					toolCallId: 'call-1'
				})
			],
			'assistant-2',
			undefined,
			false,
			undefined,
			true,
			first.recallSnapshot,
			1
		);

		expect(retrieve).toHaveBeenCalledOnce();
		expect(finalize).toHaveBeenCalledOnce();
		expect(first.recallSnapshot).toEqual({
			traceId: 'trace-1',
			blocks
		});
		expect(later.contextBlockIds).toEqual(['spomin:memory-1']);
		expect(later.requestMessages[0].content).toEqual(first.requestMessages[0].content);
		expect(later.requestMessages.at(-1)).toMatchObject({
			role: MessageRole.TOOL,
			content: 'The dependency is present.',
			tool_call_id: 'call-1'
		});
	});

	it('does not retry a failed retrieval during the same generation', async () => {
		settingsStore.updateConfig('spominEnabled', true);
		vi.spyOn(CompactionService, 'resolveActiveCompaction').mockResolvedValue(null);
		vi.spyOn(DatabaseService, 'getConversation').mockResolvedValue(undefined);
		const retrieve = vi
			.spyOn(RetrievalService, 'prepare')
			.mockRejectedValue(new Error('provider unavailable'));
		vi.spyOn(console, 'warn').mockImplementation(() => {});
		const prepareContext = Reflect.get(chatStore, 'prepareConversationContext').bind(
			chatStore
		) as PrepareConversationContext;
		const messages = [message('user-1', MessageRole.USER, 'Which database does the project use?')];

		const first = await prepareContext('conversation-1', messages, 'assistant-1');
		const later = await prepareContext(
			'conversation-1',
			messages,
			'assistant-2',
			undefined,
			false,
			undefined,
			true,
			first.recallSnapshot,
			1
		);

		expect(retrieve).toHaveBeenCalledOnce();
		expect(first.recallSnapshot).toEqual({ blocks: [] });
		expect(later.contextBlockIds).toEqual([]);
	});
});
