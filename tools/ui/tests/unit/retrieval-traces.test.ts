import { describe, expect, it } from 'vitest';
import { MessageRole, MessageType } from '$lib/enums';
import { visibleTraceMessageId } from '$lib/utils/retrieval-traces';
import type { DatabaseMessage, DatabaseRetrievalTrace } from '$lib/types';

function message(id: string, role: MessageRole): DatabaseMessage {
	return {
		id,
		convId: 'chat-1',
		type: MessageType.TEXT,
		timestamp: 1,
		role,
		content: '',
		parent: null,
		children: []
	};
}

function trace(overrides: Partial<DatabaseRetrievalTrace>): DatabaseRetrievalTrace {
	return {
		id: 'trace-1',
		conversationId: 'chat-1',
		anchorMessageId: 'user-1',
		createdAt: 1,
		query: '',
		queryFingerprint: '',
		queryTerms: [],
		compactionGeneration: 0,
		providers: {},
		hits: [],
		injectedHitIds: [],
		injectedTokenCount: 0,
		...overrides
	};
}

describe('visibleTraceMessageId', () => {
	const visibleMessages = [
		message('user-1', MessageRole.USER),
		message('assistant-active', MessageRole.ASSISTANT)
	];
	const visibleIds = new Set(visibleMessages.map((item) => item.id));

	it('keeps a trace attached to its exact visible response', () => {
		expect(
			visibleTraceMessageId(trace({ responseMessageId: 'assistant-active' }), visibleIds)
		).toBe('assistant-active');
	});

	it('does not move a hidden sibling trace to the shared user message', () => {
		expect(
			visibleTraceMessageId(trace({ responseMessageId: 'assistant-hidden' }), visibleIds)
		).toBeNull();
	});

	it('uses the anchor only for legacy traces without a response message', () => {
		expect(visibleTraceMessageId(trace({}), visibleIds)).toBe('user-1');
	});
});
