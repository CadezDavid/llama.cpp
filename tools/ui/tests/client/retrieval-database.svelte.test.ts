import { afterEach, describe, expect, it } from 'vitest';
import { DatabaseService } from '$lib/services/database.service';
import type { DatabaseRetrievalTrace } from '$lib/types';

function trace(
	conversationId: string,
	id: string,
	responseMessageId: string,
	createdAt: number
): DatabaseRetrievalTrace {
	return {
		id,
		conversationId,
		anchorMessageId: 'user-1',
		responseMessageId,
		createdAt,
		query: '',
		queryFingerprint: '',
		queryTerms: [],
		compactionGeneration: 0,
		providers: {},
		hits: [],
		injectedHitIds: [],
		injectedTokenCount: 0
	};
}

describe('retrieval trace persistence', () => {
	let conversationId: string | null = null;

	afterEach(async () => {
		if (conversationId) {
			await DatabaseService.deleteConversation(conversationId, { deleteWithForks: true });
		}
		conversationId = null;
	});

	it('finds the newest trace belonging to an exact assistant response', async () => {
		const conversation = await DatabaseService.createConversation('Retrieval trace test');
		conversationId = conversation.id;
		await DatabaseService.addRetrievalTrace(trace(conversation.id, 'trace-old', 'assistant-1', 1));
		await DatabaseService.addRetrievalTrace(
			trace(conversation.id, 'trace-other', 'assistant-2', 3)
		);
		await DatabaseService.addRetrievalTrace(trace(conversation.id, 'trace-new', 'assistant-1', 2));

		const result = await DatabaseService.getRetrievalTraceForResponse(
			conversation.id,
			'assistant-1'
		);

		expect(result?.id).toBe('trace-new');
	});
});
