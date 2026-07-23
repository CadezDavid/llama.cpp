import { describe, expect, it } from 'vitest';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import type { ExportedConversation } from '$lib/types';

describe('retrieval trace JSONL export', () => {
	it('round-trips request-specific memory context', () => {
		const data: ExportedConversation = {
			conv: {
				id: 'conversation-1',
				name: 'Trace export',
				lastModified: 1,
				currNode: 'assistant-1'
			},
			messages: [],
			retrievalTraces: [
				{
					id: 'trace-1',
					conversationId: 'conversation-1',
					anchorMessageId: 'user-1',
					responseMessageId: 'assistant-1',
					createdAt: 1,
					query: 'request',
					queryFingerprint: 'fingerprint',
					queryTerms: ['request'],
					compactionGeneration: 0,
					providers: { spomin: { status: 'ok' } },
					hits: [
						{
							id: 'spomin:memory-1',
							source: 'long-term-memory',
							score: 0.9,
							selected: true,
							contentSnapshot: 'Historical memory snapshot'
						}
					],
					injectedHitIds: ['spomin:memory-1'],
					injectedTokenCount: 8
				}
			]
		};

		const text = conversationsStore.serializeSessionToJsonl(data);
		const parsed = conversationsStore.parseSessionsJsonl(text);

		expect(text).toContain('"type":"retrieval_trace"');
		expect(parsed[0].retrievalTraces).toEqual(data.retrievalTraces);
	});
});
