import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import ChatMessageMemoryContext from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageUser/ChatMessageMemoryContext.svelte';
import type { DatabaseRetrievalTrace } from '$lib/types';

function trace(overrides: Partial<DatabaseRetrievalTrace> = {}): DatabaseRetrievalTrace {
	return {
		id: 'trace-1',
		conversationId: 'conversation-1',
		anchorMessageId: 'user-1',
		responseMessageId: 'assistant-1',
		createdAt: Date.now(),
		query: 'current request',
		queryFingerprint: 'fingerprint',
		queryTerms: ['current', 'request'],
		compactionGeneration: 1,
		providers: {
			local: { status: 'ok', detail: 'lexical+semantic' },
			spomin: { status: 'ok', detail: 'hybrid' }
		},
		hits: [],
		injectedHitIds: [],
		injectedTokenCount: 0,
		...overrides
	};
}

describe('ChatMessageMemoryContext', () => {
	it('shows exact injected snapshots separately from skipped candidates', async () => {
		const screen = await render(ChatMessageMemoryContext, {
			traces: [
				trace({
					hits: [
						{
							id: 'spomin:memory-1',
							source: 'long-term-memory',
							score: 0.91,
							selected: true,
							tokenCount: 17,
							contentSnapshot: 'The durable project preference.',
							provenance: { memoryIds: ['memory-1'], project: 'llama' }
						},
						{
							id: 'local:chunk-2',
							source: 'conversation-recall',
							score: 0.2,
							selected: false,
							reason: 'below-threshold'
						}
					],
					injectedHitIds: ['spomin:memory-1'],
					injectedTokenCount: 17
				})
			]
		});

		await expect.element(screen.getByText('Memory context: 1 item, 17 tokens')).toBeVisible();
		await screen.getByText('Memory context: 1 item, 17 tokens').click();
		await expect.element(screen.getByText('The durable project preference.')).toBeVisible();
		await expect.element(screen.getByText('Spomin memory')).toBeVisible();
		await screen.getByText('Skipped candidates (1)').click();
		await expect.element(screen.getByText(/below-threshold/)).toBeVisible();
	});

	it('makes a semantic recall failure visible without opening diagnostics', async () => {
		const screen = await render(ChatMessageMemoryContext, {
			traces: [
				trace({
					providers: {
						local: { status: 'error', detail: 'embedding unavailable' },
						spomin: { status: 'timeout' }
					}
				})
			]
		});

		await expect
			.element(screen.getByText('Memory context: unavailable - Conversation recall unavailable'))
			.toBeVisible();
	});

	it('explains when conversation recall does not apply', async () => {
		const screen = await render(ChatMessageMemoryContext, {
			traces: [
				trace({
					providers: {
						local: {
							status: 'not-applicable',
							detail: 'conversation is not compacted'
						},
						spomin: { status: 'ok', detail: 'no-results' }
					}
				})
			]
		});

		await screen.getByText('Memory context: none - no relevant matches').click();
		await expect
			.element(
				screen.getByText('Conversation recall: not applicable (conversation is not compacted)')
			)
			.toBeVisible();
	});

	it('handles legacy injected traces without a text snapshot', async () => {
		const screen = await render(ChatMessageMemoryContext, {
			traces: [
				trace({
					hits: [
						{
							id: 'local:legacy',
							source: 'conversation-recall',
							score: 0.8,
							selected: true,
							tokenCount: 10
						}
					],
					injectedHitIds: ['local:legacy'],
					injectedTokenCount: 10
				})
			]
		});

		await screen.getByText('Memory context: 1 item, 10 tokens').click();
		await expect
			.element(screen.getByText('Text snapshot is unavailable for this legacy request.'))
			.toBeVisible();
	});

	it('labels a regenerated answer as reusing the original selection', async () => {
		const screen = await render(ChatMessageMemoryContext, {
			traces: [
				trace({
					reusedFromTraceId: 'trace-original'
				})
			]
		});

		await screen.getByText('Memory context: none - no relevant matches').click();
		await expect
			.element(
				screen.getByText('Reused the original memory selection; providers were not queried again.')
			)
			.toBeVisible();
	});
});
