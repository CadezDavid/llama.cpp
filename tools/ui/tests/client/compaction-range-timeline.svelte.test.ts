import { describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';
import CompactionRangeTimeline from '$lib/components/app/dialogs/CompactionRangeTimeline.svelte';
import { MessageRole, MessageType } from '$lib/enums';

const messages: DatabaseMessage[] = [
	{
		id: 'assistant-1',
		convId: 'conversation-1',
		type: MessageType.TEXT,
		timestamp: 1,
		role: MessageRole.ASSISTANT,
		content: 'First answer with enough preview text for the boundary.',
		parent: null,
		children: []
	},
	{
		id: 'assistant-2',
		convId: 'conversation-1',
		type: MessageType.TEXT,
		timestamp: 2,
		role: MessageRole.ASSISTANT,
		content: 'Second answer.',
		parent: 'assistant-1',
		children: []
	}
];

const candidates = [
	{
		sourceMessageIds: ['user-1', 'assistant-1'],
		endMessageId: 'assistant-1',
		turnCount: 1,
		sourceTokenCount: 100
	},
	{
		sourceMessageIds: ['user-1', 'assistant-1', 'user-2', 'assistant-2'],
		endMessageId: 'assistant-2',
		turnCount: 2,
		sourceTokenCount: 400
	}
];

describe('CompactionRangeTimeline', () => {
	it('exposes token-weighted safe boundaries and keyboard selection', async () => {
		const onSelect = vi.fn();
		const screen = await render(CompactionRangeTimeline, {
			candidates,
			messages,
			selectedIndex: 0,
			totalTokenCount: 500,
			protectedTurns: 2,
			protectedTokens: 100,
			onSelect
		});
		const slider = screen.getByRole('slider', { name: 'Compact conversation through turn' });

		await expect.element(slider).toHaveAttribute('aria-valuenow', '1');
		await expect
			.element(screen.getByRole('button', { name: /Turn 1, assistant: First answer/ }))
			.toHaveAttribute('style', expect.stringContaining('20%'));
		(await slider.element()).dispatchEvent(
			new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })
		);
		expect(onSelect).toHaveBeenCalledWith(1);
	});
});
