import { beforeEach, describe, expect, it, vi } from 'vitest';

const settings = vi.hoisted(() => ({ enabled: false }));

vi.mock('$lib/stores/settings.svelte', () => ({
	config: () => ({ memoryDebugLogging: settings.enabled })
}));

import { memoryDebug, sanitizeMemoryDebugDetails } from '$lib/utils/memory-debug';

describe('memory diagnostics', () => {
	beforeEach(() => {
		settings.enabled = false;
		vi.restoreAllMocks();
	});

	it('does not write to the console unless enabled', () => {
		const debug = vi.spyOn(console, 'debug').mockImplementation(() => {});

		memoryDebug('retrieval.test', { candidateCount: 2 });

		expect(debug).not.toHaveBeenCalled();
	});

	it('redacts sensitive values when enabled', () => {
		settings.enabled = true;
		const debug = vi.spyOn(console, 'debug').mockImplementation(() => {});

		memoryDebug('retrieval.test', {
			conversationId: 'conversation-1',
			query: 'private question',
			nested: {
				apiToken: 'secret',
				promptText: 'private prompt',
				status: 200
			}
		});

		expect(debug).toHaveBeenCalledWith('[Memory] retrieval.test', {
			conversationId: 'conversation-1',
			query: '[redacted]',
			nested: {
				apiToken: '[redacted]',
				promptText: '[redacted]',
				status: 200
			}
		});
	});

	it('summarizes errors without retaining their stack', () => {
		expect(sanitizeMemoryDebugDetails({ error: new TypeError('offline') })).toEqual({
			error: { name: 'TypeError', message: 'offline' }
		});
	});
});
