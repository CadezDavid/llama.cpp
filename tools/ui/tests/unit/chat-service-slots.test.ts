import { afterEach, describe, expect, it, vi } from 'vitest';

const auth = vi.hoisted(() => ({ headers: {} as Record<string, string> }));

vi.mock('$lib/utils/api-headers', () => ({
	getAuthHeaders: () => auth.headers,
	getJsonHeaders: () => ({ 'content-type': 'application/json', ...auth.headers })
}));

import { ChatService } from '$lib/services/chat.service';

describe('ChatService slot status', () => {
	afterEach(() => {
		auth.headers = {};
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('authenticates the slot request and preserves the router model', async () => {
		auth.headers = { Authorization: 'Bearer secret' };
		const fetchMock = vi.fn().mockResolvedValue(
			new Response(JSON.stringify([{ is_processing: false }]), {
				status: 200,
				headers: { 'content-type': 'application/json' }
			})
		);
		vi.stubGlobal('fetch', fetchMock);

		await expect(ChatService.areAllSlotsIdle('model/name')).resolves.toBe(true);

		expect(fetchMock).toHaveBeenCalledWith('./slots?model=model%2Fname', {
			headers: { Authorization: 'Bearer secret' },
			signal: undefined
		});
	});

	it('omits authorization when no API key is configured', async () => {
		const fetchMock = vi
			.fn()
			.mockResolvedValue(new Response(JSON.stringify([{ is_processing: false }])));
		vi.stubGlobal('fetch', fetchMock);

		await ChatService.areAllSlotsIdle();

		expect(fetchMock).toHaveBeenCalledWith('./slots', {
			headers: {},
			signal: undefined
		});
	});

	it('reports a busy slot', async () => {
		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockResolvedValue(
					new Response(JSON.stringify([{ is_processing: false }, { is_processing: true }]))
				)
		);

		await expect(ChatService.areAllSlotsIdle()).resolves.toBe(false);
	});

	it.each([
		['an unavailable endpoint', () => Promise.resolve(new Response('', { status: 404 }))],
		['an unauthorized endpoint', () => Promise.resolve(new Response('', { status: 401 }))],
		['a malformed response', () => Promise.resolve(new Response('not-json'))],
		['a network failure', () => Promise.reject(new TypeError('offline'))]
	])('falls back to idle for %s', async (_label, response) => {
		vi.stubGlobal('fetch', vi.fn().mockImplementation(response));

		await expect(ChatService.areAllSlotsIdle()).resolves.toBe(true);
	});
});
