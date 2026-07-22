import { afterEach, describe, expect, it, vi } from 'vitest';
import { SpominService } from '$lib/services/spomin.service';

describe('SpominService', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('sends retrieval scope, auth, and current-conversation exclusion', async () => {
		const fetchMock = vi.fn().mockResolvedValue(
			new Response(
				JSON.stringify({
					query: 'database choice',
					semantic_available: true,
					results: []
				}),
				{ status: 200 }
			)
		);
		vi.stubGlobal('fetch', fetchMock);
		const service = new SpominService({
			baseUrl: 'http://127.0.0.1:8084/',
			apiToken: 'secret'
		});

		await service.retrieve({
			query: 'database choice',
			limit: 3,
			project: 'alpha',
			excludeConversationId: 'chat-1'
		});

		expect(fetchMock).toHaveBeenCalledOnce();
		expect(fetchMock.mock.calls[0][0]).toBe('http://127.0.0.1:8084/v1/memory/retrieve');
		expect(fetchMock.mock.calls[0][1].headers.authorization).toBe('Bearer secret');
		expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
			query: 'database choice',
			limit: 3,
			project: 'alpha',
			exclude_conversation_id: 'chat-1'
		});
	});

	it('treats health failures as unavailable', async () => {
		vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')));
		await expect(
			new SpominService({ baseUrl: 'http://127.0.0.1:8084' }).health()
		).resolves.toBe(false);
	});
});
