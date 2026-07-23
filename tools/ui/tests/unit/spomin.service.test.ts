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
					profile: null,
					semantic: { status: 'complete', results: [] },
					keyword: { status: 'complete', results: [] }
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
			candidateLimit: 20,
			project: 'alpha',
			excludeConversationId: 'chat-1'
		});

		expect(fetchMock).toHaveBeenCalledOnce();
		expect(fetchMock.mock.calls[0][0]).toBe('http://127.0.0.1:8084/v2/memories/query');
		expect(fetchMock.mock.calls[0][1].headers.authorization).toBe('Bearer secret');
		expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
			query: 'database choice',
			channels: ['semantic', 'keyword'],
			semantic_limit: 20,
			keyword_limit: 20,
			project: 'alpha',
			exclude_conversation_id: 'chat-1'
		});
	});

	it('treats health failures as unavailable', async () => {
		vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')));
		await expect(new SpominService({ baseUrl: 'http://127.0.0.1:8084' }).health()).resolves.toBe(
			false
		);
	});

	it('reports only explicitly injected chunk ids', async () => {
		const fetchMock = vi
			.fn()
			.mockResolvedValue(new Response(JSON.stringify({ recorded: true }), { status: 200 }));
		vi.stubGlobal('fetch', fetchMock);
		const service = new SpominService({ baseUrl: 'http://127.0.0.1:8084' });

		await expect(
			service.recordAccess({
				eventId: 'trace-1:injected',
				chunkIds: ['chunk-1', 'chunk-2'],
				contextId: 'chat-1'
			})
		).resolves.toBe(true);

		expect(fetchMock.mock.calls[0][0]).toBe('http://127.0.0.1:8084/v2/memories/access');
		expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
			event_id: 'trace-1:injected',
			client: 'llama.cpp-webui',
			event_type: 'injected',
			context_id: 'chat-1',
			chunk_ids: ['chunk-1', 'chunk-2']
		});
	});
});
