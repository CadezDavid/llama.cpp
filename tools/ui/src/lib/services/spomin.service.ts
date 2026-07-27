import type {
	MemoryProviderCapabilities,
	MemoryEmbeddingStatus,
	MemoryRecord,
	MemoryRetrievalResponse,
	SpominClientOptions
} from '$lib/types';
import { memoryDebug } from '$lib/utils/memory-debug';

export class SpominService {
	private readonly baseUrl: string;
	private readonly apiToken?: string;
	private readonly timeoutMs: number;

	constructor(options: SpominClientOptions) {
		this.baseUrl = options.baseUrl.replace(/\/+$/, '');
		this.apiToken = options.apiToken?.trim() || undefined;
		this.timeoutMs = Math.max(100, options.timeoutMs ?? 2000);
	}

	private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
		const startedAt = performance.now();
		const method = init.method ?? 'GET';
		memoryDebug('spomin.request.start', {
			method,
			path,
			timeoutMs: this.timeoutMs,
			authenticationConfigured: Boolean(this.apiToken),
			hasBody: Boolean(init.body)
		});
		try {
			const response = await fetch(`${this.baseUrl}${path}`, {
				...init,
				signal: controller.signal,
				headers: {
					accept: 'application/json',
					...(init.body ? { 'content-type': 'application/json' } : {}),
					...(this.apiToken ? { authorization: `Bearer ${this.apiToken}` } : {}),
					...init.headers
				}
			});
			memoryDebug('spomin.request.response', {
				method,
				path,
				status: response.status,
				ok: response.ok,
				durationMs: Math.round(performance.now() - startedAt)
			});
			if (!response.ok) {
				throw new Error(`Spomin request failed (${response.status})`);
			}
			if (response.status === 204) return undefined as T;
			return (await response.json()) as T;
		} catch (error) {
			memoryDebug('spomin.request.error', {
				method,
				path,
				durationMs: Math.round(performance.now() - startedAt),
				aborted: controller.signal.aborted,
				error
			});
			throw error;
		} finally {
			clearTimeout(timeout);
		}
	}

	async health(): Promise<boolean> {
		try {
			const result = await this.request<{ status: string }>('/healthz');
			const reachable = result.status === 'ok';
			memoryDebug('spomin.health.result', { reachable, status: result.status });
			return reachable;
		} catch (error) {
			memoryDebug('spomin.health.unavailable', { error });
			return false;
		}
	}

	async capabilities(): Promise<MemoryProviderCapabilities> {
		return await this.request('/v2/capabilities');
	}

	async retrieve(input: {
		query: string;
		candidateLimit?: number;
		project?: string;
		excludeConversationId?: string;
	}): Promise<MemoryRetrievalResponse> {
		const response = await this.request<MemoryRetrievalResponse>('/v2/memories/query', {
			method: 'POST',
			body: JSON.stringify({
				query: input.query,
				channels: ['semantic'],
				semantic_limit: input.candidateLimit,
				project: input.project || undefined,
				exclude_conversation_id: input.excludeConversationId
			})
		});
		if (response.semantic.status !== 'complete') {
			throw new Error(
				`Spomin semantic retrieval failed: ${response.semantic.error ?? response.semantic.status}`
			);
		}
		for (const result of response.semantic.results) {
			if (typeof result.semantic_score !== 'number' || !Number.isFinite(result.semantic_score)) {
				throw new Error(`Spomin semantic result ${result.id} has no valid semantic score`);
			}
		}
		memoryDebug('spomin.retrieve.complete', {
			semanticResultCount: response.semantic.results.length,
			semanticStatus: response.semantic.status,
			profileId: response.profile?.id
		});
		return response;
	}

	async recordAccess(input: {
		eventId: string;
		chunkIds: string[];
		contextId?: string;
	}): Promise<boolean> {
		if (!input.chunkIds.length) return false;
		const response = await this.request<{ recorded: boolean }>('/v2/memories/access', {
			method: 'POST',
			body: JSON.stringify({
				event_id: input.eventId,
				client: 'llama.cpp-webui',
				event_type: 'injected',
				context_id: input.contextId,
				chunk_ids: input.chunkIds
			})
		});
		return response.recorded;
	}

	async embeddingStatus(): Promise<MemoryEmbeddingStatus> {
		return await this.request('/v2/embeddings/status');
	}

	async reindex(force = false): Promise<MemoryEmbeddingStatus> {
		return await this.request('/v2/embeddings/reindex', {
			method: 'POST',
			body: JSON.stringify({ force })
		});
	}

	async list(input: { limit?: number; project?: string } = {}): Promise<MemoryRecord[]> {
		const params = new URLSearchParams();
		if (input.limit) params.set('limit', String(input.limit));
		if (input.project) params.set('project', input.project);
		const suffix = params.size ? `?${params}` : '';
		const response = await this.request<{ results: MemoryRecord[] }>(`/v2/memories${suffix}`);
		return response.results;
	}

	async create(input: {
		text: string;
		project?: string;
		tier?: 'archive' | 'core';
	}): Promise<MemoryRecord> {
		return await this.request('/v2/memories', {
			method: 'POST',
			body: JSON.stringify({ ...input, source: 'llama.cpp-webui' })
		});
	}

	async update(
		id: string,
		changes: { text?: string; project?: string | null; tier?: 'archive' | 'core' }
	): Promise<MemoryRecord> {
		return await this.request(`/v2/memories/${encodeURIComponent(id)}`, {
			method: 'PATCH',
			body: JSON.stringify(changes)
		});
	}

	async delete(id: string): Promise<void> {
		await this.request(`/v2/memories/${encodeURIComponent(id)}`, { method: 'DELETE' });
	}
}
