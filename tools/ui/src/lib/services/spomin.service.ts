import type {
	MemoryProviderCapabilities,
	MemoryRecord,
	MemoryRetrievalResponse,
	SpominClientOptions
} from '$lib/types';

export class SpominService {
	private readonly baseUrl: string;
	private readonly apiToken?: string;
	private readonly timeoutMs: number;

	constructor(options: SpominClientOptions) {
		this.baseUrl = options.baseUrl.replace(/\/+$/, '');
		this.apiToken = options.apiToken?.trim() || undefined;
		this.timeoutMs = Math.max(100, options.timeoutMs ?? 750);
	}

	private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
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
			if (!response.ok) {
				throw new Error(`Spomin request failed (${response.status})`);
			}
			if (response.status === 204) return undefined as T;
			return (await response.json()) as T;
		} finally {
			clearTimeout(timeout);
		}
	}

	async health(): Promise<boolean> {
		try {
			const result = await this.request<{ status: string }>('/healthz');
			return result.status === 'ok';
		} catch {
			return false;
		}
	}

	async capabilities(): Promise<MemoryProviderCapabilities> {
		return await this.request('/v1/memory/capabilities');
	}

	async retrieve(input: {
		query: string;
		limit?: number;
		project?: string;
		excludeConversationId?: string;
	}): Promise<MemoryRetrievalResponse> {
		return await this.request('/v1/memory/retrieve', {
			method: 'POST',
			body: JSON.stringify({
				query: input.query,
				limit: input.limit,
				project: input.project || undefined,
				exclude_conversation_id: input.excludeConversationId
			})
		});
	}

	async list(input: { limit?: number; project?: string } = {}): Promise<MemoryRecord[]> {
		const params = new URLSearchParams();
		if (input.limit) params.set('limit', String(input.limit));
		if (input.project) params.set('project', input.project);
		const suffix = params.size ? `?${params}` : '';
		const response = await this.request<{ results: MemoryRecord[] }>(`/v1/memories${suffix}`);
		return response.results;
	}

	async create(input: {
		text: string;
		project?: string;
		tier?: 'archive' | 'core';
	}): Promise<MemoryRecord> {
		return await this.request('/v1/memories', {
			method: 'POST',
			body: JSON.stringify({ ...input, source: 'llama.cpp-webui' })
		});
	}

	async update(
		id: string,
		changes: { text?: string; project?: string | null; tier?: 'archive' | 'core' }
	): Promise<MemoryRecord> {
		return await this.request(`/v1/memories/${encodeURIComponent(id)}`, {
			method: 'PATCH',
			body: JSON.stringify(changes)
		});
	}

	async delete(id: string): Promise<void> {
		await this.request(`/v1/memories/${encodeURIComponent(id)}`, { method: 'DELETE' });
	}
}
