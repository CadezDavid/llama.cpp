export interface MemoryProviderCapabilities {
	api_version: string;
	retrieval: string[];
	filters: string[];
	memory_mutations: string[];
	automatic_writes: boolean;
}

export interface MemoryRecord {
	id: string;
	text: string;
	source: string;
	project?: string | null;
	tier: 'archive' | 'core';
	created_at: string;
}

export interface MemoryRetrievalHit extends MemoryRecord {
	score: number;
	semantic_score?: number | null;
	keyword_score?: number | null;
	lexical_coverage?: number;
	conversation_id?: string | null;
	memory_ids?: string[];
}

export interface MemoryRetrievalResponse {
	query: string;
	semantic_available: boolean;
	degraded_reason?: string | null;
	results: MemoryRetrievalHit[];
}

export interface SpominClientOptions {
	baseUrl: string;
	apiToken?: string;
	timeoutMs?: number;
}
