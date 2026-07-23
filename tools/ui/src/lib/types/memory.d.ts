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

export type ArchiveEmbeddingStatus = 'pending' | 'ready' | 'failed';

export interface DatabaseArchiveChunk {
	id: string;
	conversationId: string;
	compactionId: string;
	generation: number;
	sourceMessageIds: string[];
	text: string;
	terms: string[];
	createdAt: number;
	embedding?: number[];
	embeddingModel?: string;
	embeddingStatus: ArchiveEmbeddingStatus;
	embeddingError?: string;
}

export interface DatabaseArchiveTerm {
	id: string;
	conversationId: string;
	chunkId: string;
	term: string;
}

export interface RetrievalTraceHit {
	id: string;
	source: 'conversation-recall' | 'long-term-memory';
	score: number;
	lexicalScore?: number;
	semanticScore?: number;
	providerScore?: number;
	selected: boolean;
	reason?: string;
	tokenCount?: number;
	/** Immutable copy of text that was actually sent to the model. */
	contentSnapshot?: string;
	/** Non-secret source identifiers captured when retrieval ran. */
	provenance?: Record<string, unknown>;
}

export interface DatabaseRetrievalTrace {
	id: string;
	conversationId: string;
	anchorMessageId: string;
	/** Assistant message prepared by this request. Missing on legacy traces. */
	responseMessageId?: string;
	createdAt: number;
	query: string;
	queryFingerprint: string;
	queryTerms: string[];
	compactionGeneration: number;
	providers: Record<string, { status: 'ok' | 'timeout' | 'error' | 'disabled'; detail?: string }>;
	hits: RetrievalTraceHit[];
	injectedHitIds: string[];
	injectedTokenCount: number;
	finalPromptTokenCount?: number;
}

export interface DatabaseRetrievalHitUsage {
	id: string;
	conversationId: string;
	hitId: string;
	source: 'conversation-recall' | 'long-term-memory';
	lastInjectedUserTurn: number;
	queryFingerprint: string;
	queryTerms: string[];
	score: number;
	compactionGeneration: number;
	updatedAt: number;
}
