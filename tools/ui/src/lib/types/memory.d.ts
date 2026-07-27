export interface MemoryProviderCapabilities {
	api_version: string;
	query_channels: string[];
	filters: string[];
	memory_mutations: string[];
	embedding_profiles: boolean;
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
	semantic_score?: number | null;
	conversation_id?: string | null;
	memory_ids?: string[];
}

export interface MemoryEmbeddingProfile {
	id: string;
	model_id: string;
	configured_model: string;
	revision: string;
	dimensions: number;
	metadata?: Record<string, unknown>;
}

export interface MemoryQueryChannel {
	status: 'complete' | 'unavailable' | 'not_requested';
	error?: string;
	results: MemoryRetrievalHit[];
}

export interface MemoryRetrievalResponse {
	query: string;
	profile?: MemoryEmbeddingProfile | null;
	semantic: MemoryQueryChannel;
}

export interface MemoryEmbeddingStatus {
	status: 'ready' | 'reindexing' | 'unavailable';
	error?: string;
	profile?: MemoryEmbeddingProfile | null;
	total: number;
	ready: number;
	pending: number;
	processing: number;
	failed: number;
	stale: number;
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
	/** Legacy field retained while old IndexedDB records remain readable. */
	terms?: string[];
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
	/** Legacy diagnostic field. New retrievals are semantic-only. */
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
	/** Original trace whose context selection was reused for regeneration. */
	reusedFromTraceId?: string;
	createdAt: number;
	query: string;
	queryFingerprint: string;
	/** Legacy diagnostic field. New retrievals do not write query terms. */
	queryTerms?: string[];
	compactionGeneration: number;
	providers: Record<
		string,
		{ status: 'ok' | 'timeout' | 'error' | 'disabled' | 'not-applicable'; detail?: string }
	>;
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
	/** Legacy cooldown field. New retrievals use queryFingerprint. */
	queryTerms?: string[];
	score: number;
	compactionGeneration: number;
	updatedAt: number;
}
