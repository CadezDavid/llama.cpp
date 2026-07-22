import type { ApiChatMessageData } from './api';
import type { DatabaseMessage } from './database';

export type ChatContextSource = 'conversation-recall' | 'long-term-memory';

export interface ChatPromptProjection {
	id: string;
	sourceMessageIds: string[];
	replacementMessages: ApiChatMessageData[];
	sourceFingerprint?: string;
	producer?: string;
	version?: number;
}

export interface ChatContextBlock {
	id: string;
	source: ChatContextSource;
	content: string;
	provenance?: Record<string, unknown>;
}

export interface PrepareChatContextInput {
	transcriptMessages: DatabaseMessage[];
	model?: string | null;
	excludeReasoning?: boolean;
	projections?: ChatPromptProjection[];
	contextBlocks?: ChatContextBlock[];
}

export interface PreparedChatContext {
	stableMessages: ApiChatMessageData[];
	requestMessages: ApiChatMessageData[];
	sourceMessageIds: string[];
	appliedProjectionIds: string[];
	contextBlockIds: string[];
	hasNonTextContent: boolean;
}
