import type { ApiChatMessageData } from './api';

export type CompactionStatus = 'pending' | 'ready';
export type CompactionProjectionAction = 'apply' | 'restore';

export interface DatabaseCompaction {
	id: string;
	conversationId: string;
	status: CompactionStatus;
	sourceMessageIds: string[];
	deltaSourceMessageIds: string[];
	sourceFingerprint: string;
	summary: string;
	summarySchemaVersion: number;
	promptVersion: number;
	modelId?: string;
	createdAt: number;
	sourceTokenCount: number;
	beforeTokenCount: number;
	projectedTokenCount: number;
	previousCompactionId?: string;
	generation: number;
}

export interface DatabaseCompactionProjectionEvent {
	id: string;
	conversationId: string;
	anchorMessageId: string;
	action: CompactionProjectionAction;
	compactionId?: string;
	createdAt: number;
}

export interface CompactionTurnUnit {
	messageIds: string[];
	startIndex: number;
	endIndex: number;
	complete: boolean;
}

export interface CompactionRangeCandidate {
	sourceMessageIds: string[];
	endMessageId: string;
	turnCount: number;
	sourceTokenCount: number;
}

export interface ResolvedCompaction {
	record: DatabaseCompaction;
	event: DatabaseCompactionProjectionEvent;
	replacementMessages: ApiChatMessageData[];
}
