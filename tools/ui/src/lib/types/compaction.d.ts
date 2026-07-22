import type { ApiChatMessageData } from './api';

export type CompactionStatus = 'pending' | 'ready';
export type CompactionProjectionAction = 'apply' | 'restore';
export type CompactionMode = 'off' | 'ask' | 'automatic';
export type CompactionActivationMode = 'manual' | 'confirmed' | 'automatic';

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
	activationMode?: CompactionActivationMode;
	contextSize?: number;
	usableInputTokenCount?: number;
	triggerPercent?: number;
	targetPercent?: number;
	attemptCount?: number;
	durationMs?: number;
	strictRetryUsed?: boolean;
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

export interface CompactionPolicy {
	mode: CompactionMode;
	contextSize: number;
	outputReserveTokens: number;
	retrievalReserveTokens: number;
	safetyMarginTokens: number;
	usableInputTokens: number;
	triggerPercent: number;
	targetPercent: number;
	triggerTokens: number;
	targetTokens: number;
	protectedTurns: number;
	protectedTailTokens: number;
}

export interface CompactionPreflightMeasurement {
	promptTokens: number;
	utilizationPercent: number;
	triggered: boolean;
	hardLimitExceeded: boolean;
}

export interface ResolvedCompaction {
	record: DatabaseCompaction;
	event: DatabaseCompactionProjectionEvent;
	replacementMessages: ApiChatMessageData[];
}
