import type { ErrorDialogType } from '$lib/enums';
import type { ApiChatCompletionToolCall, ApiChatMessageData } from './api';
import type {
	AttachmentProcessingDiagnostics,
	DatabaseMessage,
	DatabaseMessageExtra
} from './database';

export interface ChatUploadedFile {
	id: string;
	name: string;
	size: number;
	type: string;
	file: File;
	preview?: string;
	textContent?: string;
	mcpPrompt?: {
		serverName: string;
		promptName: string;
		arguments?: Record<string, string>;
	};
	isLoading?: boolean;
	loadError?: string;
	attachmentProcessing?: AttachmentProcessingResult;
}

export type AttachmentProcessingStage =
	| 'extracting'
	| 'waiting-model'
	| 'measuring'
	| 'summarizing'
	| 'indexing'
	| 'ready'
	| 'failed';

export interface ExtractedAttachmentSegment {
	text: string;
	page?: number;
	section?: string;
}

export interface ExtractedAttachment {
	text: string;
	segments: ExtractedAttachmentSegment[];
	extractor: 'text' | 'pdfjs';
}

export interface AttachmentProcessingResult {
	stage: AttachmentProcessingStage;
	mode?: 'inline' | 'indexed';
	attachmentId?: string;
	extracted?: ExtractedAttachment;
	sourceTokenCount?: number;
	summary?: string;
	error?: string;
	diagnostics?: AttachmentProcessingDiagnostics;
}

export interface ChatAttachmentDisplayItem {
	id: string;
	name: string;
	size?: number;
	preview?: string;
	isImage: boolean;
	isLoading?: boolean;
	loadError?: string;
	uploadedFile?: ChatUploadedFile;
	attachment?: DatabaseMessageExtra;
	attachmentIndex?: number;
	textContent?: string;
}

export interface ChatMessageSiblingInfo {
	message: DatabaseMessage;
	siblingIds: string[];
	currentIndex: number;
	totalSiblings: number;
}

export interface ChatMessagePromptProgress {
	cache: number;
	processed: number;
	time_ms: number;
	total: number;
}

export interface ChatMessageTimings {
	cache_n?: number;
	predicted_ms?: number;
	predicted_n?: number;
	prompt_ms?: number;
	prompt_n?: number;
	agentic?: ChatMessageAgenticTimings;
}

export interface ChatMessageAgenticTimings {
	turns: number;
	toolCallsCount: number;
	toolsMs: number;
	toolCalls?: ChatMessageToolCallTiming[];
	perTurn?: ChatMessageAgenticTurnStats[];
	llm: {
		predicted_n: number;
		predicted_ms: number;
		prompt_n: number;
		prompt_ms: number;
	};
}

export interface ChatMessageAgenticTurnStats {
	turn: number;
	llm: {
		predicted_n: number;
		predicted_ms: number;
		prompt_n: number;
		prompt_ms: number;
	};
	toolCalls: ChatMessageToolCallTiming[];
	toolsMs: number;
}

export interface ChatMessageToolCallTiming {
	name: string;
	duration_ms: number;
	success: boolean;
}

/**
 * Callbacks for streaming chat responses (used by both agentic and non-agentic paths)
 */
export interface ChatStreamCallbacks {
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallsStreaming?: (toolCalls: ApiChatCompletionToolCall[]) => void;
	onAttachments?: (messageId: string, extras: DatabaseMessageExtra[]) => void;
	onModel?: (model: string) => void;
	onCompletionId?: (id: string) => void;
	onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
	onAssistantTurnComplete?: (
		content: string,
		reasoningContent: string | undefined,
		timings: ChatMessageTimings | undefined,
		toolCalls: ApiChatCompletionToolCall[] | undefined
	) => Promise<void>;
	createToolResultMessage?: (
		toolCallId: string,
		content: string,
		extras?: DatabaseMessageExtra[]
	) => Promise<DatabaseMessage>;
	updateToolResultMessage?: (
		messageId: string,
		content: string,
		extras?: DatabaseMessageExtra[]
	) => Promise<void>;
	createAssistantMessage?: () => Promise<DatabaseMessage>;
	prepareMessages?: (turn: number) => Promise<ApiChatMessageData[] | null>;
	onFlowComplete?: (timings?: ChatMessageTimings) => void;
	onError?: (error: Error) => void;
	onTurnComplete?: (intermediateTimings: ChatMessageTimings) => void;
}

/**
 * Error dialog state for displaying server/timeout errors
 */
export interface ErrorDialogState {
	type: ErrorDialogType;
	message: string;
	contextInfo?: { n_prompt_tokens: number; n_ctx: number };
}

/**
 * Live processing stats during prompt evaluation
 */
export interface LiveProcessingStats {
	tokensProcessed: number;
	totalTokens: number;
	timeMs: number;
	tokensPerSecond: number;
	etaSecs?: number;
}

/**
 * Live generation stats during token generation
 */
export interface LiveGenerationStats {
	tokensGenerated: number;
	timeMs: number;
	tokensPerSecond: number;
}

/**
 * Options for getting attachment display items
 */
export interface AttachmentDisplayItemsOptions {
	uploadedFiles?: ChatUploadedFile[];
	attachments?: DatabaseMessageExtra[];
}

/**
 * Result of file processing operation
 */
export interface FileProcessingResult {
	extras: DatabaseMessageExtra[];
	emptyFiles: string[];
}
