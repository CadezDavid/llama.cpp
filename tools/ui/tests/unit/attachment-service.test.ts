import { afterEach, describe, expect, it, vi } from 'vitest';
import { AttachmentProcessingError, AttachmentService } from '$lib/services/attachment.service';
import { ChatService } from '$lib/services/chat.service';
import { CompactionService } from '$lib/services/compaction.service';
import { DatabaseService } from '$lib/services/database.service';
import { AttachmentType } from '$lib/enums';

const modelMocks = vi.hoisted(() => ({
	ensureModelLoaded: vi.fn(async () => undefined),
	getModelProps: vi.fn(() => ({})),
	fetchModelProps: vi.fn(async () => undefined),
	getModelContextSize: vi.fn((model: string): number =>
		model === 'Jina Embeddings v5 Text Small Retrieval' ? 2_048 : 98_304
	)
}));

const serverMocks = vi.hoisted(() => ({
	isRouterMode: vi.fn(() => true)
}));

vi.mock('$lib/stores/models.svelte', () => ({
	modelsStore: modelMocks
}));

vi.mock('$lib/stores/server.svelte', () => ({
	isRouterMode: serverMocks.isRouterMode
}));

vi.mock('$lib/stores/settings.svelte', () => ({
	config: () => ({
		max_tokens: 2_000,
		totalRecallTokenBudget: 2_500,
		embeddingBaseUrl: 'http://127.0.0.1:8080/v1',
		embeddingModel: 'Jina Embeddings v5 Text Small Retrieval',
		embeddingTimeoutMs: 60_000,
		semanticRecallThreshold: 0.58
	})
}));

describe('AttachmentService main-model summarization', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
		modelMocks.getModelContextSize.mockImplementation((model: string) =>
			model === 'Jina Embeddings v5 Text Small Retrieval' ? 2_048 : 98_304
		);
		modelMocks.ensureModelLoaded.mockClear();
		serverMocks.isRouterMode.mockReturnValue(true);
	});

	it('uses the active model for a transient summary and records it', async () => {
		vi.spyOn(CompactionService, 'createPolicy').mockReturnValue({
			usableInputTokens: 90_000
		} as ReturnType<typeof CompactionService.createPolicy>);
		const tokenize = vi
			.spyOn(ChatService, 'tokenizePrompt')
			.mockResolvedValueOnce(10_000)
			.mockResolvedValue(10);
		const measure = vi.spyOn(ChatService, 'measurePrompt').mockResolvedValue({
			tokenCount: 10_100,
			hasNonTextContent: false,
			exactForTextOnly: true
		});
		const send = vi.spyOn(ChatService, 'sendMessage').mockResolvedValue('Main-model synopsis');
		const addAttachment = vi.spyOn(DatabaseService, 'addAttachment').mockResolvedValue(undefined);
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(
				new Response(
					JSON.stringify({
						data: [{ index: 0, embedding: [1, 0] }]
					}),
					{ status: 200 }
				)
			)
		);

		const result = await AttachmentService.processExtracted(
			{ id: 'file-1', name: 'large.txt', size: 100_000, type: 'text/plain' },
			{ text: 'Attachment text', extractor: 'text', segments: [{ text: 'Attachment text' }] },
			'Gemma 4 31B'
		);

		expect(measure).toHaveBeenCalledWith(
			expect.any(Array),
			expect.objectContaining({
				model: 'Gemma 4 31B',
				enableThinking: true,
				custom: expect.objectContaining({ cache_ram_store: false })
			}),
			undefined
		);
		expect(send).toHaveBeenCalledWith(
			expect.any(Array),
			expect.objectContaining({
				model: 'Gemma 4 31B',
				temperature: 0,
				max_tokens: 2_512,
				presence_penalty: 0,
				repeat_penalty: 1,
				custom: {
					thinking_budget_tokens: 512,
					cache_ram_store: false
				}
			}),
			undefined,
			undefined
		);
		expect(fetch).toHaveBeenCalledWith(
			'/v1/embeddings',
			expect.objectContaining({
				method: 'POST',
				body: JSON.stringify({
					model: 'Jina Embeddings v5 Text Small Retrieval',
					input: 'Document: Attachment text'
				})
			})
		);
		expect(modelMocks.ensureModelLoaded).toHaveBeenCalledWith(
			'Jina Embeddings v5 Text Small Retrieval'
		);
		expect(tokenize).toHaveBeenNthCalledWith(
			2,
			'Document: Attachment text',
			'Jina Embeddings v5 Text Small Retrieval',
			undefined,
			true
		);

		await AttachmentService.persistPendingForMessage(
			[
				{
					type: AttachmentType.TEXT,
					name: 'large.txt',
					content: '',
					processingMode: 'indexed',
					attachmentId: result?.attachmentId
				}
			],
			'conversation-1',
			'message-1'
		);
		expect(addAttachment.mock.calls[0][0]).toMatchObject({
			summarizerModel: 'Gemma 4 31B',
			summary: 'Main-model synopsis',
			diagnostics: expect.objectContaining({
				stage: 'ready',
				generationModel: 'Gemma 4 31B',
				embeddingModel: 'Jina Embeddings v5 Text Small Retrieval',
				chunkCount: 1,
				batches: [
					expect.objectContaining({
						ordinal: 0,
						inputCount: 1,
						tokenCount: 10,
						status: 'ready'
					})
				]
			})
		});
	});

	it('fails before generation when the templated summary request exceeds model context', async () => {
		modelMocks.getModelContextSize.mockReturnValue(12_000);
		vi.spyOn(CompactionService, 'createPolicy').mockReturnValue({
			usableInputTokens: 10_000
		} as ReturnType<typeof CompactionService.createPolicy>);
		vi.spyOn(ChatService, 'tokenizePrompt').mockResolvedValue(1_000);
		vi.spyOn(ChatService, 'measurePrompt').mockResolvedValue({
			tokenCount: 10_000,
			hasNonTextContent: false,
			exactForTextOnly: true
		});
		const send = vi.spyOn(ChatService, 'sendMessage');

		await expect(
			AttachmentService.processExtracted(
				{ id: 'file-2', name: 'too-large.txt', size: 100_000, type: 'text/plain' },
				{ text: 'Attachment text', extractor: 'text', segments: [{ text: 'Attachment text' }] },
				'Gemma 4 31B'
			)
		).rejects.toThrow(
			'Attachment requires 12512 summary tokens, exceeding the 12000 token limit for model "Gemma 4 31B"'
		);
		expect(send).not.toHaveBeenCalled();
	});

	it('preserves the server error and records safe failure diagnostics', async () => {
		vi.spyOn(CompactionService, 'createPolicy').mockReturnValue({
			usableInputTokens: 90_000
		} as ReturnType<typeof CompactionService.createPolicy>);
		vi.spyOn(ChatService, 'tokenizePrompt').mockResolvedValueOnce(10_000).mockResolvedValue(10);
		vi.spyOn(ChatService, 'measurePrompt').mockResolvedValue({
			tokenCount: 10_100,
			hasNonTextContent: false,
			exactForTextOnly: true
		});
		vi.spyOn(ChatService, 'sendMessage').mockResolvedValue('Main-model synopsis');
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(
				new Response(
					JSON.stringify({
						error: {
							message: 'input is too large for the embedding context'
						}
					}),
					{ status: 400 }
				)
			)
		);

		let failure: unknown;
		try {
			await AttachmentService.processExtracted(
				{ id: 'file-3', name: 'private.txt', size: 100_000, type: 'text/plain' },
				{
					text: 'sensitive attachment contents',
					extractor: 'text',
					segments: [{ text: 'sensitive attachment contents' }]
				},
				'Gemma 4 31B'
			);
		} catch (error) {
			failure = error;
		}

		expect(failure).toBeInstanceOf(AttachmentProcessingError);
		expect(failure).toMatchObject({
			message: 'input is too large for the embedding context',
			diagnostics: {
				stage: 'failed',
				failure: {
					stage: 'indexing',
					message: 'input is too large for the embedding context',
					httpStatus: 400
				},
				batches: [
					expect.objectContaining({
						inputCount: 1,
						status: 'failed',
						httpStatus: 400,
						error: 'input is too large for the embedding context'
					})
				]
			}
		});
		expect(JSON.stringify((failure as AttachmentProcessingError).diagnostics)).not.toContain(
			'sensitive attachment contents'
		);
	});

	it('fails before embedding when an exact chunk token count exceeds the embedding context', async () => {
		vi.spyOn(CompactionService, 'createPolicy').mockReturnValue({
			usableInputTokens: 90_000
		} as ReturnType<typeof CompactionService.createPolicy>);
		vi.spyOn(ChatService, 'tokenizePrompt')
			.mockResolvedValueOnce(10_000)
			.mockResolvedValueOnce(2_049);
		vi.spyOn(ChatService, 'measurePrompt').mockResolvedValue({
			tokenCount: 10_100,
			hasNonTextContent: false,
			exactForTextOnly: true
		});
		vi.spyOn(ChatService, 'sendMessage').mockResolvedValue('Main-model synopsis');
		const fetch = vi.fn();
		vi.stubGlobal('fetch', fetch);

		await expect(
			AttachmentService.processExtracted(
				{ id: 'file-4', name: 'large.txt', size: 100_000, type: 'text/plain' },
				{ text: 'Attachment text', extractor: 'text', segments: [{ text: 'Attachment text' }] },
				'Gemma 4 31B'
			)
		).rejects.toThrow(
			'Attachment chunk 1 requires 2049 embedding tokens, exceeding the 2048 token limit for model "Jina Embeddings v5 Text Small Retrieval"'
		);
		expect(fetch).not.toHaveBeenCalled();
	});

	it('sends one bounded chunk per embedding request', async () => {
		vi.spyOn(CompactionService, 'createPolicy').mockReturnValue({
			usableInputTokens: 90_000
		} as ReturnType<typeof CompactionService.createPolicy>);
		vi.spyOn(ChatService, 'tokenizePrompt').mockResolvedValueOnce(10_000).mockResolvedValue(200);
		vi.spyOn(ChatService, 'measurePrompt').mockResolvedValue({
			tokenCount: 10_100,
			hasNonTextContent: false,
			exactForTextOnly: true
		});
		vi.spyOn(ChatService, 'sendMessage').mockResolvedValue('Main-model synopsis');
		const fetch = vi.fn().mockImplementation(async () => {
			return new Response(
				JSON.stringify({
					data: [{ index: 0, embedding: [1, 0] }]
				}),
				{ status: 200 }
			);
		});
		vi.stubGlobal('fetch', fetch);
		const text = Array.from({ length: 601 }, (_, index) => `word${index}`).join(' ');

		await AttachmentService.processExtracted(
			{ id: 'file-5', name: 'large.txt', size: 100_000, type: 'text/plain' },
			{ text, extractor: 'text', segments: [{ text }] },
			'Gemma 4 31B'
		);

		expect(fetch).toHaveBeenCalledTimes(3);
		for (const [, options] of fetch.mock.calls) {
			const request = JSON.parse(String((options as RequestInit).body)) as {
				input: string;
			};
			expect(typeof request.input).toBe('string');
			expect(request.input).toMatch(/^Document: /);
			expect(request.input.replace(/^Document: /, '').split(/\s+/).length).toBeLessThanOrEqual(300);
		}
	});

	it('reindexes old chunks with document prefixes and prefixes the search query', async () => {
		vi.spyOn(DatabaseService, 'getAttachment').mockResolvedValue({
			id: 'attachment-1',
			conversationId: 'conversation-1',
			messageId: 'message-1',
			name: 'paper.pdf',
			mimeType: 'application/pdf',
			size: 1_000,
			extractor: 'pdfjs',
			extractedText: 'Entropy encoding codebook pointers',
			summary: 'Paper synopsis',
			sourceTokenCount: 100,
			tokenizerModel: 'Gemma 4 31B',
			summarizerModel: 'Gemma 4 31B',
			embeddingModel: 'Jina Embeddings v5 Text Small Retrieval',
			embeddingDimensions: 2,
			chunkingVersion: 2,
			status: 'ready',
			createdAt: 1
		});
		const oldChunk = {
			id: 'chunk-1',
			attachmentId: 'attachment-1',
			conversationId: 'conversation-1',
			ordinal: 0,
			text: 'Entropy encoding codebook pointers',
			embedding: [0, 1],
			embeddingModel: 'Jina Embeddings v5 Text Small Retrieval',
			embeddingDimensions: 2,
			embeddingStatus: 'ready' as const
		};
		vi.spyOn(DatabaseService, 'getAttachmentChunks')
			.mockResolvedValueOnce([oldChunk])
			.mockResolvedValueOnce([{ ...oldChunk, embedding: [1, 0] }]);
		const updateIndex = vi
			.spyOn(DatabaseService, 'updateAttachmentIndex')
			.mockResolvedValue(undefined);
		vi.spyOn(ChatService, 'tokenizePrompt').mockResolvedValue(10);
		const fetch = vi.fn().mockImplementation(
			async () =>
				new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 0] }] }), {
					status: 200
				})
		);
		vi.stubGlobal('fetch', fetch);

		const result = JSON.parse(
			await AttachmentService.executeTool('conversation-1', AttachmentService.searchToolName, {
				attachment_id: 'attachment-1',
				query: 'entropy encoding'
			})
		) as { matches: Array<{ chunk_id: string; score: number }> };

		expect(updateIndex).toHaveBeenCalledWith(
			'attachment-1',
			'Jina Embeddings v5 Text Small Retrieval',
			4,
			[
				{
					chunkId: 'chunk-1',
					embedding: [1, 0]
				}
			]
		);
		const inputs = fetch.mock.calls.map(
			([, options]) => JSON.parse(String((options as RequestInit).body)).input
		);
		expect(inputs).toEqual([
			'Document: Entropy encoding codebook pointers',
			'Query: entropy encoding'
		]);
		expect(result.matches).toEqual([expect.objectContaining({ chunk_id: 'chunk-1', score: 1 })]);
	});

	it('reports the best rejected score when no chunk meets the threshold', async () => {
		vi.spyOn(DatabaseService, 'getAttachment').mockResolvedValue({
			id: 'attachment-2',
			conversationId: 'conversation-1',
			messageId: 'message-1',
			name: 'paper.pdf',
			mimeType: 'application/pdf',
			size: 1_000,
			extractor: 'pdfjs',
			extractedText: 'Document text',
			summary: 'Paper synopsis',
			sourceTokenCount: 100,
			tokenizerModel: 'Gemma 4 31B',
			summarizerModel: 'Gemma 4 31B',
			embeddingModel: 'Jina Embeddings v5 Text Small Retrieval',
			embeddingDimensions: 2,
			chunkingVersion: 4,
			status: 'ready',
			createdAt: 1
		});
		vi.spyOn(DatabaseService, 'getAttachmentChunks').mockResolvedValue([
			{
				id: 'chunk-2',
				attachmentId: 'attachment-2',
				conversationId: 'conversation-1',
				ordinal: 0,
				text: 'Document text',
				embedding: [0.5, Math.sqrt(0.75)],
				embeddingModel: 'Jina Embeddings v5 Text Small Retrieval',
				embeddingDimensions: 2,
				embeddingStatus: 'ready'
			}
		]);
		vi.spyOn(ChatService, 'tokenizePrompt').mockResolvedValue(10);
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(
				new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 0] }] }), {
					status: 200
				})
			)
		);

		const result = JSON.parse(
			await AttachmentService.executeTool('conversation-1', AttachmentService.searchToolName, {
				attachment_id: 'attachment-2',
				query: 'unrelated query'
			})
		) as {
			matches: unknown[];
			diagnostics: {
				best_score: number;
				threshold: number;
				indexed_chunk_count: number;
				embedding_model: string;
			};
		};
		const bestScore = result.diagnostics.best_score;

		expect(result.matches).toEqual([]);
		expect(bestScore).toBeCloseTo(0.5);
		expect(result.diagnostics).toEqual({
			best_score: bestScore,
			threshold: 0.58,
			indexed_chunk_count: 1,
			embedding_model: 'Jina Embeddings v5 Text Small Retrieval'
		});
	});
});
