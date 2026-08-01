import { describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';
import ChatAttachmentsListItemThumbnailFile from '$lib/components/app/chat/ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItemThumbnailFile.svelte';
import type { AttachmentProcessingDiagnostics, ChatUploadedFile } from '$lib/types';

function diagnostics(): AttachmentProcessingDiagnostics {
	return {
		id: 'diagnostic-1',
		createdAt: 1_000,
		updatedAt: 2_000,
		stage: 'failed',
		stageStartedAt: 1_900,
		generationModel: 'Gemma 4 31B',
		embeddingModel: 'Jina Embeddings v5 Text Small Retrieval',
		generationContextSize: 98_304,
		embeddingContextSize: 2_048,
		sourceTokenCount: 20_000,
		summaryPromptTokenCount: 20_100,
		chunkCount: 12,
		stageDurationsMs: { extracting: 30, measuring: 40, summarizing: 50, indexing: 60 },
		batches: [
			{
				ordinal: 0,
				inputCount: 1,
				tokenCount: 180,
				durationMs: 20,
				status: 'failed',
				httpStatus: 400,
				error: 'exact server error'
			}
		],
		failure: {
			stage: 'indexing',
			message: 'exact server error',
			httpStatus: 400
		}
	};
}

describe('attachment failure diagnostics', () => {
	it('shows the exact error and copies only the safe trace fields', async () => {
		const writeText = vi.spyOn(navigator.clipboard, 'writeText').mockResolvedValue(undefined);
		const uploadedFile: ChatUploadedFile = {
			id: 'file-1',
			name: 'private.txt',
			size: 100,
			type: 'text/plain',
			file: new File(['secret document text'], 'private.txt', { type: 'text/plain' }),
			textContent: 'secret document text',
			loadError: 'exact server error',
			attachmentProcessing: {
				stage: 'failed',
				error: 'exact server error',
				diagnostics: diagnostics()
			}
		};
		const screen = await render(ChatAttachmentsListItemThumbnailFile, {
			id: uploadedFile.id,
			name: uploadedFile.name,
			size: uploadedFile.size,
			textContent: uploadedFile.textContent,
			uploadedFile
		});

		await expect.element(screen.getByText('Failed: exact server error')).toBeVisible();
		await screen.getByRole('button', { name: 'Copy safe diagnostics', exact: true }).click();

		expect(writeText).toHaveBeenCalledOnce();
		const copied = writeText.mock.calls[0][0];
		expect(copied).toContain('"httpStatus": 400');
		expect(copied).toContain('"message": "exact server error"');
		expect(copied).not.toContain('secret document text');
		expect(copied).not.toContain('private.txt');
	});
});
