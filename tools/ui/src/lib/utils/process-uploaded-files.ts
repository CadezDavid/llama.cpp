import { isSvgMimeType, svgBase64UrlToPngDataURL } from './svg-to-png';
import { isWebpMimeType, webpBase64UrlToPngDataURL } from './webp-to-png';
import { heicFileToJpegDataURL, isHeicMimeType } from './heic-to-jpeg';
import { FileTypeCategory } from '$lib/enums';
import { SETTINGS_KEYS } from '$lib/constants';
import { modelsStore } from '$lib/stores/models.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { toast } from 'svelte-sonner';
import { getFileTypeCategory } from '$lib/utils';
import { extractPDFDocument } from './pdf-processing';
import { AttachmentProcessingError, AttachmentService } from '$lib/services/attachment.service';
import type { ExtractedAttachment } from '$lib/types';

/**
 * Read a file as a data URL (base64 encoded)
 * @param file - The file to read
 * @returns Promise resolving to the data URL string
 */
function readFileAsDataURL(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
}

/**
 * Read a file as UTF-8 text
 * @param file - The file to read
 * @returns Promise resolving to the text content
 */
function readFileAsUTF8(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsText(file);
	});
}

/**
 * Process uploaded files into ChatUploadedFile format with previews and content
 *
 * This function processes various file types and generates appropriate previews:
 * - Images: Base64 data URLs with format normalization (SVG/WebP → PNG)
 * - Text files: UTF-8 content extraction
 * - PDFs: Metadata only (processed later in conversion pipeline)
 * - Audio: Base64 data URLs for preview
 *
 * @param files - Array of File objects to process
 * @returns Promise resolving to array of ChatUploadedFile objects
 */
export async function processFilesToChatUploaded(
	files: File[],
	activeModelId?: string,
	onUpdate?: (file: ChatUploadedFile) => void
): Promise<ChatUploadedFile[]> {
	const results: ChatUploadedFile[] = [];

	for (const file of files) {
		const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
		const diagnostics = AttachmentService.createDiagnostics();
		const base: ChatUploadedFile = {
			id,
			name: file.name,
			size: file.size,
			type: file.type,
			file,
			isLoading: true,
			attachmentProcessing: { stage: 'extracting', diagnostics }
		};
		onUpdate?.(base);

		try {
			if (getFileTypeCategory(file.type) === FileTypeCategory.IMAGE) {
				let preview = await readFileAsDataURL(file);

				// Normalize SVG and WebP to PNG, and HEIC to compressed JPEG, in previews
				if (isSvgMimeType(file.type)) {
					try {
						preview = await svgBase64UrlToPngDataURL(preview);
					} catch (err) {
						console.error('Failed to convert SVG to PNG:', err);
					}
				} else if (isWebpMimeType(file.type)) {
					try {
						preview = await webpBase64UrlToPngDataURL(preview);
					} catch (err) {
						console.error('Failed to convert WebP to PNG:', err);
					}
				} else if (isHeicMimeType(file.type)) {
					try {
						preview = await heicFileToJpegDataURL(file);
					} catch (err) {
						throw new Error(
							`Failed to convert HEIC image: ${err instanceof Error ? err.message : String(err)}`
						);
					}
				}

				const ready = { ...base, preview, isLoading: false, attachmentProcessing: undefined };
				results.push(ready);
				onUpdate?.(ready);
			} else if (getFileTypeCategory(file.type) === FileTypeCategory.PDF) {
				// Show suggestion toast if vision model is available but PDF as image is disabled
				const hasVisionSupport = activeModelId
					? modelsStore.modelSupportsVision(activeModelId)
					: false;
				const currentConfig = settingsStore.config;
				if (hasVisionSupport && !currentConfig.pdfAsImage) {
					toast.info(`You can enable parsing PDF as images with vision models.`, {
						duration: 8000,
						action: {
							label: 'Enable PDF as Images',
							onClick: () => {
								settingsStore.updateConfig(SETTINGS_KEYS.PDF_AS_IMAGE, true);
								toast.success('PDF parsing as images enabled!', {
									duration: 3000
								});
							}
						}
					});
				}
				if (hasVisionSupport && currentConfig.pdfAsImage) {
					const ready = {
						...base,
						isLoading: false,
						attachmentProcessing: undefined
					};
					results.push(ready);
					onUpdate?.(ready);
					continue;
				}
				const extracted = await extractPDFDocument(file);
				if (!activeModelId) {
					const waitingDiagnostics = AttachmentService.setDiagnosticStage(
						diagnostics,
						'waiting-model'
					);
					const pending = {
						...base,
						textContent: extracted.text,
						attachmentProcessing: {
							stage: 'waiting-model' as const,
							extracted,
							diagnostics: waitingDiagnostics
						},
						isLoading: false
					};
					results.push(pending);
					onUpdate?.(pending);
					continue;
				}
				const processing = await AttachmentService.processExtracted(
					base,
					extracted,
					activeModelId,
					undefined,
					(stage, stageDiagnostics) =>
						onUpdate?.({
							...base,
							textContent: extracted.text,
							attachmentProcessing: {
								stage,
								extracted,
								diagnostics: stageDiagnostics
							},
							isLoading: true
						}),
					diagnostics
				);
				const ready = {
					...base,
					textContent: extracted.text,
					attachmentProcessing: processing,
					isLoading: false
				};
				results.push(ready);
				onUpdate?.(ready);
			} else if (getFileTypeCategory(file.type) === FileTypeCategory.AUDIO) {
				// Generate preview URL for audio files
				const preview = await readFileAsDataURL(file);
				const ready = { ...base, preview, isLoading: false, attachmentProcessing: undefined };
				results.push(ready);
				onUpdate?.(ready);
			} else if (getFileTypeCategory(file.type) === FileTypeCategory.VIDEO) {
				// Generate preview URL for video files
				const preview = await readFileAsDataURL(file);
				const ready = { ...base, preview, isLoading: false, attachmentProcessing: undefined };
				results.push(ready);
				onUpdate?.(ready);
			} else {
				// Fallback: treat unknown files as text
				const textContent = await readFileAsUTF8(file);
				if (!textContent.trim()) throw new Error('Attachment is empty');
				const extracted: ExtractedAttachment = {
					text: textContent,
					extractor: 'text',
					segments: [{ text: textContent }]
				};
				if (!activeModelId) {
					const waitingDiagnostics = AttachmentService.setDiagnosticStage(
						diagnostics,
						'waiting-model'
					);
					const pending = {
						...base,
						textContent,
						attachmentProcessing: {
							stage: 'waiting-model' as const,
							extracted,
							diagnostics: waitingDiagnostics
						},
						isLoading: false
					};
					results.push(pending);
					onUpdate?.(pending);
					continue;
				}
				const processing = await AttachmentService.processExtracted(
					base,
					extracted,
					activeModelId,
					undefined,
					(stage, stageDiagnostics) =>
						onUpdate?.({
							...base,
							textContent,
							attachmentProcessing: {
								stage,
								extracted,
								diagnostics: stageDiagnostics
							},
							isLoading: true
						}),
					diagnostics
				);
				const ready = {
					...base,
					textContent,
					attachmentProcessing: processing,
					isLoading: false
				};
				results.push(ready);
				onUpdate?.(ready);
			}
		} catch (error) {
			console.error('Error processing file', file.name, error);
			const message = error instanceof Error ? error.message : String(error);
			const failed = {
				...base,
				isLoading: false,
				loadError: message,
				attachmentProcessing: {
					stage: 'failed' as const,
					error: message,
					diagnostics:
						error instanceof AttachmentProcessingError
							? error.diagnostics
							: AttachmentService.failDiagnostics(diagnostics, message)
				}
			};
			results.push(failed);
			onUpdate?.(failed);
		}
	}

	return results;
}
