/**
 * File upload lifecycle for the ChatScreen form.
 *
 * Owns the queue of processed `ChatUploadedFile`, the rejection-by-capability
 * dialog state, and the dual-layer validation pipeline (general format +
 * model modality). The caller provides the active model's capabilities and ID
 * as reactive getters so validation tracks the model in real time.
 */

import { processFilesToChatUploaded } from '$lib/utils/browser-only';
import { isFileTypeSupported, filterFilesByModalities } from '$lib/utils';
import { AttachmentProcessingError, AttachmentService } from '$lib/services/attachment.service';
import { modelsStore } from '$lib/stores/models.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';

interface UseChatScreenFileUploadOptions {
	capabilities: () => { hasVision: boolean; hasAudio: boolean; hasVideo: boolean };
	activeModelId: () => string | null | undefined;
}

export interface FileErrorData {
	generallyUnsupported: File[];
	modalityUnsupported: File[];
	modalityReasons: Record<string, string>;
	supportedTypes: string[];
}

export function useChatScreenFileUpload(options: UseChatScreenFileUploadOptions) {
	let uploadedFiles = $state<ChatUploadedFile[]>([]);
	let showFileErrorDialog = $state(false);
	let fileErrorData = $state<FileErrorData>({
		generallyUnsupported: [],
		modalityUnsupported: [],
		modalityReasons: {},
		supportedTypes: []
	});
	let classifiedModelId = $state<string | null | undefined>(options.activeModelId());

	$effect(() => {
		const modelId = options.activeModelId();
		if (!modelId) return;
		if (isRouterMode() && !modelsStore.isModelLoaded(modelId)) return;
		const modelChanged = modelId !== classifiedModelId;
		classifiedModelId = modelId;
		for (const file of uploadedFiles) {
			const processing = file.attachmentProcessing;
			const shouldProcess =
				processing?.stage === 'waiting-model' ||
				(modelChanged && processing?.stage === 'ready' && processing.mode === 'inline');
			if (!shouldProcess || !processing.extracted) {
				continue;
			}
			uploadedFiles = uploadedFiles.map((candidate) =>
				candidate.id === file.id
					? {
							...candidate,
							isLoading: true,
							loadError: undefined,
							attachmentProcessing: {
								...processing,
								stage: 'measuring'
							}
						}
					: candidate
			);
			AttachmentService.processExtracted(
				file,
				processing.extracted,
				modelId,
				undefined,
				(stage, diagnostics) => {
					uploadedFiles = uploadedFiles.map((candidate) =>
						candidate.id === file.id
							? {
									...candidate,
									attachmentProcessing: {
										...processing,
										stage,
										diagnostics
									}
								}
							: candidate
					);
				},
				processing.diagnostics
			)
				.then((updated) => {
					if (!uploadedFiles.some((candidate) => candidate.id === file.id)) {
						AttachmentService.discardPending(updated?.attachmentId);
						return;
					}
					uploadedFiles = uploadedFiles.map((candidate) =>
						candidate.id === file.id
							? { ...candidate, isLoading: false, attachmentProcessing: updated }
							: candidate
					);
				})
				.catch((error: unknown) => {
					const message = error instanceof Error ? error.message : String(error);
					uploadedFiles = uploadedFiles.map((candidate) =>
						candidate.id === file.id
							? {
									...candidate,
									isLoading: false,
									loadError: message,
									attachmentProcessing: {
										stage: 'failed',
										error: message,
										diagnostics:
											error instanceof AttachmentProcessingError
												? error.diagnostics
												: processing.diagnostics
													? AttachmentService.failDiagnostics(processing.diagnostics, message)
													: undefined
									}
								}
							: candidate
					);
				});
		}
	});

	async function processFiles(files: File[]) {
		const generallySupported: File[] = [];
		const generallyUnsupported: File[] = [];

		for (const file of files) {
			if (isFileTypeSupported(file.name, file.type)) {
				generallySupported.push(file);
			} else {
				generallyUnsupported.push(file);
			}
		}

		const { supportedFiles, unsupportedFiles, modalityReasons } = filterFilesByModalities(
			generallySupported,
			options.capabilities()
		);

		const allUnsupportedFiles = [...generallyUnsupported, ...unsupportedFiles];

		if (allUnsupportedFiles.length > 0) {
			const supportedTypes: string[] = ['text files', 'PDFs'];
			const caps = options.capabilities();
			if (caps.hasVision) supportedTypes.push('images');
			if (caps.hasAudio) supportedTypes.push('audio files');
			if (caps.hasVideo) supportedTypes.push('video files');

			fileErrorData = {
				generallyUnsupported,
				modalityUnsupported: unsupportedFiles,
				modalityReasons,
				supportedTypes
			};
			showFileErrorDialog = true;
		}

		if (supportedFiles.length > 0) {
			const modelId = options.activeModelId();
			const readyModelId =
				modelId && (!isRouterMode() || modelsStore.isModelLoaded(modelId)) ? modelId : undefined;
			await processFilesToChatUploaded(supportedFiles, readyModelId, (updated) => {
				const index = uploadedFiles.findIndex((candidate) => candidate.id === updated.id);
				uploadedFiles =
					index === -1
						? [...uploadedFiles, updated]
						: uploadedFiles.map((candidate, candidateIndex) =>
								candidateIndex === index ? updated : candidate
							);
			});
		}
	}

	function handleFileUpload(files: File[]) {
		return processFiles(files);
	}

	function handleFileRemove(fileId: string) {
		const file = uploadedFiles.find((candidate) => candidate.id === fileId);
		if (file) {
			AttachmentService.discardPending(file.attachmentProcessing?.attachmentId);
		}
		uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
	}

	return {
		get uploadedFiles() {
			return uploadedFiles;
		},
		set uploadedFiles(value) {
			uploadedFiles = value;
		},
		get showFileErrorDialog() {
			return showFileErrorDialog;
		},
		set showFileErrorDialog(value) {
			showFileErrorDialog = value;
		},
		get fileErrorData() {
			return fileErrorData;
		},
		handleFileUpload,
		handleFileRemove
	};
}
