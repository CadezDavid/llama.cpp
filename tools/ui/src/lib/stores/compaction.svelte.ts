import { toast } from 'svelte-sonner';
import { isAbortError } from '$lib/utils/abort';
import { memoryDebug } from '$lib/utils/memory-debug';
import { isRouterMode, contextSize as serverContextSize } from '$lib/stores/server.svelte';
import {
	modelsStore,
	selectedModelContextSize,
	selectedModelName
} from '$lib/stores/models.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { config } from '$lib/stores/settings.svelte';
import { ChatContextService } from '$lib/services/chat-context.service';
import {
	CompactionService,
	COMPACTION_PROMPT_VERSION,
	COMPACTION_SCHEMA_VERSION
} from '$lib/services/compaction.service';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import type {
	CompactionBudget,
	CompactionGenerationResult
} from '$lib/services/compaction.service';
import type {
	CompactionRangeCandidate,
	CompactionMode,
	CompactionPolicy,
	CompactionPreflightMeasurement,
	DatabaseCompaction,
	DatabaseMessage,
	ResolvedCompaction,
	SettingsChatServiceOptions
} from '$lib/types';

export type CompactionDialogMode = 'create' | 'view' | 'ask';
export type CompactionDialogStep = 'measure' | 'generate' | 'review';

export interface CompactionPreflightInput {
	conversationId: string;
	anchorMessageId: string;
	messages: DatabaseMessage[];
	model?: string;
	contextSize: number;
	maxOutputTokens?: number;
	retrievalReserveTokens?: number;
	mode: CompactionMode;
	triggerPercent: number;
	targetPercent: number;
	protectedTurns: number;
	measurementOptions?: SettingsChatServiceOptions;
}

class CompactionStore {
	open = $state(false);
	mode = $state<CompactionDialogMode>('create');
	step = $state<CompactionDialogStep>('measure');
	loading = $state(false);
	applying = $state(false);
	error = $state<string | null>(null);
	validationErrors = $state<string[]>([]);
	activeCompaction = $state<ResolvedCompaction | null>(null);
	messages = $state<DatabaseMessage[]>([]);
	candidates = $state<CompactionRangeCandidate[]>([]);
	selectedCandidateIndex = $state(-1);
	model = $state<string | undefined>(undefined);
	contextSize = $state(0);
	beforeTokenCount = $state(0);
	sourceTokenCount = $state(0);
	projectedTokenCount = $state(0);
	restoredTokenCount = $state(0);
	summary = $state('');
	budget = $state<CompactionBudget | null>(null);
	policy = $state<CompactionPolicy | null>(null);
	preflightMeasurement = $state<CompactionPreflightMeasurement | null>(null);
	operationStatus = $state<string | null>(null);
	private abortController: AbortController | null = null;
	private pendingCompactionId: string | null = null;
	private preflightInput: CompactionPreflightInput | null = null;
	private askResolver: ((proceed: boolean) => void) | null = null;
	private snoozePercentByConversation = new Map<string, number>();
	private operations = new Map<string, Promise<boolean>>();
	private operationTail: Promise<void> = Promise.resolve();

	get selectedCandidate(): CompactionRangeCandidate | null {
		return this.candidates[this.selectedCandidateIndex] ?? null;
	}

	get sourceMessages(): DatabaseMessage[] {
		const ids = new Set(
			this.mode === 'view'
				? (this.activeCompaction?.record.sourceMessageIds ?? [])
				: (this.selectedCandidate?.sourceMessageIds ?? [])
		);
		return this.messages.filter((message) => ids.has(message.id));
	}

	get canApply(): boolean {
		return (
			this.step === 'review' &&
			this.validationErrors.length === 0 &&
			!!this.pendingCompactionId &&
			!this.applying
		);
	}

	get restoreExceedsContext(): boolean {
		return this.restoredTokenCount > this.contextSize;
	}

	get canSkipAsk(): boolean {
		return this.preflightMeasurement?.hardLimitExceeded === false;
	}

	async preflight(input: CompactionPreflightInput): Promise<boolean> {
		if (input.mode === 'off') {
			memoryDebug('compaction.preflight.skipped', {
				conversationId: input.conversationId,
				reason: 'disabled'
			});
			return true;
		}
		const existing = this.operations.get(input.conversationId);
		if (existing) {
			memoryDebug('compaction.preflight.join-existing', {
				conversationId: input.conversationId
			});
			return await existing;
		}
		const operation = this.operationTail
			.catch(() => undefined)
			.then(() => this.runPreflight(input))
			.finally(() => {
				this.operations.delete(input.conversationId);
			});
		this.operationTail = operation.then(
			() => undefined,
			() => undefined
		);
		this.operations.set(input.conversationId, operation);
		return await operation;
	}

	private async runPreflight(input: CompactionPreflightInput): Promise<boolean> {
		try {
			await this.preparePreflight(input);
			const measurement = this.preflightMeasurement;
			if (!measurement?.triggered) {
				memoryDebug('compaction.preflight.continue', {
					conversationId: input.conversationId,
					reason: 'below-trigger'
				});
				return true;
			}

			const snoozePercent = this.snoozePercentByConversation.get(input.conversationId);
			if (
				!measurement.hardLimitExceeded &&
				snoozePercent !== undefined &&
				measurement.utilizationPercent < snoozePercent
			) {
				memoryDebug('compaction.preflight.continue', {
					conversationId: input.conversationId,
					reason: 'snoozed',
					utilizationPercent: measurement.utilizationPercent,
					snoozePercent
				});
				return true;
			}

			if (this.policy?.mode === 'ask') {
				memoryDebug('compaction.preflight.await-confirmation', {
					conversationId: input.conversationId,
					hardLimitExceeded: measurement.hardLimitExceeded,
					utilizationPercent: measurement.utilizationPercent
				});
				this.mode = 'ask';
				this.step = 'measure';
				this.open = true;
				return await new Promise<boolean>((resolve) => {
					this.askResolver = resolve;
				});
			}

			return await this.runAutomaticCompaction();
		} catch (error) {
			memoryDebug('compaction.preflight.error', {
				conversationId: input.conversationId,
				hardLimitExceeded: this.preflightMeasurement?.hardLimitExceeded,
				error
			});
			this.error = error instanceof Error ? error.message : String(error);
			if (this.preflightMeasurement?.hardLimitExceeded) {
				this.mode = 'ask';
				this.open = true;
				return await new Promise<boolean>((resolve) => {
					this.askResolver = resolve;
				});
			}
			toast.warning('Compaction could not run; sending with the existing context');
			return true;
		}
	}

	private async preparePreflight(input: CompactionPreflightInput): Promise<void> {
		this.loading = true;
		this.error = null;
		this.validationErrors = [];
		this.summary = '';
		this.projectedTokenCount = 0;
		this.preflightInput = input;
		this.messages = [...input.messages];
		this.model = input.model;
		this.contextSize = input.contextSize;
		this.policy = CompactionService.createPolicy(input);
		this.operationStatus = 'Measuring context...';
		try {
			await DatabaseService.cleanupAbandonedCompactions();
			this.activeCompaction = await CompactionService.resolveActiveCompaction(
				input.conversationId,
				this.messages
			);
			const prepared = await ChatContextService.prepare({
				transcriptMessages: this.messages,
				model: this.model,
				excludeReasoning: !!input.measurementOptions?.excludeReasoningFromContext,
				projections: this.activeCompaction
					? [CompactionService.toProjection(this.activeCompaction)]
					: []
			});
			this.beforeTokenCount = (
				await ChatService.measurePrompt(prepared.stableMessages, {
					...input.measurementOptions,
					model: this.model
				})
			).tokenCount;
			this.preflightMeasurement = CompactionService.evaluatePreflight(
				this.beforeTokenCount,
				this.policy
			);
			memoryDebug('compaction.preflight.measured', {
				conversationId: input.conversationId,
				beforeTokenCount: this.beforeTokenCount,
				activeCompactionId: this.activeCompaction?.record.id,
				measurement: this.preflightMeasurement
			});
			if (!this.preflightMeasurement.triggered) return;

			const protectedTurns = await this.resolveProtectedTurnCount(this.policy);
			this.candidates = CompactionService.buildRangeCandidates(
				this.messages,
				this.activeCompaction?.record,
				protectedTurns
			);
			if (this.candidates.length === 0) {
				throw new Error('There are not enough complete old turns to compact safely');
			}
			for (let index = 0; index < this.candidates.length; index++) {
				await this.measureCandidate(index);
			}
			memoryDebug('compaction.candidates.measured', {
				conversationId: input.conversationId,
				protectedTurns,
				candidateCount: this.candidates.length,
				candidates: this.candidates.map((candidate) => ({
					endMessageId: candidate.endMessageId,
					turnCount: candidate.turnCount,
					sourceMessageCount: candidate.sourceMessageIds.length,
					sourceTokenCount: candidate.sourceTokenCount
				}))
			});
			const largest = this.candidates[this.candidates.length - 1];
			const allowance = CompactionService.getBudget(
				this.contextSize,
				largest.sourceTokenCount,
				this.beforeTokenCount
			).summaryOutputTokens;
			let selected = CompactionService.selectAutomaticCandidate(
				this.candidates,
				this.beforeTokenCount,
				this.policy.targetTokens,
				allowance
			);
			if (selected < 0) selected = this.candidates.length - 1;
			await this.selectCandidate(selected);
			if (!this.budget || this.sourceTokenCount < this.budget.minimumSourceTokens) {
				throw new Error('No compaction range is large enough to produce safe token savings');
			}
		} finally {
			this.loading = false;
			this.operationStatus = null;
		}
	}

	private async resolveProtectedTurnCount(policy: CompactionPolicy): Promise<number> {
		const complete = CompactionService.groupTurns(this.messages).filter((turn) => turn.complete);
		let protectedCount = 0;
		let protectedTokens = 0;
		for (let index = complete.length - 1; index >= 0; index--) {
			const turn = complete[index];
			const turnMessages = this.messages.slice(turn.startIndex, turn.endIndex + 1);
			protectedTokens += await ChatService.tokenizePrompt(
				CompactionService.serializeMessages(turnMessages),
				this.model
			);
			protectedCount++;
			if (
				protectedCount >= policy.protectedTurns &&
				protectedTokens >= policy.protectedTailTokens
			) {
				break;
			}
		}
		return protectedCount;
	}

	private async measureCandidate(index: number): Promise<void> {
		const candidate = this.candidates[index];
		if (!candidate || candidate.sourceTokenCount > 0) return;
		const source = this.messages.filter((message) =>
			candidate.sourceMessageIds.includes(message.id)
		);
		candidate.sourceTokenCount = await ChatService.tokenizePrompt(
			CompactionService.serializeMessages(source),
			this.model
		);
	}

	private async runAutomaticCompaction(): Promise<boolean> {
		const input = this.preflightInput;
		const candidate = this.selectedCandidate;
		const policy = this.policy;
		const budget = this.budget;
		if (!input || !candidate || !policy || !budget) return false;

		this.abortController?.abort();
		this.abortController = new AbortController();
		this.step = 'generate';
		this.operationStatus = 'Compacting conversation...';
		const startedAt = performance.now();
		const activationMode = policy.mode === 'ask' ? 'confirmed' : 'automatic';
		const statusToastId =
			policy.mode === 'automatic' ? toast.loading('Compacting conversation...') : undefined;
		const previous = this.activeCompaction?.record;
		let previousForPrompt = previous;
		let deltaSourceMessageIds = CompactionService.getDeltaSourceIds(
			candidate.sourceMessageIds,
			previous
		);
		if (deltaSourceMessageIds.length === 0) {
			deltaSourceMessageIds = [...candidate.sourceMessageIds];
			previousForPrompt = undefined;
		}
		const source = this.messages.filter((message) =>
			candidate.sourceMessageIds.includes(message.id)
		);
		memoryDebug('compaction.automatic.start', {
			conversationId: input.conversationId,
			activationMode,
			sourceMessageCount: candidate.sourceMessageIds.length,
			deltaMessageCount: deltaSourceMessageIds.length,
			beforeTokenCount: this.beforeTokenCount,
			sourceTokenCount: this.sourceTokenCount
		});

		try {
			const pending = await DatabaseService.createPendingCompaction({
				conversationId: input.conversationId,
				sourceMessageIds: candidate.sourceMessageIds,
				deltaSourceMessageIds,
				sourceFingerprint: await ChatContextService.fingerprintMessages(source),
				summary: '',
				summarySchemaVersion: COMPACTION_SCHEMA_VERSION,
				promptVersion: COMPACTION_PROMPT_VERSION,
				modelId: this.model,
				sourceTokenCount: this.sourceTokenCount,
				beforeTokenCount: this.beforeTokenCount,
				projectedTokenCount: 0,
				previousCompactionId: previous?.id,
				generation: (previous?.generation ?? 0) + 1,
				activationMode,
				contextSize: policy.contextSize,
				usableInputTokenCount: policy.usableInputTokens,
				triggerPercent: policy.triggerPercent,
				targetPercent: policy.targetPercent
			});
			this.pendingCompactionId = pending.id;

			let accepted = false;
			let attempts = 0;
			for (let attempt = 1; attempt <= 2; attempt++) {
				attempts = attempt;
				const strict = attempt === 2;
				this.operationStatus = strict
					? 'Compacting conversation (strict retry)...'
					: 'Compacting conversation...';
				const maxTokens = strict
					? Math.max(512, Math.floor(budget.summaryOutputTokens * 0.6))
					: budget.summaryOutputTokens;
				const result = await CompactionService.generateSummary({
					messages: this.messages,
					sourceMessageIds: candidate.sourceMessageIds,
					deltaSourceMessageIds,
					previousCompaction: previousForPrompt,
					model: this.model,
					maxTokens,
					strict,
					signal: this.abortController.signal
				});
				this.summary = result.summary;
				this.validationErrors = result.validationErrors;
				if (result.validationErrors.length > 0) continue;

				const previewRecord: DatabaseCompaction = { ...pending, summary: result.summary };
				const prepared = await ChatContextService.prepare({
					transcriptMessages: this.messages,
					model: this.model,
					excludeReasoning: !!input.measurementOptions?.excludeReasoningFromContext,
					projections: [
						{
							id: pending.id,
							sourceMessageIds: candidate.sourceMessageIds,
							replacementMessages: CompactionService.replacementMessages(previewRecord),
							sourceFingerprint: pending.sourceFingerprint
						}
					]
				});
				this.projectedTokenCount = (
					await ChatService.measurePrompt(prepared.stableMessages, {
						...input.measurementOptions,
						model: this.model
					})
				).tokenCount;
				accepted = CompactionService.acceptsAutomaticProjection(
					this.beforeTokenCount,
					this.projectedTokenCount,
					policy,
					budget.minimumSavingsTokens
				);
				memoryDebug('compaction.automatic.attempt', {
					conversationId: input.conversationId,
					attempt,
					strict,
					projectedTokenCount: this.projectedTokenCount,
					savedTokens: this.beforeTokenCount - this.projectedTokenCount,
					accepted,
					validationErrorCount: this.validationErrors.length
				});
				if (accepted) break;
				this.validationErrors = [
					`The projected prompt remains above ${policy.targetPercent + 5}% of usable input or saves too few tokens`
				];
			}

			if (!accepted) throw new Error(this.validationErrors[0] ?? 'Compaction was not effective');
			await DatabaseService.activateCompaction(
				pending.id,
				{
					summary: this.summary,
					projectedTokenCount: this.projectedTokenCount,
					activationMode,
					contextSize: policy.contextSize,
					usableInputTokenCount: policy.usableInputTokens,
					triggerPercent: policy.triggerPercent,
					targetPercent: policy.targetPercent,
					attemptCount: attempts,
					durationMs: Math.round(performance.now() - startedAt),
					strictRetryUsed: attempts > 1
				},
				input.anchorMessageId
			);
			this.pendingCompactionId = null;
			this.snoozePercentByConversation.delete(input.conversationId);
			this.activeCompaction = await CompactionService.resolveActiveCompaction(
				input.conversationId,
				this.messages
			);
			this.open = false;
			memoryDebug('compaction.automatic.activated', {
				conversationId: input.conversationId,
				compactionId: pending.id,
				attempts,
				durationMs: Math.round(performance.now() - startedAt),
				projectedTokenCount: this.projectedTokenCount,
				savedTokens: this.beforeTokenCount - this.projectedTokenCount
			});
			toast.success(
				policy.mode === 'automatic'
					? 'Conversation compacted automatically'
					: 'Conversation compacted'
			);
			return true;
		} catch (error) {
			memoryDebug('compaction.automatic.error', {
				conversationId: input.conversationId,
				pendingCompactionId: this.pendingCompactionId,
				aborted: isAbortError(error),
				error
			});
			if (this.pendingCompactionId) {
				await DatabaseService.deletePendingCompaction(this.pendingCompactionId);
				this.pendingCompactionId = null;
			}
			if (isAbortError(error)) return false;
			this.error = error instanceof Error ? error.message : String(error);
			if (this.preflightMeasurement?.hardLimitExceeded) {
				toast.error('Compaction failed and the request exceeds usable model input');
				return false;
			}
			toast.warning('Compaction failed; sending with the existing context');
			return true;
		} finally {
			if (statusToastId !== undefined) toast.dismiss(statusToastId);
			this.operationStatus = null;
			this.step = 'measure';
		}
	}

	async confirmAskCompaction(): Promise<void> {
		const resolver = this.askResolver;
		this.askResolver = null;
		const proceed = await this.runAutomaticCompaction();
		this.open = false;
		resolver?.(proceed);
	}

	skipAskCompaction(): void {
		if (!this.canSkipAsk || !this.preflightInput || !this.preflightMeasurement) return;
		this.snoozePercentByConversation.set(
			this.preflightInput.conversationId,
			Math.min(95, this.preflightMeasurement.utilizationPercent + 5)
		);
		const resolver = this.askResolver;
		this.askResolver = null;
		this.open = false;
		resolver?.(true);
	}

	cancelAskCompaction(): void {
		const resolver = this.askResolver;
		this.askResolver = null;
		this.open = false;
		resolver?.(false);
	}

	private resolveModel(messages: DatabaseMessage[]): string | undefined {
		if (!isRouterMode()) return undefined;
		const selected = selectedModelName();
		if (selected) return selected;
		return ChatService.findLatestAssistantModel(messages) ?? undefined;
	}

	private resolveContextSize(model?: string): number {
		if (isRouterMode()) {
			if (model) return modelsStore.getModelContextSize(model) ?? selectedModelContextSize() ?? 0;
			return selectedModelContextSize() ?? 0;
		}
		return serverContextSize() ?? 0;
	}

	async openCreate(): Promise<void> {
		this.preflightInput = null;
		this.policy = null;
		this.preflightMeasurement = null;
		this.mode = 'create';
		this.step = 'measure';
		this.open = true;
		await this.load();
	}

	async openView(): Promise<void> {
		this.mode = 'view';
		this.step = 'review';
		this.open = true;
		await this.load();
		if (this.activeCompaction) {
			this.summary = this.activeCompaction.record.summary;
			this.sourceTokenCount = this.activeCompaction.record.sourceTokenCount;
			this.beforeTokenCount = this.activeCompaction.record.beforeTokenCount;
			this.projectedTokenCount = this.activeCompaction.record.projectedTokenCount;
			const restored = await ChatContextService.prepare({
				transcriptMessages: this.messages,
				model: this.model
			});
			this.restoredTokenCount = (
				await ChatService.measurePrompt(restored.stableMessages, { model: this.model })
			).tokenCount;
		}
	}

	async refreshActive(): Promise<void> {
		const conversation = conversationsStore.activeConversation;
		if (!conversation) {
			this.activeCompaction = null;
			return;
		}
		this.activeCompaction = await CompactionService.resolveActiveCompaction(
			conversation.id,
			conversationsStore.activeMessages
		);
	}

	private async load(): Promise<void> {
		this.loading = true;
		this.error = null;
		this.validationErrors = [];
		this.summary = '';
		this.projectedTokenCount = 0;
		try {
			const conversation = conversationsStore.activeConversation;
			if (!conversation?.currNode) throw new Error('No active conversation');
			await DatabaseService.cleanupAbandonedCompactions();
			this.messages = [...conversationsStore.activeMessages];
			this.model = this.resolveModel(this.messages);
			if (isRouterMode() && !this.model) throw new Error('Select a model before compacting');
			this.contextSize = this.resolveContextSize(this.model);
			if (!this.contextSize) throw new Error('The current model context size is unavailable');
			const currentConfig = config();
			this.policy = CompactionService.createPolicy({
				mode: String(currentConfig.compactionMode ?? 'ask'),
				contextSize: this.contextSize,
				maxOutputTokens: Number(currentConfig.max_tokens) || undefined,
				retrievalReserveTokens: Number(currentConfig.totalRecallTokenBudget) || 2500,
				triggerPercent: Number(currentConfig.compactionTriggerPercent) || 78,
				targetPercent: Number(currentConfig.compactionTargetPercent) || 50,
				protectedTurns: Number(currentConfig.compactionProtectedTurns) || 8
			});
			this.activeCompaction = await CompactionService.resolveActiveCompaction(
				conversation.id,
				this.messages
			);
			const projection = this.activeCompaction
				? [CompactionService.toProjection(this.activeCompaction)]
				: [];
			const prepared = await ChatContextService.prepare({
				transcriptMessages: this.messages,
				model: this.model,
				projections: projection
			});
			this.beforeTokenCount = (
				await ChatService.measurePrompt(prepared.stableMessages, { model: this.model })
			).tokenCount;
			const protectedTurns = await this.resolveProtectedTurnCount(this.policy);
			this.candidates = CompactionService.buildRangeCandidates(
				this.messages,
				this.activeCompaction?.record,
				protectedTurns
			);
			if (this.mode === 'create') {
				if (this.candidates.length === 0) {
					throw new Error('There are not enough complete old turns to compact safely');
				}
				await this.selectBestCandidate();
				for (let index = 0; index < this.candidates.length; index++) {
					await this.measureCandidate(index);
				}
			}
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.loading = false;
		}
	}

	private async selectBestCandidate(): Promise<void> {
		for (let index = this.candidates.length - 1; index >= 0; index--) {
			await this.selectCandidate(index);
			if (!this.budget) continue;
			let previousForPrompt = this.activeCompaction?.record;
			let deltaSourceMessageIds = CompactionService.getDeltaSourceIds(
				this.selectedCandidate!.sourceMessageIds,
				previousForPrompt
			);
			if (deltaSourceMessageIds.length === 0) {
				deltaSourceMessageIds = [...this.selectedCandidate!.sourceMessageIds];
				previousForPrompt = undefined;
			}
			const input = CompactionService.buildSummaryMessages({
				messages: this.messages,
				sourceMessageIds: this.selectedCandidate!.sourceMessageIds,
				deltaSourceMessageIds,
				previousCompaction: previousForPrompt,
				model: this.model,
				maxTokens: this.budget.summaryOutputTokens
			});
			const inputTokens = (await ChatService.measurePrompt(input, { model: this.model }))
				.tokenCount;
			if (
				inputTokens + this.budget.summaryOutputTokens + this.budget.safetyMarginTokens <=
				this.contextSize
			) {
				return;
			}
		}
		throw new Error('No safe compaction range fits the current model context');
	}

	async selectCandidate(index: number): Promise<void> {
		const candidate = this.candidates[index];
		if (!candidate) return;
		this.selectedCandidateIndex = index;
		await this.measureCandidate(index);
		if (this.selectedCandidateIndex !== index) return;
		this.sourceTokenCount = candidate.sourceTokenCount;
		this.budget = CompactionService.getBudget(
			this.contextSize,
			this.sourceTokenCount,
			this.beforeTokenCount
		);
	}

	async generate(): Promise<void> {
		const conversation = conversationsStore.activeConversation;
		const candidate = this.selectedCandidate;
		if (!conversation?.currNode || !candidate || !this.budget) return;
		if (this.sourceTokenCount < this.budget.minimumSourceTokens) {
			this.error = `Select at least ${this.budget.minimumSourceTokens.toLocaleString()} source tokens`;
			return;
		}

		this.abortController?.abort();
		this.abortController = new AbortController();
		this.step = 'generate';
		this.error = null;
		this.validationErrors = [];
		const previous = this.activeCompaction?.record;
		let previousForPrompt = previous;
		let deltaSourceMessageIds = CompactionService.getDeltaSourceIds(
			candidate.sourceMessageIds,
			previous
		);
		if (deltaSourceMessageIds.length === 0) {
			deltaSourceMessageIds = [...candidate.sourceMessageIds];
			previousForPrompt = undefined;
		}
		const source = this.messages.filter((message) =>
			candidate.sourceMessageIds.includes(message.id)
		);
		try {
			const pending = await DatabaseService.createPendingCompaction({
				conversationId: conversation.id,
				sourceMessageIds: candidate.sourceMessageIds,
				deltaSourceMessageIds,
				sourceFingerprint: await ChatContextService.fingerprintMessages(source),
				summary: '',
				summarySchemaVersion: COMPACTION_SCHEMA_VERSION,
				promptVersion: COMPACTION_PROMPT_VERSION,
				modelId: this.model,
				sourceTokenCount: this.sourceTokenCount,
				beforeTokenCount: this.beforeTokenCount,
				projectedTokenCount: 0,
				previousCompactionId: previous?.id,
				generation: (previous?.generation ?? 0) + 1,
				activationMode: 'manual',
				contextSize: this.contextSize
			});
			this.pendingCompactionId = pending.id;
			const result: CompactionGenerationResult = await CompactionService.generateSummary({
				messages: this.messages,
				sourceMessageIds: candidate.sourceMessageIds,
				deltaSourceMessageIds,
				previousCompaction: previousForPrompt,
				model: this.model,
				maxTokens: this.budget.summaryOutputTokens,
				signal: this.abortController.signal
			});
			this.summary = result.summary;
			this.validationErrors = result.validationErrors;
			if (result.validationErrors.length > 0) {
				await DatabaseService.deletePendingCompaction(pending.id);
				this.pendingCompactionId = null;
				this.step = 'review';
				return;
			}

			const previewRecord: DatabaseCompaction = { ...pending, summary: result.summary };
			const prepared = await ChatContextService.prepare({
				transcriptMessages: this.messages,
				model: this.model,
				projections: [
					{
						id: pending.id,
						sourceMessageIds: candidate.sourceMessageIds,
						replacementMessages: CompactionService.replacementMessages(previewRecord),
						sourceFingerprint: pending.sourceFingerprint
					}
				]
			});
			this.projectedTokenCount = (
				await ChatService.measurePrompt(prepared.stableMessages, { model: this.model })
			).tokenCount;
			const savings = this.beforeTokenCount - this.projectedTokenCount;
			if (savings < this.budget.minimumSavingsTokens) {
				this.validationErrors = [
					`Compaction saves ${savings.toLocaleString()} tokens; at least ${this.budget.minimumSavingsTokens.toLocaleString()} are required`
				];
				await DatabaseService.deletePendingCompaction(pending.id);
				this.pendingCompactionId = null;
			}
			this.step = 'review';
		} catch (error) {
			if (this.pendingCompactionId) {
				await DatabaseService.deletePendingCompaction(this.pendingCompactionId);
				this.pendingCompactionId = null;
			}
			if (!isAbortError(error)) this.error = error instanceof Error ? error.message : String(error);
			this.step = 'measure';
		}
	}

	async apply(): Promise<void> {
		const conversation = conversationsStore.activeConversation;
		if (!conversation?.currNode || !this.pendingCompactionId || !this.canApply) return;
		this.applying = true;
		const compactionId = this.pendingCompactionId;
		memoryDebug('compaction.manual.apply.start', {
			conversationId: conversation.id,
			compactionId,
			beforeTokenCount: this.beforeTokenCount,
			projectedTokenCount: this.projectedTokenCount
		});
		try {
			await DatabaseService.activateCompaction(
				this.pendingCompactionId,
				{
					summary: this.summary,
					projectedTokenCount: this.projectedTokenCount,
					activationMode: 'manual',
					contextSize: this.contextSize,
					attemptCount: 1,
					strictRetryUsed: false
				},
				conversation.currNode
			);
			this.pendingCompactionId = null;
			await this.refreshActive();
			this.open = false;
			memoryDebug('compaction.manual.apply.complete', {
				conversationId: conversation.id,
				compactionId,
				savedTokens: this.beforeTokenCount - this.projectedTokenCount
			});
			toast.success('Conversation compacted');
		} catch (error) {
			memoryDebug('compaction.manual.apply.error', {
				conversationId: conversation.id,
				compactionId,
				error
			});
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.applying = false;
		}
	}

	async restore(): Promise<void> {
		const conversation = conversationsStore.activeConversation;
		if (!conversation?.currNode) return;
		memoryDebug('compaction.restore.start', {
			conversationId: conversation.id,
			anchorMessageId: conversation.currNode,
			compactionId: this.activeCompaction?.record.id
		});
		await DatabaseService.createCompactionRestoreEvent(conversation.id, conversation.currNode);
		await this.refreshActive();
		this.open = false;
		memoryDebug('compaction.restore.complete', { conversationId: conversation.id });
		toast.success('Original conversation context restored');
	}

	async close(): Promise<void> {
		if (this.mode === 'ask' && this.askResolver) {
			this.cancelAskCompaction();
			return;
		}
		this.abortController?.abort();
		this.abortController = null;
		if (this.pendingCompactionId) {
			await DatabaseService.deletePendingCompaction(this.pendingCompactionId);
			this.pendingCompactionId = null;
		}
		this.open = false;
	}
}

export const compactionStore = new CompactionStore();
