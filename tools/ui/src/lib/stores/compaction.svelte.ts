import { toast } from 'svelte-sonner';
import { isAbortError } from '$lib/utils/abort';
import { isRouterMode, contextSize as serverContextSize } from '$lib/stores/server.svelte';
import {
	modelsStore,
	selectedModelContextSize,
	selectedModelName
} from '$lib/stores/models.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
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
	DatabaseCompaction,
	DatabaseMessage,
	ResolvedCompaction
} from '$lib/types';

export type CompactionDialogMode = 'create' | 'view';
export type CompactionDialogStep = 'measure' | 'generate' | 'review';

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
	private abortController: AbortController | null = null;
	private pendingCompactionId: string | null = null;

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
			this.candidates = CompactionService.buildRangeCandidates(
				this.messages,
				this.activeCompaction?.record
			);
			if (this.mode === 'create') {
				if (this.candidates.length === 0) {
					throw new Error('There are not enough complete old turns to compact safely');
				}
				await this.selectBestCandidate();
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
		const source = this.messages.filter((message) =>
			candidate.sourceMessageIds.includes(message.id)
		);
		this.sourceTokenCount = await ChatService.tokenizePrompt(
			CompactionService.serializeMessages(source),
			this.model
		);
		candidate.sourceTokenCount = this.sourceTokenCount;
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
				generation: (previous?.generation ?? 0) + 1
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
		try {
			await DatabaseService.activateCompaction(
				this.pendingCompactionId,
				{ summary: this.summary, projectedTokenCount: this.projectedTokenCount },
				conversation.currNode
			);
			this.pendingCompactionId = null;
			await this.refreshActive();
			this.open = false;
			toast.success('Conversation compacted');
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.applying = false;
		}
	}

	async restore(): Promise<void> {
		const conversation = conversationsStore.activeConversation;
		if (!conversation?.currNode) return;
		await DatabaseService.createCompactionRestoreEvent(conversation.id, conversation.currNode);
		await this.refreshActive();
		this.open = false;
		toast.success('Original conversation context restored');
	}

	async close(): Promise<void> {
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
