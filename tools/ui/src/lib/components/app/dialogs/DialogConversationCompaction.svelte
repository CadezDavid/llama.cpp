<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { AlertTriangle, ArchiveRestore, Minimize2 } from '@lucide/svelte';
	import { compactionStore } from '$lib/stores/compaction.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import CompactionRangeTimeline from './CompactionRangeTimeline.svelte';

	let observedConversationId = $state<string | null>(null);
	let observedMessageCount = $state(-1);

	$effect(() => {
		const id = conversationsStore.activeConversation?.id ?? null;
		const count = conversationsStore.activeMessages.length;
		if (id === observedConversationId && count === observedMessageCount) return;
		observedConversationId = id;
		observedMessageCount = count;
		void compactionStore.refreshActive();
	});

	function handleOpenChange(open: boolean) {
		if (!open) void compactionStore.close();
	}

	function formatTokens(value: number): string {
		return value.toLocaleString();
	}
</script>

<Dialog.Root open={compactionStore.open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-3xl" showCloseButton={!compactionStore.applying}>
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				{#if compactionStore.mode === 'view'}
					<ArchiveRestore class="h-5 w-5" />
					Compacted conversation state
				{:else if compactionStore.mode === 'ask'}
					<AlertTriangle class="h-5 w-5" />
					Conversation context is getting full
				{:else}
					<Minimize2 class="h-5 w-5" />
					Compact conversation
				{/if}
			</Dialog.Title>
			<Dialog.Description>
				Original messages remain unchanged. Only the context sent to the model is replaced.
			</Dialog.Description>
		</Dialog.Header>

		{#if compactionStore.loading}
			<div class="py-10 text-center text-sm text-muted-foreground">Measuring conversation...</div>
		{:else if compactionStore.error}
			<div
				class="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive"
			>
				{compactionStore.error}
			</div>
		{:else if compactionStore.mode === 'view' && compactionStore.activeCompaction}
			<div class="grid gap-4">
				<div class="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
					<div>
						<div class="text-muted-foreground">Generation</div>
						{compactionStore.activeCompaction.record.generation}
					</div>
					<div>
						<div class="text-muted-foreground">Model</div>
						{compactionStore.activeCompaction.record.modelId ?? 'Current server model'}
					</div>
					<div>
						<div class="text-muted-foreground">Source</div>
						{formatTokens(compactionStore.sourceTokenCount)} tokens
					</div>
					<div>
						<div class="text-muted-foreground">Projected</div>
						{formatTokens(compactionStore.projectedTokenCount)} tokens
					</div>
					<div>
						<div class="text-muted-foreground">Activation</div>
						{compactionStore.activeCompaction.record.activationMode ?? 'manual'}
					</div>
					<div>
						<div class="text-muted-foreground">Saved</div>
						{formatTokens(compactionStore.beforeTokenCount - compactionStore.projectedTokenCount)} tokens
					</div>
					<div>
						<div class="text-muted-foreground">Attempts</div>
						{compactionStore.activeCompaction.record.attemptCount ?? 1}
					</div>
					<div>
						<div class="text-muted-foreground">Duration</div>
						{compactionStore.activeCompaction.record.durationMs !== undefined
							? `${(compactionStore.activeCompaction.record.durationMs / 1000).toFixed(1)} s`
							: 'Not recorded'}
					</div>
				</div>
				<pre
					class="max-h-80 overflow-auto whitespace-pre-wrap rounded-md border bg-muted/30 p-4 text-xs">{compactionStore.summary}</pre>
				{#if compactionStore.restoreExceedsContext}
					<div class="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
						<div class="flex items-center gap-2 font-medium">
							<AlertTriangle class="h-4 w-4" />Restored history exceeds the current model context
						</div>
						<p class="mt-1 text-muted-foreground">
							The original path measures {formatTokens(compactionStore.restoredTokenCount)} tokens and
							may fail on the next request.
						</p>
					</div>
				{/if}
				<details class="rounded-md border p-3">
					<summary class="cursor-pointer text-sm font-medium"
						>View {compactionStore.sourceMessages.length} original source messages</summary
					>
					<div class="mt-3 max-h-72 space-y-3 overflow-auto">
						{#each compactionStore.sourceMessages as message (message.id)}
							<div class="rounded border bg-muted/20 p-3 text-xs">
								<div class="mb-1 font-medium uppercase text-muted-foreground">{message.role}</div>
								<div class="whitespace-pre-wrap">{message.content || '(empty)'}</div>
							</div>
						{/each}
					</div>
				</details>
			</div>
		{:else if compactionStore.mode === 'ask'}
			<div class="grid gap-4">
				{#if compactionStore.step === 'generate'}
					<div class="py-10 text-center text-sm text-muted-foreground">
						{compactionStore.operationStatus ?? 'Compacting conversation...'}
					</div>
				{:else}
					<div class="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
						<div>
							<div class="text-muted-foreground">Current prompt</div>
							{formatTokens(compactionStore.beforeTokenCount)}
						</div>
						<div>
							<div class="text-muted-foreground">Usable input</div>
							{formatTokens(compactionStore.policy?.usableInputTokens ?? 0)}
						</div>
						<div>
							<div class="text-muted-foreground">Used</div>
							{Math.round(compactionStore.preflightMeasurement?.utilizationPercent ?? 0)}%
						</div>
						<div>
							<div class="text-muted-foreground">Target</div>
							{compactionStore.policy?.targetPercent ?? 0}%
						</div>
					</div>
					<div class="grid gap-2">
						<div class="text-sm font-medium">Compact through turn</div>
						<CompactionRangeTimeline
							candidates={compactionStore.candidates}
							messages={compactionStore.messages}
							selectedIndex={compactionStore.selectedCandidateIndex}
							totalTokenCount={compactionStore.beforeTokenCount}
							protectedTurns={compactionStore.protectedTurnCount}
							protectedTokens={compactionStore.protectedTokenCount}
							onSelect={(index) => void compactionStore.selectCandidate(index)}
						/>
					</div>
					<p class="text-sm text-muted-foreground">
						Compaction keeps the original messages and replaces only the context sent to the model.
						The recent conversation remains literal.
					</p>
					{#if !compactionStore.canSkipAsk}
						<div class="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm">
							This request no longer fits with the configured answer reserve. It must be compacted
							or cancelled.
						</div>
					{/if}
				{/if}
			</div>
		{:else if compactionStore.mode === 'create'}
			<div class="grid gap-4">
				<div class="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
					<div>
						<div class="text-muted-foreground">Model context</div>
						{formatTokens(compactionStore.contextSize)}
					</div>
					<div>
						<div class="text-muted-foreground">Prompt before compaction</div>
						{formatTokens(compactionStore.beforeTokenCount)}
					</div>
					<div>
						<div class="text-muted-foreground">Original history selected</div>
						{formatTokens(compactionStore.sourceTokenCount)}
					</div>
					<div>
						<div class="text-muted-foreground">Recent history kept</div>
						{compactionStore.protectedTurnCount} turns /
						{formatTokens(compactionStore.protectedTokenCount)} tokens
					</div>
				</div>

				{#if compactionStore.step === 'measure'}
					<div class="grid gap-2">
						<div class="text-sm font-medium">Compact through turn</div>
						<CompactionRangeTimeline
							candidates={compactionStore.candidates}
							messages={compactionStore.messages}
							selectedIndex={compactionStore.selectedCandidateIndex}
							totalTokenCount={compactionStore.beforeTokenCount}
							protectedTurns={compactionStore.protectedTurnCount}
							protectedTokens={compactionStore.protectedTokenCount}
							onSelect={(index) => void compactionStore.selectCandidate(index)}
						/>
					</div>
				{:else if compactionStore.step === 'generate'}
					<div class="py-10 text-center text-sm text-muted-foreground">
						Generating structured conversation state...
					</div>
				{:else}
					<div class="grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
						<div>
							<div class="text-muted-foreground">Before</div>
							{formatTokens(compactionStore.beforeTokenCount)}
						</div>
						<div>
							<div class="text-muted-foreground">After</div>
							{formatTokens(compactionStore.projectedTokenCount)}
						</div>
						<div>
							<div class="text-muted-foreground">Saved</div>
							{formatTokens(compactionStore.beforeTokenCount - compactionStore.projectedTokenCount)}
						</div>
					</div>
					<pre
						class="max-h-80 overflow-auto whitespace-pre-wrap rounded-md border bg-muted/30 p-4 text-xs">{compactionStore.summary ||
							'(No valid compacted state)'}</pre>
					{#if compactionStore.validationErrors.length > 0}
						<div class="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
							<div class="mb-2 flex items-center gap-2 font-medium">
								<AlertTriangle class="h-4 w-4" />Cannot apply this result
							</div>
							<ul class="list-disc space-y-1 pl-5">
								{#each compactionStore.validationErrors as error (error)}<li>{error}</li>{/each}
							</ul>
						</div>
					{/if}
				{/if}
			</div>
		{/if}

		<Dialog.Footer>
			{#if compactionStore.mode !== 'ask'}
				<Button variant="outline" onclick={() => void compactionStore.close()}>Cancel</Button>
			{/if}
			{#if compactionStore.mode === 'view' && compactionStore.activeCompaction}
				<Button variant="outline" onclick={() => void compactionStore.openCreate()}
					>Recompact</Button
				>
				<Button variant="destructive" onclick={() => void compactionStore.restore()}
					>Restore original history</Button
				>
			{:else if compactionStore.mode === 'ask'}
				{#if compactionStore.error}
					<Button variant="outline" onclick={() => compactionStore.cancelAskCompaction()}
						>Cancel send</Button
					>
				{:else if compactionStore.step !== 'generate'}
					<Button variant="outline" onclick={() => compactionStore.cancelAskCompaction()}
						>Cancel send</Button
					>
					<Button
						variant="outline"
						disabled={!compactionStore.canSkipAsk}
						onclick={() => compactionStore.skipAskCompaction()}>Send without compacting</Button
					>
					<Button onclick={() => void compactionStore.confirmAskCompaction()}
						>Compact and send</Button
					>
				{/if}
			{:else if compactionStore.mode === 'create' && !compactionStore.error}
				{#if compactionStore.step === 'measure'}
					<Button onclick={() => void compactionStore.generate()}>Generate preview</Button>
				{:else if compactionStore.step === 'review'}
					<Button variant="outline" onclick={() => void compactionStore.generate()}>Retry</Button>
					<Button disabled={!compactionStore.canApply} onclick={() => void compactionStore.apply()}
						>{compactionStore.applying ? 'Applying...' : 'Apply compaction'}</Button
					>
				{/if}
			{/if}
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
