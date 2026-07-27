<script lang="ts">
	import { copyToClipboard } from '$lib/utils';
	import type { DatabaseRetrievalTrace, RetrievalTraceHit } from '$lib/types';

	interface Props {
		traces?: DatabaseRetrievalTrace[];
	}

	let { traces = [] }: Props = $props();

	const orderedTraces = $derived(
		[...traces].sort((left, right) => left.createdAt - right.createdAt)
	);

	function injected(trace: DatabaseRetrievalTrace): RetrievalTraceHit[] {
		const ids = new Set(trace.injectedHitIds);
		return trace.hits.filter((hit) => ids.has(hit.id));
	}

	function skipped(trace: DatabaseRetrievalTrace): RetrievalTraceHit[] {
		const ids = new Set(trace.injectedHitIds);
		return trace.hits.filter((hit) => !ids.has(hit.id));
	}

	function providerProblem(trace: DatabaseRetrievalTrace): string | null {
		for (const [name, provider] of Object.entries(trace.providers)) {
			if (provider.status === 'timeout') return `${providerLabel(name)} timed out`;
			if (provider.status === 'error') return `${providerLabel(name)} unavailable`;
		}
		return null;
	}

	function summary(trace: DatabaseRetrievalTrace): string {
		const count = injected(trace).length;
		if (count > 0) {
			return `Memory context: ${count} ${count === 1 ? 'item' : 'items'}, ${trace.injectedTokenCount.toLocaleString()} tokens`;
		}
		const problem = providerProblem(trace);
		if (problem) return `Memory context: unavailable - ${problem}`;
		const enabled = Object.values(trace.providers).some((provider) =>
			['ok', 'timeout', 'error'].includes(provider.status)
		);
		return enabled ? 'Memory context: none - no relevant matches' : 'Memory context: disabled';
	}

	function providerLabel(value: string): string {
		if (value === 'spomin') return 'Spomin';
		if (value === 'local') return 'Conversation recall';
		return value;
	}

	function providerStatus(value: string): string {
		return value === 'not-applicable' ? 'not applicable' : value;
	}

	function sourceLabel(hit: RetrievalTraceHit): string {
		return hit.source === 'long-term-memory' ? 'Spomin memory' : 'Compacted conversation';
	}

	function value(provenance: Record<string, unknown> | undefined, key: string): string | null {
		const item = provenance?.[key];
		if (typeof item === 'string' || typeof item === 'number') return String(item);
		if (Array.isArray(item)) return item.filter((entry) => typeof entry === 'string').join(', ');
		return null;
	}

	function score(hit: RetrievalTraceHit): string {
		return Number.isFinite(hit.score) ? hit.score.toFixed(3) : 'n/a';
	}

	function timestamp(value: number): string {
		return new Date(value).toLocaleString();
	}

	async function copyDiagnostics(trace: DatabaseRetrievalTrace): Promise<void> {
		const diagnostic = {
			id: trace.id,
			createdAt: new Date(trace.createdAt).toISOString(),
			anchorMessageId: trace.anchorMessageId,
			responseMessageId: trace.responseMessageId,
			reusedFromTraceId: trace.reusedFromTraceId,
			providers: trace.providers,
			injectedTokenCount: trace.injectedTokenCount,
			finalPromptTokenCount: trace.finalPromptTokenCount,
			hits: trace.hits.map(
				({
					id,
					source,
					score,
					semanticScore,
					providerScore,
					selected,
					reason,
					tokenCount
				}) => ({
					id,
					source,
					score,
					semanticScore,
					providerScore,
					selected,
					reason,
					tokenCount
				})
			)
		};
		await copyToClipboard(JSON.stringify(diagnostic, null, 2), 'Memory diagnostics copied');
	}
</script>

{#if orderedTraces.length > 0}
	<div class="w-full max-w-[80%] text-xs text-muted-foreground">
		{#each orderedTraces as trace, index (trace.id)}
			<details class="group rounded-md border border-border/60 bg-muted/15">
				<summary
					class="cursor-pointer list-none px-3 py-2 outline-none marker:hidden focus-visible:ring-2 focus-visible:ring-ring"
				>
					<span class="font-medium text-foreground/75">{summary(trace)}</span>
					{#if orderedTraces.length > 1}
						<span class="ml-2">Request {index + 1}</span>
					{/if}
				</summary>

				<div class="grid gap-3 border-t border-border/60 px-3 py-3">
					<div class="flex flex-wrap items-center justify-between gap-2">
						<span>{timestamp(trace.createdAt)}</span>
						<button
							type="button"
							class="rounded border px-2 py-1 hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring"
							onclick={() => void copyDiagnostics(trace)}
						>
							Copy safe diagnostics
						</button>
					</div>

					{#if trace.reusedFromTraceId}
						<div class="rounded border bg-background/50 p-2">
							Reused the original memory selection; providers were not queried again.
						</div>
					{/if}

					<div class="grid gap-1 rounded border bg-background/50 p-2">
						<div class="font-medium text-foreground/80">Providers</div>
						{#each Object.entries(trace.providers) as [name, provider] (name)}
							<div>
								{providerLabel(name)}: {providerStatus(provider.status)}{provider.detail
									? ` (${provider.detail})`
									: ''}
							</div>
						{/each}
					</div>

					{#if injected(trace).length > 0}
						<div class="grid gap-2">
							<div class="font-medium text-foreground/80">Sent to the model</div>
							{#each injected(trace) as hit (`${hit.source}:${hit.id}`)}
								<div class="grid gap-2 rounded border bg-background/70 p-3">
									<div class="flex flex-wrap justify-between gap-2">
										<span class="font-medium text-foreground/80">{sourceLabel(hit)}</span>
										<span>score {score(hit)} | {hit.tokenCount ?? 0} tokens</span>
									</div>
									<div class="break-all">
										ID: {value(hit.provenance, 'chunkId') ??
											value(hit.provenance, 'memoryIds') ??
											hit.id}
										{#if value(hit.provenance, 'project')}
											| project: {value(hit.provenance, 'project')}
										{/if}
									</div>
									{#if hit.contentSnapshot}
										<div class="whitespace-pre-wrap break-words text-foreground/85">
											{hit.contentSnapshot}
										</div>
									{:else}
										<div class="italic">Text snapshot is unavailable for this legacy request.</div>
									{/if}
									<div class="italic">
										Historical request snapshot; the original source may have changed or been
										deleted.
									</div>
								</div>
							{/each}
						</div>
					{/if}

					{#if skipped(trace).length > 0}
						<details class="rounded border bg-background/40 p-2">
							<summary class="cursor-pointer font-medium text-foreground/80">
								Skipped candidates ({skipped(trace).length})
							</summary>
							<div class="mt-2 grid gap-1">
								{#each skipped(trace) as hit (`${hit.source}:${hit.id}`)}
									<div class="break-all">
										{sourceLabel(hit)} | {hit.id} | score {score(hit)} | {hit.reason ??
											'not selected'} | {hit.tokenCount ?? 0} tokens
									</div>
								{/each}
							</div>
						</details>
					{/if}
				</div>
			</details>
		{/each}
	</div>
{/if}
