<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { SpominService } from '$lib/services';
	import { DatabaseService } from '$lib/services/database.service';
	import { config } from '$lib/stores/settings.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import type { DatabaseRetrievalTrace, MemoryRecord } from '$lib/types';

	let records = $state<MemoryRecord[]>([]);
	let loading = $state(false);
	let error = $state<string | null>(null);
	let reachable = $state<boolean | null>(null);
	let text = $state('');
	let project = $state('');
	let tier = $state<'archive' | 'core'>('archive');
	let conversationProject = $state(conversationsStore.activeConversation?.memoryProject ?? '');
	let traces = $state<DatabaseRetrievalTrace[]>([]);

	function client(): SpominService {
		const current = config();
		return new SpominService({
			baseUrl: String(current.spominBaseUrl || 'http://127.0.0.1:8084'),
			apiToken: String(current.spominApiToken || ''),
			timeoutMs: Number(current.spominTimeoutMs) || 2000
		});
	}

	async function refresh(): Promise<void> {
		loading = true;
		error = null;
		const service = client();
		try {
			if (conversationsStore.activeConversation) {
				traces = await DatabaseService.getRetrievalTraces(
					conversationsStore.activeConversation.id,
					10
				);
			}
			reachable = await service.health();
			if (!reachable) throw new Error('Spomin is not reachable');
			records = await service.list({ limit: 50, project: String(config().spominProject || '') });
		} catch (reason) {
			error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			loading = false;
		}
	}

	async function create(): Promise<void> {
		if (!text.trim()) return;
		loading = true;
		error = null;
		try {
			await client().create({ text: text.trim(), project: project.trim() || undefined, tier });
			text = '';
			await refresh();
		} catch (reason) {
			error = reason instanceof Error ? reason.message : String(reason);
			loading = false;
		}
	}

	async function save(record: MemoryRecord): Promise<void> {
		loading = true;
		error = null;
		try {
			await client().update(record.id, {
				text: record.text.trim(),
				project: record.project?.trim() || null,
				tier: record.tier
			});
			await refresh();
		} catch (reason) {
			error = reason instanceof Error ? reason.message : String(reason);
			loading = false;
		}
	}

	async function remove(record: MemoryRecord): Promise<void> {
		if (!confirm('Delete this memory permanently?')) return;
		loading = true;
		error = null;
		try {
			await client().delete(record.id);
			records = records.filter((item) => item.id !== record.id);
		} catch (reason) {
			error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			loading = false;
		}
	}
</script>

<section class="space-y-4 border-t border-border/30 pt-6">
	{#if conversationsStore.activeConversation}
		<div class="space-y-2 rounded-md border p-3">
			<label class="text-sm font-medium" for="conversation-memory-project">
				Current conversation project override
			</label>
			<div class="flex gap-2">
				<Input
					id="conversation-memory-project"
					bind:value={conversationProject}
					placeholder={String(config().spominProject || 'Use global default')}
				/>
				<Button
					variant="outline"
					onclick={() => void conversationsStore.setMemoryProject(conversationProject)}
				>
					Save override
				</Button>
			</div>
		</div>
	{/if}

	<div class="flex items-center justify-between gap-3">
		<div>
			<h4 class="font-medium">Spomin memory manager</h4>
			<p class="text-xs text-muted-foreground">
				Memories are only written, changed, or deleted when you use these controls.
			</p>
		</div>
		<Button variant="outline" disabled={loading} onclick={() => void refresh()}>
			{loading ? 'Working...' : 'Connect and refresh'}
		</Button>
	</div>

	{#if reachable !== null}
		<p class="text-xs {reachable ? 'text-green-600' : 'text-destructive'}">
			{reachable ? 'Spomin connected' : 'Spomin unavailable'}
		</p>
	{/if}
	{#if error}<p class="text-sm text-destructive">{error}</p>{/if}

	<div class="space-y-2 rounded-md border p-3">
		<textarea
			class="min-h-20 w-full rounded-md border bg-background p-2 text-sm"
			bind:value={text}
			placeholder="Add one durable fact, preference, or project detail"
		></textarea>
		<div class="flex flex-wrap gap-2">
			<Input class="min-w-48 flex-1" bind:value={project} placeholder="Project (optional)" />
			<select class="rounded-md border bg-background px-3 text-sm" bind:value={tier}>
				<option value="archive">Archive</option>
				<option value="core">Core</option>
			</select>
			<Button disabled={loading || !text.trim()} onclick={() => void create()}>Add memory</Button>
		</div>
	</div>

	{#each records as record (record.id)}
		<div class="space-y-2 rounded-md border p-3">
			<textarea
				class="min-h-20 w-full rounded-md border bg-background p-2 text-sm"
				bind:value={record.text}
			></textarea>
			<div class="flex flex-wrap gap-2">
				<Input class="min-w-48 flex-1" bind:value={record.project} placeholder="Global" />
				<select class="rounded-md border bg-background px-3 text-sm" bind:value={record.tier}>
					<option value="archive">Archive</option>
					<option value="core">Core</option>
				</select>
				<Button variant="outline" disabled={loading} onclick={() => void save(record)}>Save</Button>
				<Button variant="destructive" disabled={loading} onclick={() => void remove(record)}>
					Delete
				</Button>
			</div>
			<p class="truncate text-xs text-muted-foreground" title={record.id}>{record.id}</p>
		</div>
	{/each}

	{#if traces.length}
		<div class="space-y-2 border-t border-border/30 pt-4">
			<h4 class="font-medium">Recent recall decisions</h4>
			{#each traces as trace (trace.id)}
				<details class="rounded-md border p-3 text-xs">
					<summary class="cursor-pointer">
						{new Date(trace.createdAt).toLocaleString()} - {trace.injectedHitIds.length} injected,
						{trace.injectedTokenCount} tokens
					</summary>
					<p class="mt-2 whitespace-pre-wrap text-muted-foreground">{trace.query}</p>
					<p class="mt-2">Providers: {JSON.stringify(trace.providers)}</p>
					<ul class="mt-2 space-y-1">
						{#each trace.hits as hit (`${hit.source}:${hit.id}`)}
							<li>
								{hit.selected ? 'injected' : `skipped (${hit.reason ?? 'ranking'})`} -
								{hit.source}:{hit.id} - score {hit.score.toFixed(3)}
							</li>
						{/each}
					</ul>
				</details>
			{/each}
		</div>
	{/if}
</section>
