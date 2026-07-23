<script lang="ts">
	import type { CompactionRangeCandidate, DatabaseMessage } from '$lib/types';

	interface Props {
		candidates: CompactionRangeCandidate[];
		messages: DatabaseMessage[];
		selectedIndex: number;
		totalTokenCount: number;
		protectedTurns: number;
		onSelect: (index: number) => void;
	}

	let { candidates, messages, selectedIndex, totalTokenCount, protectedTurns, onSelect }: Props =
		$props();

	let track: HTMLDivElement | null = $state(null);
	let dragging = $state(false);

	const selected = $derived(candidates[selectedIndex]);
	const largestSource = $derived(candidates.at(-1)?.sourceTokenCount ?? 0);
	const scaleTokens = $derived(Math.max(totalTokenCount, largestSource, 1));
	const selectedPercent = $derived(
		Math.min(100, ((selected?.sourceTokenCount ?? 0) / scaleTokens) * 100)
	);
	const protectedPercent = $derived(
		Math.max(0, 100 - Math.min(100, (largestSource / scaleTokens) * 100))
	);

	function endpoint(candidate: CompactionRangeCandidate): DatabaseMessage | undefined {
		return messages.find((message) => message.id === candidate.endMessageId);
	}

	function preview(message: DatabaseMessage | undefined): string {
		const normalized = message?.content.replace(/\s+/g, ' ').trim() || '(empty message)';
		return normalized.length > 30 ? `${normalized.slice(0, 30)}...` : normalized;
	}

	function pointPercent(candidate: CompactionRangeCandidate): number {
		return Math.min(100, (candidate.sourceTokenCount / scaleTokens) * 100);
	}

	function pointLabel(candidate: CompactionRangeCandidate): string {
		const message = endpoint(candidate);
		return `Turn ${candidate.turnCount}, ${message?.role ?? 'message'}: ${preview(message)}. ${candidate.sourceMessageIds.length} messages, ${candidate.sourceTokenCount.toLocaleString()} tokens.`;
	}

	function selectFromClientX(clientX: number): void {
		if (!track || candidates.length === 0) return;
		const bounds = track.getBoundingClientRect();
		const ratio = Math.max(0, Math.min(1, (clientX - bounds.left) / bounds.width));
		const targetTokens = ratio * scaleTokens;
		let nearest = 0;
		let distance = Number.POSITIVE_INFINITY;
		for (let index = 0; index < candidates.length; index++) {
			const current = Math.abs(candidates[index].sourceTokenCount - targetTokens);
			if (current < distance) {
				distance = current;
				nearest = index;
			}
		}
		onSelect(nearest);
	}

	function handlePointerDown(event: PointerEvent): void {
		dragging = true;
		track?.setPointerCapture(event.pointerId);
		selectFromClientX(event.clientX);
	}

	function handlePointerMove(event: PointerEvent): void {
		if (dragging) selectFromClientX(event.clientX);
	}

	function handlePointerEnd(event: PointerEvent): void {
		dragging = false;
		if (track?.hasPointerCapture(event.pointerId)) track.releasePointerCapture(event.pointerId);
	}

	function handleKeydown(event: KeyboardEvent): void {
		let next = selectedIndex;
		if (event.key === 'ArrowLeft') next--;
		else if (event.key === 'ArrowRight') next++;
		else if (event.key === 'Home') next = 0;
		else if (event.key === 'End') next = candidates.length - 1;
		else return;
		event.preventDefault();
		onSelect(Math.max(0, Math.min(candidates.length - 1, next)));
	}
</script>

{#if selected}
	<div class="grid gap-3">
		<div
			bind:this={track}
			role="slider"
			tabindex="0"
			aria-label="Compact conversation through turn"
			aria-valuemin="1"
			aria-valuemax={candidates.length}
			aria-valuenow={selectedIndex + 1}
			aria-valuetext={pointLabel(selected)}
			class="relative h-12 touch-none select-none rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring"
			onkeydown={handleKeydown}
			onpointerdown={handlePointerDown}
			onpointermove={handlePointerMove}
			onpointerup={handlePointerEnd}
			onpointercancel={handlePointerEnd}
		>
			<div class="absolute left-0 right-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-muted">
				<div
					class="h-full rounded-l-full bg-primary transition-[width]"
					style:width={`${selectedPercent}%`}
				></div>
				{#if protectedPercent > 0}
					<div
						class="absolute right-0 top-0 h-full rounded-r-full bg-amber-500/25"
						style:width={`${protectedPercent}%`}
						title={`${protectedTurns} recent turns remain literal`}
					></div>
				{/if}
			</div>

			{#each candidates as candidate, index (candidate.endMessageId)}
				<button
					type="button"
					class="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-background shadow-sm focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-ring {index <=
					selectedIndex
						? 'bg-primary'
						: 'bg-muted-foreground'}"
					style:left={`${pointPercent(candidate)}%`}
					aria-label={pointLabel(candidate)}
					title={pointLabel(candidate)}
					onclick={(event) => {
						event.stopPropagation();
						onSelect(index);
					}}
				></button>
			{/each}
		</div>

		<div class="grid grid-cols-2 gap-3 rounded-md border bg-muted/20 p-3 text-sm sm:grid-cols-4">
			<div>
				<div class="text-muted-foreground">Compact</div>
				{selected.turnCount} turns
			</div>
			<div>
				<div class="text-muted-foreground">Messages</div>
				{selected.sourceMessageIds.length}
			</div>
			<div>
				<div class="text-muted-foreground">Source tokens</div>
				{selected.sourceTokenCount.toLocaleString()}
			</div>
			<div>
				<div class="text-muted-foreground">Recent history kept</div>
				{protectedTurns}+ turns
			</div>
		</div>

		<p class="text-xs text-muted-foreground">
			Drag or use the arrow keys to choose a safe boundary. The amber section remains literal. Tool
			calls and results stay together.
		</p>
	</div>
{/if}
