import type { DatabaseRetrievalTrace } from '$lib/types';

export function visibleTraceMessageId(
	trace: DatabaseRetrievalTrace,
	visibleMessageIds: ReadonlySet<string>
): string | null {
	if (trace.responseMessageId) {
		return visibleMessageIds.has(trace.responseMessageId) ? trace.responseMessageId : null;
	}
	return visibleMessageIds.has(trace.anchorMessageId) ? trace.anchorMessageId : null;
}
