import { SETTINGS_KEYS } from '$lib/constants';
import { config } from '$lib/stores/settings.svelte';

const SENSITIVE_KEY =
	/(authorization|api.?key|api.?token|access.?token|secret|password|bearer|^(content|text|query|summary|prompt|embedding|embeddings)$|(content|text|query|summary|prompt|embedding|embeddings)(text|value|data|body)$)/i;

function sanitize(value: unknown, key = '', depth = 0): unknown {
	if (SENSITIVE_KEY.test(key)) return '[redacted]';
	if (depth >= 4) return '[truncated]';
	if (value instanceof Error) {
		return { name: value.name, message: value.message };
	}
	if (Array.isArray(value)) {
		return value.map((item) => sanitize(item, '', depth + 1));
	}
	if (value && typeof value === 'object') {
		return Object.fromEntries(
			Object.entries(value).map(([childKey, childValue]) => [
				childKey,
				sanitize(childValue, childKey, depth + 1)
			])
		);
	}
	return value;
}

export function memoryDebug(event: string, details: Record<string, unknown> = {}): void {
	if (!config()[SETTINGS_KEYS.MEMORY_DEBUG_LOGGING]) return;
	console.debug(`[Memory] ${event}`, sanitize(details));
}

export function sanitizeMemoryDebugDetails(
	details: Record<string, unknown>
): Record<string, unknown> {
	return sanitize(details) as Record<string, unknown>;
}
