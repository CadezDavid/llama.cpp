import { describe, expect, it } from 'vitest';
import { isModelSelectionAvailable } from '$lib/components/app/models/utils';
import type { ModelOption } from '$lib/types/models';

const options = [
	{ id: 'gemma-id', model: 'Gemma 4 31B' },
	{ id: 'embed-id', model: 'Jina Embeddings' }
] as ModelOption[];

describe('isModelSelectionAvailable', () => {
	it('uses the explicit selection instead of an unavailable previous response model', () => {
		expect(isModelSelectionAvailable(options, 'gemma-id', 'Qwen3.6 27B')).toBe(true);
	});

	it('falls back to the conversation model when there is no explicit selection', () => {
		expect(isModelSelectionAvailable(options, null, 'Qwen3.6 27B')).toBe(false);
		expect(isModelSelectionAvailable(options, null, 'Gemma 4 31B')).toBe(true);
	});
});
