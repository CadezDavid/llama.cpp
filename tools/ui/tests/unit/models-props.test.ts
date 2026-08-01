import { afterEach, describe, expect, it, vi } from 'vitest';
import { PropsService } from '$lib/services/props.service';
import { modelsStore } from '$lib/stores/models.svelte';
import type { ApiLlamaCppServerProps } from '$lib/types';

describe('modelsStore model props', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		modelsStore.clear();
	});

	it('shares an in-flight props request with concurrent callers', async () => {
		let resolveRequest: ((props: ApiLlamaCppServerProps) => void) | undefined;
		const request = new Promise<ApiLlamaCppServerProps>((resolve) => {
			resolveRequest = resolve;
		});
		const fetch = vi.spyOn(PropsService, 'fetchForModel').mockReturnValue(request);

		const first = modelsStore.fetchModelProps('Gemma 4 31B');
		const second = modelsStore.fetchModelProps('Gemma 4 31B');
		expect(fetch).toHaveBeenCalledOnce();
		expect(modelsStore.isModelPropsFetching('Gemma 4 31B')).toBe(true);

		const props = {
			default_generation_settings: { n_ctx: 98_304 }
		} as ApiLlamaCppServerProps;
		resolveRequest?.(props);

		await expect(first).resolves.toBe(props);
		await expect(second).resolves.toBe(props);
		expect(modelsStore.getModelContextSize('Gemma 4 31B')).toBe(98_304);
		expect(modelsStore.isModelPropsFetching('Gemma 4 31B')).toBe(false);
	});
});
