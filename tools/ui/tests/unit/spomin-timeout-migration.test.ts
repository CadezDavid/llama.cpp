import { beforeEach, describe, expect, it } from 'vitest';
import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants';
import { SETTINGS_KEYS } from '$lib/constants/settings-keys';
import { MigrationService } from '$lib/services/migration.service';

const storage = new Map<string, string>();
const localStoragePolyfill = {
	getItem: (key: string) => storage.get(key) ?? null,
	setItem: (key: string, value: string) => storage.set(key, value),
	removeItem: (key: string) => storage.delete(key),
	clear: () => storage.clear(),
	key: (index: number) => [...storage.keys()][index] ?? null,
	get length() {
		return storage.size;
	}
} as Storage;

Object.defineProperty(globalThis, 'localStorage', {
	value: localStoragePolyfill,
	configurable: true
});

describe('Spomin timeout migration', () => {
	const migration = MigrationService.getMigrations().find(
		(item) => item.id === 'spomin-timeout-default-v1'
	);

	beforeEach(() => storage.clear());

	it('raises the persisted old default to 2000 ms', async () => {
		localStorage.setItem(
			CONFIG_LOCALSTORAGE_KEY,
			JSON.stringify({ [SETTINGS_KEYS.SPOMIN_TIMEOUT_MS]: 750 })
		);

		await migration?.run();

		expect(JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}')).toMatchObject({
			[SETTINGS_KEYS.SPOMIN_TIMEOUT_MS]: 2000
		});
	});

	it('preserves a custom timeout', async () => {
		localStorage.setItem(
			CONFIG_LOCALSTORAGE_KEY,
			JSON.stringify({ [SETTINGS_KEYS.SPOMIN_TIMEOUT_MS]: 3500 })
		);

		await migration?.run();

		expect(JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}')).toMatchObject({
			[SETTINGS_KEYS.SPOMIN_TIMEOUT_MS]: 3500
		});
	});
});
