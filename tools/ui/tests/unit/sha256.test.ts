import { afterEach, describe, expect, it, vi } from 'vitest';
import { sha256 } from '$lib/utils/sha256';

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('sha256', () => {
	it('matches SHA-256 when Web Crypto is unavailable', async () => {
		vi.stubGlobal('crypto', undefined);

		await expect(sha256('abc')).resolves.toBe(
			'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
		);
		await expect(sha256('a'.repeat(1000))).resolves.toBe(
			'41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3'
		);
	});

	it('falls back when subtle.digest rejects', async () => {
		vi.stubGlobal('crypto', {
			subtle: {
				digest: vi.fn().mockRejectedValue(new Error('Unavailable in insecure context'))
			}
		});

		await expect(sha256('')).resolves.toBe(
			'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
		);
	});
});
