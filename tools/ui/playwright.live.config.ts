import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.LLAMA_SERVER_URL;

export default defineConfig({
	testDir: 'tests/e2e-live',
	testMatch: ['**/*.e2e.ts'],
	timeout: 10 * 60 * 1000,
	expect: {
		timeout: 5 * 60 * 1000
	},
	fullyParallel: false,
	forbidOnly: !!process.env.CI,
	retries: 0,
	workers: 1,
	reporter: 'line',
	use: {
		baseURL,
		trace: 'retain-on-failure',
		screenshot: 'only-on-failure',
		video: 'retain-on-failure'
	},
	projects: [
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'] }
		}
	]
});
