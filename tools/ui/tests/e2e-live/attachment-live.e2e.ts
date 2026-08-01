import { expect, test, type Page, type TestInfo } from '@playwright/test';

const serverURL = process.env.LLAMA_SERVER_URL;
const apiKey = process.env.LLAMA_API_KEY;
const model = process.env.LLAMA_TEST_MODEL;
const embeddingModel = process.env.LLAMA_TEST_EMBEDDING_MODEL;
const attachmentFile = process.env.ATTACHMENT_TEST_FILE;

const requiredEnvironment = {
	LLAMA_SERVER_URL: serverURL,
	LLAMA_TEST_MODEL: model,
	LLAMA_TEST_EMBEDDING_MODEL: embeddingModel,
	ATTACHMENT_TEST_FILE: attachmentFile
};

function missingEnvironment(): string[] {
	return Object.entries(requiredEnvironment)
		.filter(([, value]) => !value)
		.map(([name]) => name);
}

async function attachFailureArtifacts(
	page: Page,
	testInfo: TestInfo,
	consoleMessages: string[],
	failedResponses: string[]
): Promise<void> {
	await testInfo.attach('browser-console', {
		body: consoleMessages.join('\n'),
		contentType: 'text/plain'
	});
	await testInfo.attach('failed-responses', {
		body: failedResponses.join('\n'),
		contentType: 'text/plain'
	});
	const diagnosticsButton = page.getByRole('button', {
		name: 'Copy safe diagnostics',
		exact: true
	});
	if (await diagnosticsButton.isVisible().catch(() => false)) {
		await diagnosticsButton.click();
		const diagnostics = await page.evaluate(() => navigator.clipboard.readText());
		await testInfo.attach('attachment-diagnostics', {
			body: diagnostics,
			contentType: 'application/json'
		});
	}
}

test.describe('live attachment pipeline', () => {
	test.skip(missingEnvironment().length > 0, `Missing: ${missingEnvironment().join(', ')}`);

	test('waits for a model, summarizes, indexes, and exposes attachment tools', async ({
		page,
		request
	}, testInfo) => {
		const consoleMessages: string[] = [];
		const failedResponses: string[] = [];
		const chatRequests: unknown[] = [];
		page.on('console', (message) => consoleMessages.push(`${message.type()}: ${message.text()}`));
		page.on('response', async (response) => {
			if (response.status() >= 400) {
				failedResponses.push(`${response.status()} ${response.url()}`);
			}
			if (
				response.url().endsWith('/v1/chat/completions') &&
				response.request().method() === 'POST'
			) {
				chatRequests.push(response.request().postDataJSON());
			}
		});

		try {
			const headers = apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined;
			await request.post(`${serverURL}/models/unload`, {
				headers,
				data: { model }
			});
			await page.addInitScript(
				({ apiKey, embeddingModel }) => {
					const key = 'LlamaUi.config';
					const current = JSON.parse(localStorage.getItem(key) ?? '{}');
					localStorage.setItem(
						key,
						JSON.stringify({
							...current,
							...(apiKey ? { apiKey } : {}),
							embeddingModel,
							pdfAsImage: false
						})
					);
				},
				{ apiKey, embeddingModel }
			);
			await page.goto('/');

			const fileInput = page.locator('input[type="file"]');
			await fileInput.setInputFiles(attachmentFile!);
			const attachment = page.locator('[data-attachment-stage]').first();
			await expect(attachment).toHaveAttribute('data-attachment-stage', 'waiting-model', {
				timeout: 30_000
			});

			await page.evaluate(() => {
				const stages: string[] = [];
				const collect = () => {
					const stage = document
						.querySelector('[data-attachment-stage]')
						?.getAttribute('data-attachment-stage');
					if (stage && stages.at(-1) !== stage) stages.push(stage);
				};
				collect();
				new MutationObserver(collect).observe(document.body, {
					attributes: true,
					childList: true,
					subtree: true,
					attributeFilter: ['data-attachment-stage']
				});
				Object.assign(window, { __attachmentStages: stages });
			});

			await page.locator('.chat-screen-form-wrapper button[aria-haspopup="menu"]').last().click();
			await page.getByPlaceholder('Search models...').fill(model!);
			await page.getByRole('option').filter({ hasText: model! }).click();

			await expect(attachment).toHaveAttribute('data-attachment-stage', 'ready');
			await expect(page.getByText('Indexed', { exact: true })).toBeVisible();
			const stages = await page.evaluate(
				() => (window as typeof window & { __attachmentStages?: string[] }).__attachmentStages ?? []
			);
			expect(stages).toEqual(
				expect.arrayContaining(['waiting-model', 'measuring', 'summarizing', 'indexing', 'ready'])
			);

			const textarea = page.locator('.chat-screen-form-wrapper textarea');
			await textarea.fill(
				'Use attachment_search to find the main result, then use attachment_read and answer in one sentence.'
			);
			await textarea.press('Enter');
			await expect
				.poll(
					() =>
						chatRequests.some((body) => {
							const tools = (body as { tools?: Array<{ function?: { name?: string } }> }).tools;
							const names = tools?.map((tool) => tool.function?.name) ?? [];
							return names.includes('attachment_search') && names.includes('attachment_read');
						}),
					{ timeout: 5 * 60 * 1000 }
				)
				.toBe(true);
			await expect
				.poll(
					() =>
						chatRequests.some((body) =>
							(
								body as {
									messages?: Array<{ role?: string }>;
								}
							).messages?.some((message) => message.role === 'tool')
						),
					{ timeout: 5 * 60 * 1000 }
				)
				.toBe(true);
		} catch (error) {
			await attachFailureArtifacts(page, testInfo, consoleMessages, failedResponses);
			throw error;
		}
	});
});
