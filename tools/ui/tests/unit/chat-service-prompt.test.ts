import { afterEach, describe, expect, it, vi } from 'vitest';
import { ContentPartType, MessageRole } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import type { ApiChatMessageData } from '$lib/types';
import type { DatabaseMessage } from '$lib/types';

describe('ChatService prompt requests', () => {
	afterEach(() => {
		ChatService.clearPromptMeasurementCache();
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('reuses exact prompt measurements and reports cache metrics', async () => {
		const fetchMock = vi
			.fn()
			.mockResolvedValueOnce(new Response(JSON.stringify({ prompt: '<bos>Hello' })))
			.mockResolvedValueOnce(new Response(JSON.stringify({ tokens: [1, 2, 3] })));
		vi.stubGlobal('fetch', fetchMock);
		const messages = [{ role: MessageRole.USER, content: 'Cache me' }];

		await ChatService.measurePrompt(messages);
		await ChatService.measurePrompt(messages);

		expect(fetchMock).toHaveBeenCalledTimes(2);
		expect(ChatService.getPromptMeasurementCacheMetrics()).toEqual({
			hits: 1,
			misses: 1,
			size: 1
		});
	});

	it('uses one request builder for template and completion settings', () => {
		const request = ChatService.buildChatCompletionRequest(
			[{ role: MessageRole.USER, content: 'Hello' }],
			{
				model: 'router-model',
				stream: true,
				enableThinking: true,
				continueFinalMessage: true,
				samplers: 'top_k;top_p',
				custom: { chat_template_kwargs: { enable_thinking: true, custom_flag: true } }
			}
		);

		expect(request).toMatchObject({
			model: 'router-model',
			stream: true,
			return_progress: true,
			continue_final_message: true,
			add_generation_prompt: false,
			samplers: ['top_k', 'top_p'],
			chat_template_kwargs: { enable_thinking: true, custom_flag: true }
		});
	});

	it('applies the template and tokenizes it without adding special tokens twice', async () => {
		const fetchMock = vi
			.fn()
			.mockResolvedValueOnce(
				new Response(JSON.stringify({ prompt: '<bos>Hello' }), {
					status: 200,
					headers: { 'content-type': 'application/json' }
				})
			)
			.mockResolvedValueOnce(
				new Response(JSON.stringify({ tokens: [1, 2, 3] }), {
					status: 200,
					headers: { 'content-type': 'application/json' }
				})
			);
		vi.stubGlobal('fetch', fetchMock);

		const measurement = await ChatService.measurePrompt(
			[{ role: MessageRole.USER, content: 'Hello' }],
			{ model: 'router-model', enableThinking: false }
		);

		expect(measurement).toEqual({
			tokenCount: 3,
			hasNonTextContent: false,
			exactForTextOnly: true
		});
		expect(fetchMock).toHaveBeenCalledTimes(2);
		expect(fetchMock.mock.calls[0][0]).toContain('apply-template');
		expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toMatchObject({
			model: 'router-model',
			messages: [{ role: MessageRole.USER, content: 'Hello' }]
		});
		expect(fetchMock.mock.calls[1][0]).toContain('tokenize');
		expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({
			content: '<bos>Hello',
			add_special: false,
			parse_special: true,
			model: 'router-model'
		});
	});

	it('marks multimodal counts as text-only measurements', async () => {
		const fetchMock = vi
			.fn()
			.mockResolvedValueOnce(
				new Response(JSON.stringify({ prompt: '<image>Describe' }), { status: 200 })
			)
			.mockResolvedValueOnce(new Response(JSON.stringify({ tokens: [1, 2] }), { status: 200 }));
		vi.stubGlobal('fetch', fetchMock);
		const messages: ApiChatMessageData[] = [
			{
				role: MessageRole.USER,
				content: [
					{ type: ContentPartType.TEXT, text: 'Describe' },
					{ type: ContentPartType.IMAGE_URL, image_url: { url: 'data:image/png;base64,AA==' } }
				]
			}
		];

		await expect(ChatService.measurePrompt(messages)).resolves.toMatchObject({
			hasNonTextContent: true,
			exactForTextOnly: false
		});
	});

	it('uses the model from the latest assistant response', () => {
		const messages = [
			{ role: MessageRole.ASSISTANT, model: 'older-model' },
			{ role: MessageRole.USER },
			{ role: MessageRole.ASSISTANT, model: 'current-model' }
		] as DatabaseMessage[];

		expect(ChatService.findLatestAssistantModel(messages)).toBe('current-model');
	});
});
