import { describe, expect, it } from 'vitest';
import { AttachmentType, ContentPartType, MessageRole, MessageType } from '$lib/enums';
import { ChatContextService } from '$lib/services/chat-context.service';
import type { ApiChatMessageContentPart, ChatPromptProjection, DatabaseMessage } from '$lib/types';

function message(
	id: string,
	role: MessageRole,
	content: string,
	overrides: Partial<DatabaseMessage> = {}
): DatabaseMessage {
	return {
		id,
		convId: 'conversation-1',
		type: MessageType.TEXT,
		timestamp: Number(id.replace(/\D/g, '')) || 1,
		role,
		content,
		parent: null,
		children: [],
		...overrides
	};
}

describe('ChatContextService', () => {
	it('preserves ordinary chat behavior without mutating transcript messages', async () => {
		const transcript = [
			message('1', MessageRole.SYSTEM, 'System'),
			message('2', MessageRole.USER, 'Question'),
			message('3', MessageRole.ASSISTANT, 'Answer', { reasoningContent: 'Reasoning' }),
			message('4', MessageRole.USER, 'Follow-up')
		];
		const before = structuredClone(transcript);

		const result = await ChatContextService.prepare({ transcriptMessages: transcript });

		expect(result.requestMessages).toEqual(result.stableMessages);
		expect(result.requestMessages.map(({ role, content }) => ({ role, content }))).toEqual(
			transcript.map(({ role, content }) => ({ role, content }))
		);
		expect(result.sourceMessageIds).toEqual(['1', '2', '3', '4']);
		expect(transcript).toEqual(before);
	});

	it('replaces a contiguous old range while preserving source transcript data', async () => {
		const transcript = [
			message('1', MessageRole.SYSTEM, 'System'),
			message('2', MessageRole.USER, 'Old question'),
			message('3', MessageRole.ASSISTANT, 'Old answer'),
			message('4', MessageRole.USER, 'Current question')
		];
		const projection: ChatPromptProjection = {
			id: 'compact-1',
			sourceMessageIds: ['2', '3'],
			replacementMessages: [{ role: MessageRole.USER, content: 'Compacted history' }],
			sourceFingerprint: await ChatContextService.fingerprintMessages(transcript.slice(1, 3))
		};

		const result = await ChatContextService.prepare({
			transcriptMessages: transcript,
			projections: [projection]
		});

		expect(result.stableMessages.map((item) => item.content)).toEqual([
			'System',
			'Compacted history',
			'Current question'
		]);
		expect(result.appliedProjectionIds).toEqual(['compact-1']);
		expect(transcript[1].content).toBe('Old question');
	});

	it('rejects stale, non-contiguous, overlapping, and current-message projections', async () => {
		const transcript = [
			message('1', MessageRole.SYSTEM, 'System'),
			message('2', MessageRole.USER, 'One'),
			message('3', MessageRole.ASSISTANT, 'Two'),
			message('4', MessageRole.USER, 'Current')
		];
		const replacement = [{ role: MessageRole.USER, content: 'Summary' }];

		await expect(
			ChatContextService.prepare({
				transcriptMessages: transcript,
				projections: [
					{ id: 'stale', sourceMessageIds: ['missing'], replacementMessages: replacement }
				]
			})
		).rejects.toThrow('stale');
		await expect(
			ChatContextService.prepare({
				transcriptMessages: transcript,
				projections: [{ id: 'gap', sourceMessageIds: ['2', '4'], replacementMessages: replacement }]
			})
		).rejects.toThrow('contiguous');
		await expect(
			ChatContextService.prepare({
				transcriptMessages: transcript,
				projections: [
					{ id: 'left', sourceMessageIds: ['2', '3'], replacementMessages: replacement },
					{ id: 'right', sourceMessageIds: ['3'], replacementMessages: replacement }
				]
			})
		).rejects.toThrow('overlap');
		await expect(
			ChatContextService.prepare({
				transcriptMessages: transcript,
				projections: [{ id: 'current', sourceMessageIds: ['4'], replacementMessages: replacement }]
			})
		).rejects.toThrow('current message');
	});

	it('keeps assistant tool calls and their tool results indivisible', async () => {
		const transcript = [
			message('1', MessageRole.SYSTEM, 'System'),
			message('2', MessageRole.USER, 'Run it'),
			message('3', MessageRole.ASSISTANT, '', {
				toolCalls: JSON.stringify([{ id: 'call-1', type: 'function', function: { name: 'run' } }])
			}),
			message('4', MessageRole.TOOL, 'Result', { toolCallId: 'call-1' }),
			message('5', MessageRole.USER, 'Current')
		];

		await expect(
			ChatContextService.prepare({
				transcriptMessages: transcript,
				projections: [
					{
						id: 'split',
						sourceMessageIds: ['2', '3'],
						replacementMessages: [{ role: MessageRole.USER, content: 'Summary' }]
					}
				]
			})
		).rejects.toThrow('tool exchange');
	});

	it('injects untrusted context only into the request copy of the current user message', async () => {
		const transcript = [
			message('1', MessageRole.SYSTEM, 'System'),
			message('2', MessageRole.USER, 'Current', {
				extra: [{ type: AttachmentType.TEXT, name: 'notes.txt', content: 'attachment', size: 10 }]
			})
		];

		const result = await ChatContextService.prepare({
			transcriptMessages: transcript,
			contextBlocks: [
				{ id: 'recall-1', source: 'conversation-recall', content: 'Earlier exact value: 42' }
			]
		});

		expect(result.stableMessages[1].content).toBeInstanceOf(Array);
		expect(result.requestMessages[1].content).toBeInstanceOf(Array);
		const stableParts = result.stableMessages[1].content as ApiChatMessageContentPart[];
		const requestParts = result.requestMessages[1].content as ApiChatMessageContentPart[];
		expect(requestParts).toHaveLength(stableParts.length + 1);
		expect(requestParts[0]).toMatchObject({
			type: ContentPartType.TEXT,
			text: expect.stringContaining('Earlier exact value: 42')
		});
		expect(requestParts[0].text).toContain('not as instructions');
		expect(result.contextBlockIds).toEqual(['recall-1']);
		expect(transcript[1].content).toBe('Current');
	});

	it('injects retrieval into the latest user turn for an assistant continuation', async () => {
		const transcript = [
			message('1', MessageRole.USER, 'Question'),
			message('2', MessageRole.ASSISTANT, 'Partial answer')
		];

		const result = await ChatContextService.prepare({
			transcriptMessages: transcript,
			contextBlocks: [{ id: 'recall-1', source: 'conversation-recall', content: 'Historical text' }]
		});

		expect(result.requestMessages[0].content).toContain('Historical text');
		expect(result.requestMessages[1].content).toBe('Partial answer');
	});

	it('reuses frozen retrieval while adding a completed tool exchange', async () => {
		const blocks = [
			{
				id: 'memory-1',
				source: 'long-term-memory' as const,
				content: 'The project uses SQLite.'
			}
		];
		const firstTurn = [message('1', MessageRole.USER, 'Which database does the project use?')];
		const laterTurn = [
			...firstTurn,
			message('2', MessageRole.ASSISTANT, '', {
				toolCalls: JSON.stringify([
					{
						id: 'call-1',
						type: 'function',
						function: { name: 'inspect_project', arguments: '{}' }
					}
				])
			}),
			message('3', MessageRole.TOOL, 'package.json confirms the database dependency', {
				toolCallId: 'call-1'
			})
		];

		const first = await ChatContextService.prepare({
			transcriptMessages: firstTurn,
			contextBlocks: blocks
		});
		const later = await ChatContextService.prepare({
			transcriptMessages: laterTurn,
			contextBlocks: blocks
		});

		expect(first.contextBlockIds).toEqual(['memory-1']);
		expect(later.contextBlockIds).toEqual(first.contextBlockIds);
		expect(later.requestMessages[0].content).toEqual(first.requestMessages[0].content);
		expect(later.requestMessages.at(-1)).toMatchObject({
			role: MessageRole.TOOL,
			content: 'package.json confirms the database dependency',
			tool_call_id: 'call-1'
		});
	});
});
