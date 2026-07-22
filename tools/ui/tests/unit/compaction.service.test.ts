import { afterEach, describe, expect, it, vi } from 'vitest';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import { COMPACTION_SECTIONS, CompactionService } from '$lib/services/compaction.service';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { ChatContextService } from '$lib/services/chat-context.service';
import type {
	DatabaseCompaction,
	DatabaseCompactionProjectionEvent,
	DatabaseMessage
} from '$lib/types';

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

function completeTurn(index: number): DatabaseMessage[] {
	return [
		message(`u${index}`, MessageRole.USER, `Question ${index}`),
		message(`a${index}`, MessageRole.ASSISTANT, `Answer ${index}`)
	];
}

function validSummary(): string {
	return COMPACTION_SECTIONS.map((section) => `${section}\n- Preserved value`).join('\n\n');
}

function record(source: DatabaseMessage[], id = 'compact-1'): DatabaseCompaction {
	return {
		id,
		conversationId: 'conversation-1',
		status: 'ready',
		sourceMessageIds: source.map((item) => item.id),
		deltaSourceMessageIds: source.map((item) => item.id),
		sourceFingerprint: '',
		summary: validSummary(),
		summarySchemaVersion: 1,
		promptVersion: 1,
		createdAt: 10,
		sourceTokenCount: 2000,
		beforeTokenCount: 4000,
		projectedTokenCount: 1500,
		generation: 1
	};
}

describe('CompactionService', () => {
	afterEach(() => vi.restoreAllMocks());

	it('groups complete turns while keeping a tool exchange indivisible', () => {
		const messages = [
			message('root', MessageRole.SYSTEM, '', { type: MessageType.ROOT }),
			message('u1', MessageRole.USER, 'Run it'),
			message('a1', MessageRole.ASSISTANT, '', {
				toolCalls: JSON.stringify([{ id: 'call-1', function: { name: 'run' } }])
			}),
			message('t1', MessageRole.TOOL, 'result', { toolCallId: 'call-1' }),
			message('a2', MessageRole.ASSISTANT, 'Done'),
			message('u2', MessageRole.USER, 'Current')
		];

		const units = CompactionService.groupTurns(messages);

		expect(units).toHaveLength(2);
		expect(units[0]).toMatchObject({
			messageIds: ['u1', 'a1', 't1', 'a2'],
			complete: true
		});
		expect(units[1]).toMatchObject({ messageIds: ['u2'], complete: false });
	});

	it('protects the eight most recent complete turns and offers safe ending boundaries', () => {
		const messages = Array.from({ length: 11 }, (_, index) => completeTurn(index + 1)).flat();
		messages.push(message('current', MessageRole.USER, 'Current'));

		const candidates = CompactionService.buildRangeCandidates(messages);

		expect(candidates).toHaveLength(3);
		expect(candidates[0].sourceMessageIds).toEqual(['u1', 'a1']);
		expect(candidates[2].sourceMessageIds).toEqual(['u1', 'a1', 'u2', 'a2', 'u3', 'a3']);
	});

	it('extends an existing source range with only new delta messages', () => {
		const messages = Array.from({ length: 11 }, (_, index) => completeTurn(index + 1)).flat();
		const previous = record(messages.slice(0, 2));

		const candidates = CompactionService.buildRangeCandidates(messages, previous);
		const last = candidates.at(-1)!;

		expect(last.sourceMessageIds.slice(0, 2)).toEqual(previous.sourceMessageIds);
		expect(CompactionService.getDeltaSourceIds(last.sourceMessageIds, previous)).not.toContain(
			'u1'
		);
	});

	it('serializes exact tool and attachment metadata as historical data', () => {
		const serialized = CompactionService.serializeMessages([
			message('a1', MessageRole.ASSISTANT, '', { toolCalls: '[{"id":"call-1"}]' }),
			message('t1', MessageRole.TOOL, '42', {
				toolCallId: 'call-1',
				extra: [{ type: AttachmentType.TEXT, name: 'notes.txt', content: 'exact path /tmp/a' }]
			})
		]);

		expect(serialized).toContain('call-1');
		expect(serialized).toContain('notes.txt');
		expect(serialized).toContain('exact path /tmp/a');
	});

	it('requires every structured section once and in order', () => {
		expect(CompactionService.validateSummary(validSummary())).toEqual([]);
		expect(
			CompactionService.validateSummary(validSummary().replace('UNCERTAINTIES', ''))
		).toContain('Expected exactly one UNCERTAINTIES section');
		expect(CompactionService.validateSummary(`\`\`\`\n${validSummary()}\n\`\`\``)).toContain(
			'The compacted state must not use code fences'
		);
	});

	it('builds a broadly compatible synthetic user and assistant replacement pair', () => {
		const replacement = CompactionService.replacementMessages(record([]));

		expect(replacement.map((item) => item.role)).toEqual([MessageRole.USER, MessageRole.ASSISTANT]);
		expect(replacement[1].content).toContain('BEGIN COMPACTED STATE');
	});

	it('uses the current model for a bounded non-streaming summary request', async () => {
		const send = vi.spyOn(ChatService, 'sendMessage').mockResolvedValue(validSummary());

		const result = await CompactionService.generateSummary({
			messages: completeTurn(1),
			sourceMessageIds: ['u1', 'a1'],
			deltaSourceMessageIds: ['u1', 'a1'],
			model: 'current-model',
			maxTokens: 768
		});

		expect(result.validationErrors).toEqual([]);
		expect(send).toHaveBeenCalledOnce();
		expect(send.mock.calls[0][1]).toMatchObject({
			model: 'current-model',
			stream: false,
			enableThinking: false,
			temperature: 0.2,
			max_tokens: 768
		});
	});

	it('resolves the deepest projection event on the active path', async () => {
		const path = [
			message('system', MessageRole.SYSTEM, 'System'),
			...completeTurn(1),
			...completeTurn(2),
			message('leaf', MessageRole.USER, 'Current')
		];
		const first = record(path.slice(1, 3), 'compact-1');
		first.sourceFingerprint = await ChatContextService.fingerprintMessages(path.slice(1, 3));
		const second = record(path.slice(1, 5), 'compact-2');
		second.sourceFingerprint = await ChatContextService.fingerprintMessages(path.slice(1, 5));
		const events: DatabaseCompactionProjectionEvent[] = [
			{
				id: 'event-1',
				conversationId: 'conversation-1',
				anchorMessageId: 'a1',
				action: 'apply',
				compactionId: first.id,
				createdAt: 10
			},
			{
				id: 'event-2',
				conversationId: 'conversation-1',
				anchorMessageId: 'a2',
				action: 'apply',
				compactionId: second.id,
				createdAt: 20
			}
		];
		vi.spyOn(DatabaseService, 'getConversationCompactions').mockResolvedValue([first, second]);
		vi.spyOn(DatabaseService, 'getConversationCompactionProjectionEvents').mockResolvedValue(
			events
		);

		const resolved = await CompactionService.resolveActiveCompaction('conversation-1', path);

		expect(resolved?.record.id).toBe('compact-2');
	});

	it('lets a later restore event disable inherited compaction on the current path', async () => {
		const path = [...completeTurn(1), message('leaf', MessageRole.USER, 'Current')];
		const active = record(path.slice(0, 2));
		vi.spyOn(DatabaseService, 'getConversationCompactions').mockResolvedValue([active]);
		vi.spyOn(DatabaseService, 'getConversationCompactionProjectionEvents').mockResolvedValue([
			{
				id: 'apply',
				conversationId: 'conversation-1',
				anchorMessageId: 'a1',
				action: 'apply',
				compactionId: active.id,
				createdAt: 1
			},
			{
				id: 'restore',
				conversationId: 'conversation-1',
				anchorMessageId: 'leaf',
				action: 'restore',
				createdAt: 2
			}
		]);

		await expect(
			CompactionService.resolveActiveCompaction('conversation-1', path)
		).resolves.toBeNull();
	});
});
