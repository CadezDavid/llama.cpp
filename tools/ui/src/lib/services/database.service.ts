import Dexie, { type EntityTable } from 'dexie';
import { findDescendantMessages, uuid, filterByLeafNodeId } from '$lib/utils';
import {
	IDXDB_TABLES,
	IDXDB_STORES,
	IDXDB_STORES_V1,
	IDXDB_STORES_V3,
	STORAGE_APP_NAME
} from '$lib/constants';
import { MessageRole } from '$lib/enums';
import type { McpServerOverride } from '$lib/types/database';
import type { ExportedConversation } from '$lib/types/database';
import type { DatabaseCompaction, DatabaseCompactionProjectionEvent } from '$lib/types';
import type {
	DatabaseArchiveChunk,
	DatabaseArchiveTerm,
	DatabaseRetrievalTrace,
	DatabaseRetrievalHitUsage
} from '$lib/types';
import { ChatContextService } from './chat-context.service';
import { memoryDebug } from '$lib/utils/memory-debug';

function createArchiveChunks(
	record: DatabaseCompaction,
	messages: DatabaseMessage[]
): DatabaseArchiveChunk[] {
	const archivedIds = new Set(record.deltaSourceMessageIds);
	return messages
		.filter((message) => archivedIds.has(message.id) && message.content.trim())
		.flatMap((message) => {
			const words = message.content.trim().split(/\s+/);
			const pieces: string[] = [];
			for (let offset = 0; offset < words.length; offset += 500) {
				pieces.push(words.slice(offset, offset + 500).join(' '));
			}
			return pieces.map((piece) => ({
				id: uuid(),
				conversationId: record.conversationId,
				compactionId: record.id,
				generation: record.generation,
				sourceMessageIds: [message.id],
				text: `${message.role}: ${piece}`,
				terms: Array.from(new Set(piece.toLowerCase().match(/[\p{L}\p{N}_-]+/gu) ?? [])),
				createdAt: Date.now(),
				embeddingStatus: 'pending' as const
			}));
		});
}

class LlamaUiDatabase extends Dexie {
	[IDXDB_TABLES.conversations]!: EntityTable<DatabaseConversation, string>;
	[IDXDB_TABLES.messages]!: EntityTable<DatabaseMessage, string>;
	[IDXDB_TABLES.compactions]!: EntityTable<DatabaseCompaction, 'id'>;
	[IDXDB_TABLES.compactionProjectionEvents]!: EntityTable<DatabaseCompactionProjectionEvent, 'id'>;
	[IDXDB_TABLES.archiveChunks]!: EntityTable<DatabaseArchiveChunk, 'id'>;
	[IDXDB_TABLES.archiveTerms]!: EntityTable<DatabaseArchiveTerm, 'id'>;
	[IDXDB_TABLES.retrievalTraces]!: EntityTable<DatabaseRetrievalTrace, 'id'>;
	[IDXDB_TABLES.retrievalHitUsage]!: EntityTable<DatabaseRetrievalHitUsage, 'id'>;

	constructor() {
		super(STORAGE_APP_NAME);

		this.version(1).stores(IDXDB_STORES_V1);
		this.version(2).stores(IDXDB_STORES);
		this.version(3).stores(IDXDB_STORES_V3);
	}
}

const db = new LlamaUiDatabase();

export class DatabaseService {
	/**
	 *
	 *
	 * Conversations
	 *
	 *
	 */

	/**
	 * Creates a new conversation.
	 *
	 * @param name - Name of the conversation
	 * @returns The created conversation
	 */
	static async createConversation(name: string): Promise<DatabaseConversation> {
		const conversation: DatabaseConversation = {
			id: uuid(),
			name,
			lastModified: Date.now(),
			currNode: ''
		};

		await db[IDXDB_TABLES.conversations].add(conversation);
		return conversation;
	}

	/**
	 *
	 *
	 * Messages
	 *
	 *
	 */

	/**
	 * Creates a new message branch by adding a message and updating parent/child relationships.
	 * Also updates the conversation's currNode to point to the new message.
	 *
	 * @param message - Message to add (without id)
	 * @param parentId - Parent message ID to attach to
	 * @returns The created message
	 */
	static async createMessageBranch(
		message: Omit<DatabaseMessage, 'id'>,
		parentId: string | null
	): Promise<DatabaseMessage> {
		return await db.transaction(
			'rw',
			[db[IDXDB_TABLES.conversations], db[IDXDB_TABLES.messages]],
			async () => {
				// Handle null parent (root message case)
				if (parentId !== null) {
					const parentMessage = await db[IDXDB_TABLES.messages].get(parentId);
					if (!parentMessage) {
						throw new Error(`Parent message ${parentId} not found`);
					}
				}

				const newMessage: DatabaseMessage = {
					...message,
					id: uuid(),
					parent: parentId,
					toolCalls: message.toolCalls ?? '',
					children: []
				};

				await db[IDXDB_TABLES.messages].add(newMessage);

				// Update parent's children array if parent exists
				if (parentId !== null) {
					const parentMessage = await db[IDXDB_TABLES.messages].get(parentId);
					if (parentMessage) {
						await db[IDXDB_TABLES.messages].update(parentId, {
							children: [...parentMessage.children, newMessage.id]
						});
					}
				}

				await this.updateConversation(message.convId, {
					currNode: newMessage.id
				});

				return newMessage;
			}
		);
	}

	/**
	 * Creates a root message for a new conversation.
	 * Root messages are not displayed but serve as the tree root for branching.
	 *
	 * @param convId - Conversation ID
	 * @returns The created root message
	 */
	static async createRootMessage(convId: string): Promise<string> {
		const rootMessage: DatabaseMessage = {
			id: uuid(),
			convId,
			type: 'root',
			timestamp: Date.now(),
			role: MessageRole.SYSTEM,
			content: '',
			parent: null,
			toolCalls: '',
			children: []
		};

		await db[IDXDB_TABLES.messages].add(rootMessage);
		return rootMessage.id;
	}

	/**
	 * Creates a system prompt message for a conversation.
	 *
	 * @param convId - Conversation ID
	 * @param systemPrompt - The system prompt content (must be non-empty)
	 * @param parentId - Parent message ID (typically the root message)
	 * @returns The created system message
	 * @throws Error if systemPrompt is empty
	 */
	static async createSystemMessage(
		convId: string,
		systemPrompt: string,
		parentId: string
	): Promise<DatabaseMessage> {
		const trimmedPrompt = systemPrompt.trim();
		if (!trimmedPrompt) {
			throw new Error('Cannot create system message with empty content');
		}

		const systemMessage: DatabaseMessage = {
			id: uuid(),
			convId,
			type: MessageRole.SYSTEM,
			timestamp: Date.now(),
			role: MessageRole.SYSTEM,
			content: trimmedPrompt,
			parent: parentId,
			children: []
		};

		await db[IDXDB_TABLES.messages].add(systemMessage);

		const parentMessage = await db[IDXDB_TABLES.messages].get(parentId);
		if (parentMessage) {
			await db[IDXDB_TABLES.messages].update(parentId, {
				children: [...parentMessage.children, systemMessage.id]
			});
		}

		return systemMessage;
	}

	/**
	 * Deletes a conversation and all its messages.
	 *
	 * @param id - Conversation ID
	 */
	static async deleteConversation(
		id: string,
		options?: { deleteWithForks?: boolean }
	): Promise<void> {
		await db.transaction(
			'rw',
			[
				db[IDXDB_TABLES.conversations],
				db[IDXDB_TABLES.messages],
				db[IDXDB_TABLES.compactions],
				db[IDXDB_TABLES.compactionProjectionEvents],
				db[IDXDB_TABLES.archiveChunks],
				db[IDXDB_TABLES.archiveTerms],
				db[IDXDB_TABLES.retrievalTraces],
				db[IDXDB_TABLES.retrievalHitUsage]
			],
			async () => {
				if (options?.deleteWithForks) {
					// Recursively collect all descendant IDs
					const idsToDelete: string[] = [];
					const queue = [id];

					while (queue.length > 0) {
						const parentId = queue.pop()!;
						const children = await db[IDXDB_TABLES.conversations]
							.filter((c) => c.forkedFromConversationId === parentId)
							.toArray();

						for (const child of children) {
							idsToDelete.push(child.id);
							queue.push(child.id);
						}
					}

					for (const forkId of idsToDelete) {
						await db[IDXDB_TABLES.conversations].delete(forkId);
						await db[IDXDB_TABLES.messages].where('convId').equals(forkId).delete();
						await db[IDXDB_TABLES.compactions].where('conversationId').equals(forkId).delete();
						await db[IDXDB_TABLES.compactionProjectionEvents]
							.where('conversationId')
							.equals(forkId)
							.delete();
						await db[IDXDB_TABLES.archiveChunks].where('conversationId').equals(forkId).delete();
						await db[IDXDB_TABLES.archiveTerms].where('conversationId').equals(forkId).delete();
						await db[IDXDB_TABLES.retrievalTraces].where('conversationId').equals(forkId).delete();
						await db[IDXDB_TABLES.retrievalHitUsage]
							.where('conversationId')
							.equals(forkId)
							.delete();
					}
				} else {
					await this.reparentDirectChildren(id);
				}

				await db[IDXDB_TABLES.conversations].delete(id);
				await db[IDXDB_TABLES.messages].where('convId').equals(id).delete();
				await db[IDXDB_TABLES.compactions].where('conversationId').equals(id).delete();
				await db[IDXDB_TABLES.compactionProjectionEvents]
					.where('conversationId')
					.equals(id)
					.delete();
				await db[IDXDB_TABLES.archiveChunks].where('conversationId').equals(id).delete();
				await db[IDXDB_TABLES.archiveTerms].where('conversationId').equals(id).delete();
				await db[IDXDB_TABLES.retrievalTraces].where('conversationId').equals(id).delete();
				await db[IDXDB_TABLES.retrievalHitUsage].where('conversationId').equals(id).delete();
			}
		);
	}

	/**
	 * Reparents direct children of `parentId` to the nearest surviving
	 * ancestor (or promotes them to top-level when the immediate parent was
	 * top-level). Walking skips any ancestor listed in `excludeIds`, since
	 * those will be deleted in the same batch — leaving a grandchild pointing
	 * at an `excludeIds` entry would orphan it. Children whose own id is in
	 * `excludeIds` are dropped from the updates (the bulk-delete pass will
	 * remove them). `prefetched` may carry a pre-fetched ancestor map to
	 * avoid repeat reads inside a bulk transaction.
	 */
	private static async reparentDirectChildren(
		parentId: string,
		excludeIds: ReadonlySet<string> = new Set(),
		prefetched?: ReadonlyMap<string, DatabaseConversation>
	): Promise<void> {
		const conv = prefetched?.get(parentId) ?? (await db[IDXDB_TABLES.conversations].get(parentId));
		if (!conv) return;

		let newParent = conv.forkedFromConversationId;
		const visited = new Set<string>([parentId]);
		while (newParent && excludeIds.has(newParent)) {
			if (visited.has(newParent)) {
				newParent = undefined;
				break;
			}
			visited.add(newParent);
			const next =
				prefetched?.get(newParent) ?? (await db[IDXDB_TABLES.conversations].get(newParent));
			if (!next) {
				newParent = undefined;
				break;
			}
			newParent = next.forkedFromConversationId;
		}

		const directChildren = await db[IDXDB_TABLES.conversations]
			.filter((c) => c.forkedFromConversationId === parentId)
			.toArray();

		const updates: DatabaseConversation[] = [];
		for (const child of directChildren) {
			if (excludeIds.has(child.id)) continue;
			updates.push({ ...child, forkedFromConversationId: newParent });
		}
		if (updates.length === 0) return;
		await db[IDXDB_TABLES.conversations].bulkPut(updates);
	}

	/**
	 * Deletes multiple conversations in a single transaction. Each deleted
	 * conversation has its direct children reparented to the nearest surviving
	 * ancestor (or promoted to top-level). Children also in `ids` are dropped
	 * entirely rather than reparented.
	 *
	 * @param ids - Conversation IDs to delete
	 */
	static async bulkDeleteConversations(ids: string[]): Promise<void> {
		const cleanIds = ids.filter((id): id is string => typeof id === 'string' && id.length > 0);
		if (cleanIds.length === 0) return;
		const idSet = new Set(cleanIds);

		await db.transaction(
			'rw',
			[
				db[IDXDB_TABLES.conversations],
				db[IDXDB_TABLES.messages],
				db[IDXDB_TABLES.compactions],
				db[IDXDB_TABLES.compactionProjectionEvents],
				db[IDXDB_TABLES.archiveChunks],
				db[IDXDB_TABLES.archiveTerms],
				db[IDXDB_TABLES.retrievalTraces],
				db[IDXDB_TABLES.retrievalHitUsage]
			],
			async () => {
				// Pre-load each to-delete conversation so the per-id reparent
				// walk-up doesn't ping-pong the same ancestry chain.
				const prefetched = new Map<string, DatabaseConversation>();
				let frontier = [...cleanIds];
				const requested = new Set<string>(frontier);
				while (frontier.length > 0) {
					const fetched = await db[IDXDB_TABLES.conversations].bulkGet(frontier);
					frontier = [];
					for (let i = 0; i < fetched.length; i++) {
						const conv = fetched[i];
						if (!conv || !conv.id) continue;
						prefetched.set(conv.id, conv);
						const ancestor = conv.forkedFromConversationId;
						if (ancestor && !prefetched.has(ancestor) && !requested.has(ancestor)) {
							frontier.push(ancestor);
							requested.add(ancestor);
						}
					}
				}

				for (const id of cleanIds) {
					await this.reparentDirectChildren(id, idSet, prefetched);
				}

				await db[IDXDB_TABLES.conversations].bulkDelete(cleanIds);
				await db[IDXDB_TABLES.messages].where('convId').anyOf(cleanIds).delete();
				await db[IDXDB_TABLES.compactions].where('conversationId').anyOf(cleanIds).delete();
				await db[IDXDB_TABLES.compactionProjectionEvents]
					.where('conversationId')
					.anyOf(cleanIds)
					.delete();
				await db[IDXDB_TABLES.archiveChunks].where('conversationId').anyOf(cleanIds).delete();
				await db[IDXDB_TABLES.archiveTerms].where('conversationId').anyOf(cleanIds).delete();
				await db[IDXDB_TABLES.retrievalTraces].where('conversationId').anyOf(cleanIds).delete();
				await db[IDXDB_TABLES.retrievalHitUsage].where('conversationId').anyOf(cleanIds).delete();
			}
		);
	}

	/**
	 * Deletes a message and removes it from its parent's children array.
	 *
	 * @param messageId - ID of the message to delete
	 */
	static async deleteMessage(messageId: string): Promise<void> {
		await db.transaction(
			'rw',
			[
				db[IDXDB_TABLES.messages],
				db[IDXDB_TABLES.compactions],
				db[IDXDB_TABLES.compactionProjectionEvents],
				db[IDXDB_TABLES.archiveChunks],
				db[IDXDB_TABLES.archiveTerms],
				db[IDXDB_TABLES.retrievalTraces]
			],
			async () => {
				const message = await db[IDXDB_TABLES.messages].get(messageId);
				if (!message) return;

				// Remove this message from its parent's children array
				if (message.parent) {
					const parent = await db[IDXDB_TABLES.messages].get(message.parent);
					if (parent) {
						parent.children = parent.children.filter((childId: string) => childId !== messageId);
						await db[IDXDB_TABLES.messages].put(parent);
					}
				}

				const invalidCompactions = await db[IDXDB_TABLES.compactions]
					.where('conversationId')
					.equals(message.convId)
					.filter((record) => record.sourceMessageIds.includes(messageId))
					.toArray();
				const invalidIds = invalidCompactions.map((record) => record.id);
				await db[IDXDB_TABLES.messages].delete(messageId);
				await db[IDXDB_TABLES.compactionProjectionEvents]
					.where('anchorMessageId')
					.equals(messageId)
					.delete();
				const deletedTraceCount = await db[IDXDB_TABLES.retrievalTraces]
					.where('conversationId')
					.equals(message.convId)
					.filter(
						(trace) => trace.anchorMessageId === messageId || trace.responseMessageId === messageId
					)
					.delete();
				memoryDebug('retrieval.trace.delete-message', {
					conversationId: message.convId,
					messageId,
					deletedTraceCount
				});
				if (invalidIds.length > 0) {
					const invalidChunks = await db[IDXDB_TABLES.archiveChunks]
						.where('compactionId')
						.anyOf(invalidIds)
						.primaryKeys();
					if (invalidChunks.length) {
						await db[IDXDB_TABLES.archiveTerms].where('chunkId').anyOf(invalidChunks).delete();
						await db[IDXDB_TABLES.archiveChunks].bulkDelete(invalidChunks);
					}
					await db[IDXDB_TABLES.compactionProjectionEvents]
						.where('conversationId')
						.equals(message.convId)
						.filter((event) => !!event.compactionId && invalidIds.includes(event.compactionId))
						.delete();
					await db[IDXDB_TABLES.compactions].bulkDelete(invalidIds);
				}
			}
		);
	}

	/**
	 * Deletes a message and all its descendant messages (cascading deletion).
	 * This removes the entire branch starting from the specified message.
	 *
	 * @param conversationId - ID of the conversation containing the message
	 * @param messageId - ID of the root message to delete (along with all descendants)
	 * @returns Array of all deleted message IDs
	 */
	static async deleteMessageCascading(
		conversationId: string,
		messageId: string
	): Promise<string[]> {
		return await db.transaction(
			'rw',
			[
				db[IDXDB_TABLES.messages],
				db[IDXDB_TABLES.compactions],
				db[IDXDB_TABLES.compactionProjectionEvents],
				db[IDXDB_TABLES.archiveChunks],
				db[IDXDB_TABLES.archiveTerms],
				db[IDXDB_TABLES.retrievalTraces]
			],
			async () => {
				// Get all messages in the conversation to find descendants
				const allMessages = await db[IDXDB_TABLES.messages]
					.where('convId')
					.equals(conversationId)
					.toArray();

				// Find all descendant messages
				const descendants = findDescendantMessages(allMessages, messageId);
				const allToDelete = [messageId, ...descendants];

				// Get the message to delete for parent cleanup
				const message = await db[IDXDB_TABLES.messages].get(messageId);
				if (message && message.parent) {
					const parent = await db[IDXDB_TABLES.messages].get(message.parent);
					if (parent) {
						parent.children = parent.children.filter((childId: string) => childId !== messageId);
						await db[IDXDB_TABLES.messages].put(parent);
					}
				}

				// Delete all messages in the branch
				const invalidCompactions = (
					await db[IDXDB_TABLES.compactions]
						.where('conversationId')
						.equals(conversationId)
						.toArray()
				).filter((record) =>
					record.sourceMessageIds.some((sourceId) => allToDelete.includes(sourceId))
				);
				const invalidIds = invalidCompactions.map((record) => record.id);
				await db[IDXDB_TABLES.messages].bulkDelete(allToDelete);
				await db[IDXDB_TABLES.compactionProjectionEvents]
					.where('anchorMessageId')
					.anyOf(allToDelete)
					.delete();
				const deletedIds = new Set(allToDelete);
				const deletedTraceCount = await db[IDXDB_TABLES.retrievalTraces]
					.where('conversationId')
					.equals(conversationId)
					.filter(
						(trace) =>
							deletedIds.has(trace.anchorMessageId) ||
							(!!trace.responseMessageId && deletedIds.has(trace.responseMessageId))
					)
					.delete();
				memoryDebug('retrieval.trace.delete-branch', {
					conversationId,
					deletedMessageCount: allToDelete.length,
					deletedTraceCount
				});
				if (invalidIds.length > 0) {
					const invalidChunks = await db[IDXDB_TABLES.archiveChunks]
						.where('compactionId')
						.anyOf(invalidIds)
						.primaryKeys();
					if (invalidChunks.length) {
						await db[IDXDB_TABLES.archiveTerms].where('chunkId').anyOf(invalidChunks).delete();
						await db[IDXDB_TABLES.archiveChunks].bulkDelete(invalidChunks);
					}
					await db[IDXDB_TABLES.compactionProjectionEvents]
						.where('conversationId')
						.equals(conversationId)
						.filter((event) => !!event.compactionId && invalidIds.includes(event.compactionId))
						.delete();
					await db[IDXDB_TABLES.compactions].bulkDelete(invalidIds);
				}

				return allToDelete;
			}
		);
	}

	/**
	 * Gets all conversations, sorted by last modified time (newest first).
	 *
	 * @returns Array of conversations
	 */
	static async getAllConversations(): Promise<DatabaseConversation[]> {
		return await db[IDXDB_TABLES.conversations].orderBy('lastModified').reverse().toArray();
	}

	/**
	 * Gets a conversation by ID.
	 *
	 * @param id - Conversation ID
	 * @returns The conversation if found, otherwise undefined
	 */
	static async getConversation(id: string): Promise<DatabaseConversation | undefined> {
		return await db[IDXDB_TABLES.conversations].get(id);
	}

	/**
	 * Gets all messages in a conversation, sorted by timestamp (oldest first).
	 *
	 * @param convId - Conversation ID
	 * @returns Array of messages in the conversation
	 */
	static async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await db[IDXDB_TABLES.messages].where('convId').equals(convId).sortBy('timestamp');
	}

	/**
	 * Compaction records are immutable once ready. A pending record is harmless
	 * until an apply projection event is committed with it.
	 */
	static async createPendingCompaction(
		record: Omit<DatabaseCompaction, 'id' | 'status' | 'createdAt'>
	): Promise<DatabaseCompaction> {
		const pending: DatabaseCompaction = {
			...record,
			sourceMessageIds: [...record.sourceMessageIds],
			deltaSourceMessageIds: [...record.deltaSourceMessageIds],
			id: uuid(),
			status: 'pending',
			createdAt: Date.now()
		};
		await db[IDXDB_TABLES.compactions].add(pending);
		return pending;
	}

	static async deletePendingCompaction(id: string): Promise<void> {
		const record = await db[IDXDB_TABLES.compactions].get(id);
		if (record?.status === 'pending') await db[IDXDB_TABLES.compactions].delete(id);
	}

	static async activateCompaction(
		id: string,
		updates: Pick<DatabaseCompaction, 'summary' | 'projectedTokenCount'> &
			Partial<
				Pick<
					DatabaseCompaction,
					| 'activationMode'
					| 'contextSize'
					| 'usableInputTokenCount'
					| 'triggerPercent'
					| 'targetPercent'
					| 'attemptCount'
					| 'durationMs'
					| 'strictRetryUsed'
				>
			>,
		anchorMessageId: string
	): Promise<DatabaseCompactionProjectionEvent> {
		return await db.transaction(
			'rw',
			[
				db[IDXDB_TABLES.conversations],
				db[IDXDB_TABLES.messages],
				db[IDXDB_TABLES.compactions],
				db[IDXDB_TABLES.compactionProjectionEvents],
				db[IDXDB_TABLES.archiveChunks],
				db[IDXDB_TABLES.archiveTerms]
			],
			async () => {
				const record = await db[IDXDB_TABLES.compactions].get(id);
				if (!record || record.status !== 'pending') {
					throw new Error(`Pending compaction ${id} not found`);
				}
				const anchor = await db[IDXDB_TABLES.messages].get(anchorMessageId);
				if (!anchor || anchor.convId !== record.conversationId) {
					throw new Error(`Compaction anchor ${anchorMessageId} is not in the conversation`);
				}
				const conversation = await db[IDXDB_TABLES.conversations].get(record.conversationId);
				if (!conversation?.currNode) throw new Error('The conversation has no active branch');
				const conversationMessages = await db[IDXDB_TABLES.messages]
					.where('convId')
					.equals(record.conversationId)
					.toArray();
				const activePath = filterByLeafNodeId(
					conversationMessages,
					conversation.currNode,
					true
				) as DatabaseMessage[];
				const pathIndexes = new Map(activePath.map((message, index) => [message.id, index]));
				if (!pathIndexes.has(anchorMessageId)) {
					throw new Error(`Compaction anchor ${anchorMessageId} is no longer active`);
				}
				const sourceIndexes = record.sourceMessageIds.map((sourceId) => pathIndexes.get(sourceId));
				const sourceStart = sourceIndexes[0];
				if (
					sourceStart === undefined ||
					sourceIndexes.some((sourceIndex, offset) => sourceIndex !== sourceStart + offset)
				) {
					throw new Error(`Compaction ${id} source range is no longer active`);
				}
				const sourceMessages = activePath.slice(sourceStart, sourceStart + sourceIndexes.length);
				const sourceFingerprint = await Dexie.waitFor(
					ChatContextService.fingerprintMessages(sourceMessages)
				);
				if (sourceFingerprint !== record.sourceFingerprint) {
					throw new Error(`Compaction ${id} source messages changed before activation`);
				}
				await db[IDXDB_TABLES.compactions].put({ ...record, ...updates, status: 'ready' });
				const event: DatabaseCompactionProjectionEvent = {
					id: uuid(),
					conversationId: record.conversationId,
					anchorMessageId,
					action: 'apply',
					compactionId: id,
					createdAt: Date.now()
				};
				await db[IDXDB_TABLES.compactionProjectionEvents].add(event);
				const archiveChunks = createArchiveChunks(record, sourceMessages);
				const archiveTerms: DatabaseArchiveTerm[] = archiveChunks.flatMap((chunk) =>
					chunk.terms.map((term) => ({
						id: `${chunk.id}:${term}`,
						conversationId: chunk.conversationId,
						chunkId: chunk.id,
						term
					}))
				);
				if (archiveChunks.length) await db[IDXDB_TABLES.archiveChunks].bulkAdd(archiveChunks);
				if (archiveTerms.length) await db[IDXDB_TABLES.archiveTerms].bulkAdd(archiveTerms);
				return event;
			}
		);
	}

	static async createCompactionRestoreEvent(
		conversationId: string,
		anchorMessageId: string
	): Promise<DatabaseCompactionProjectionEvent> {
		return await db.transaction(
			'rw',
			[db[IDXDB_TABLES.messages], db[IDXDB_TABLES.compactionProjectionEvents]],
			async () => {
				const anchor = await db[IDXDB_TABLES.messages].get(anchorMessageId);
				if (!anchor || anchor.convId !== conversationId) {
					throw new Error(`Compaction anchor ${anchorMessageId} is not in the conversation`);
				}
				const event: DatabaseCompactionProjectionEvent = {
					id: uuid(),
					conversationId,
					anchorMessageId,
					action: 'restore',
					createdAt: Date.now()
				};
				await db[IDXDB_TABLES.compactionProjectionEvents].add(event);
				return event;
			}
		);
	}

	static async getConversationCompactions(conversationId: string): Promise<DatabaseCompaction[]> {
		return await db[IDXDB_TABLES.compactions]
			.where('conversationId')
			.equals(conversationId)
			.sortBy('createdAt');
	}

	static async getConversationCompactionProjectionEvents(
		conversationId: string
	): Promise<DatabaseCompactionProjectionEvent[]> {
		return await db[IDXDB_TABLES.compactionProjectionEvents]
			.where('conversationId')
			.equals(conversationId)
			.sortBy('createdAt');
	}

	static async cleanupAbandonedCompactions(maxAgeMs = 24 * 60 * 60 * 1000): Promise<void> {
		const cutoff = Date.now() - maxAgeMs;
		const abandoned = await db[IDXDB_TABLES.compactions]
			.where('status')
			.equals('pending')
			.filter((record) => record.createdAt < cutoff)
			.primaryKeys();
		if (abandoned.length > 0) await db[IDXDB_TABLES.compactions].bulkDelete(abandoned);
	}

	static async getConversationArchiveChunks(
		conversationId: string
	): Promise<DatabaseArchiveChunk[]> {
		return await db[IDXDB_TABLES.archiveChunks]
			.where('conversationId')
			.equals(conversationId)
			.toArray();
	}

	static async getPendingArchiveChunks(
		conversationId: string,
		limit = 16
	): Promise<DatabaseArchiveChunk[]> {
		return await db[IDXDB_TABLES.archiveChunks]
			.where('conversationId')
			.equals(conversationId)
			.filter((chunk) => chunk.embeddingStatus === 'pending')
			.limit(limit)
			.toArray();
	}

	static async updateArchiveChunk(
		id: string,
		changes: Partial<DatabaseArchiveChunk>
	): Promise<void> {
		await db[IDXDB_TABLES.archiveChunks].update(id, changes);
	}

	static async addRetrievalTrace(trace: DatabaseRetrievalTrace): Promise<void> {
		await db[IDXDB_TABLES.retrievalTraces].add(trace);
		memoryDebug('retrieval.trace.persisted', {
			traceId: trace.id,
			conversationId: trace.conversationId,
			anchorMessageId: trace.anchorMessageId,
			responseMessageId: trace.responseMessageId,
			hitCount: trace.hits.length,
			injectedCount: trace.injectedHitIds.length,
			injectedTokenCount: trace.injectedTokenCount
		});
	}

	static async getRetrievalTraces(
		conversationId: string,
		limit?: number
	): Promise<DatabaseRetrievalTrace[]> {
		const traces = await db[IDXDB_TABLES.retrievalTraces]
			.where('conversationId')
			.equals(conversationId)
			.toArray();
		traces.sort((left, right) => right.createdAt - left.createdAt);
		return limit === undefined ? traces : traces.slice(0, limit);
	}

	static async getRetrievalHitUsage(conversationId: string): Promise<DatabaseRetrievalHitUsage[]> {
		return await db[IDXDB_TABLES.retrievalHitUsage]
			.where('conversationId')
			.equals(conversationId)
			.toArray();
	}

	static async putRetrievalHitUsage(records: DatabaseRetrievalHitUsage[]): Promise<void> {
		if (records.length) await db[IDXDB_TABLES.retrievalHitUsage].bulkPut(records);
	}

	/**
	 * Loads multiple conversations with all of their messages in two bulk
	 * reads. Missing conversations are silently omitted from the result.
	 *
	 * @param convIds - Conversation IDs to load
	 * @returns Map of id -> { conv, messages }. Messages are sorted ascending by timestamp.
	 */
	static async getConversationsWithMessages(
		convIds: string[]
	): Promise<Map<string, ExportedConversation>> {
		const result = new Map<string, ExportedConversation>();
		const cleanIds = convIds.filter((id): id is string => typeof id === 'string' && id.length > 0);
		if (cleanIds.length === 0) return result;

		const [
			convs,
			allMessages,
			allCompactions,
			allProjectionEvents,
			allArchiveChunks,
			allRetrievalTraces,
			allRetrievalUsage
		] = await Promise.all([
			db[IDXDB_TABLES.conversations].bulkGet(cleanIds),
			db[IDXDB_TABLES.messages].where('convId').anyOf(cleanIds).toArray(),
			db[IDXDB_TABLES.compactions].where('conversationId').anyOf(cleanIds).toArray(),
			db[IDXDB_TABLES.compactionProjectionEvents].where('conversationId').anyOf(cleanIds).toArray(),
			db[IDXDB_TABLES.archiveChunks].where('conversationId').anyOf(cleanIds).toArray(),
			db[IDXDB_TABLES.retrievalTraces].where('conversationId').anyOf(cleanIds).toArray(),
			db[IDXDB_TABLES.retrievalHitUsage].where('conversationId').anyOf(cleanIds).toArray()
		]);

		const messagesByConv = new Map<string, DatabaseMessage[]>();
		for (const msg of allMessages) {
			const bucket = messagesByConv.get(msg.convId);
			if (bucket) bucket.push(msg);
			else messagesByConv.set(msg.convId, [msg]);
		}
		const compactionsByConv = new Map<string, DatabaseCompaction[]>();
		for (const record of allCompactions) {
			const bucket = compactionsByConv.get(record.conversationId);
			if (bucket) bucket.push(record);
			else compactionsByConv.set(record.conversationId, [record]);
		}
		const eventsByConv = new Map<string, DatabaseCompactionProjectionEvent[]>();
		for (const event of allProjectionEvents) {
			const bucket = eventsByConv.get(event.conversationId);
			if (bucket) bucket.push(event);
			else eventsByConv.set(event.conversationId, [event]);
		}
		const archivesByConv = new Map<string, DatabaseArchiveChunk[]>();
		for (const item of allArchiveChunks) {
			const bucket = archivesByConv.get(item.conversationId);
			if (bucket) bucket.push(item);
			else archivesByConv.set(item.conversationId, [item]);
		}
		const tracesByConv = new Map<string, DatabaseRetrievalTrace[]>();
		for (const item of allRetrievalTraces) {
			const bucket = tracesByConv.get(item.conversationId);
			if (bucket) bucket.push(item);
			else tracesByConv.set(item.conversationId, [item]);
		}
		const usageByConv = new Map<string, DatabaseRetrievalHitUsage[]>();
		for (const item of allRetrievalUsage) {
			const bucket = usageByConv.get(item.conversationId);
			if (bucket) bucket.push(item);
			else usageByConv.set(item.conversationId, [item]);
		}

		for (let i = 0; i < cleanIds.length; i++) {
			const conv = convs[i];
			if (!conv) continue;
			const messages = (messagesByConv.get(conv.id) ?? []).sort(
				(a, b) => a.timestamp - b.timestamp
			);
			result.set(conv.id, {
				conv,
				messages,
				compactions: compactionsByConv.get(conv.id) ?? [],
				compactionProjectionEvents: eventsByConv.get(conv.id) ?? [],
				archiveChunks: archivesByConv.get(conv.id) ?? [],
				retrievalTraces: tracesByConv.get(conv.id) ?? [],
				retrievalHitUsage: usageByConv.get(conv.id) ?? []
			});
		}
		return result;
	}

	/**
	 * Updates a conversation.
	 *
	 * @param id - Conversation ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the conversation is updated
	 */
	static async updateConversation(
		id: string,
		updates: Partial<Omit<DatabaseConversation, 'id'>>
	): Promise<void> {
		await db[IDXDB_TABLES.conversations].update(id, {
			...updates,
			lastModified: Date.now()
		});
	}

	/**
	 *
	 *
	 * Navigation
	 *
	 *
	 */

	/**
	 * Toggles the pinned status of a conversation.
	 *
	 * @param id - Conversation ID
	 * @returns The new pinned status
	 */
	static async toggleConversationPin(id: string): Promise<boolean> {
		const conversation = await db.conversations.get(id);
		if (!conversation) {
			throw new Error(`Conversation ${id} not found`);
		}
		const newPinnedState = !conversation.pinned;
		await this.updateConversation(id, { pinned: newPinnedState });
		return newPinnedState;
	}

	/**
	 * Toggles the pinned status of each conversation in `ids` inside a single
	 * transaction. Treats `pinned === undefined` as `false`, matching the
	 * semantics of {@link toggleConversationPin} where `!undefined` evaluates
	 * to `true`. Returns the resulting pinned state for every id that was
	 * updated; missing ids are omitted from the map.
	 *
	 * @param ids - Conversation IDs to toggle
	 * @returns Map of id -> new pinned state
	 */
	static async bulkToggleConversationPins(ids: string[]): Promise<Map<string, boolean>> {
		const cleanIds = ids.filter((id): id is string => typeof id === 'string' && id.length > 0);
		const result = new Map<string, boolean>();
		if (cleanIds.length === 0) return result;

		const now = Date.now();
		await db.transaction('rw', db[IDXDB_TABLES.conversations], async () => {
			const convs = await db[IDXDB_TABLES.conversations].bulkGet(cleanIds);
			const updates: DatabaseConversation[] = [];
			for (let i = 0; i < cleanIds.length; i++) {
				const conv = convs[i];
				if (!conv) continue;
				const newPinned = !conv.pinned;
				updates.push({ ...conv, pinned: newPinned, lastModified: now });
				result.set(cleanIds[i], newPinned);
			}
			if (updates.length === 0) return;
			await db[IDXDB_TABLES.conversations].bulkPut(updates);
		});
		return result;
	}

	/**
	 * Updates the conversation's current node (active branch).
	 * This determines which conversation path is currently being viewed.
	 *
	 * @param convId - Conversation ID
	 * @param nodeId - Message ID to set as current node
	 */
	static async updateCurrentNode(convId: string, nodeId: string): Promise<void> {
		await this.updateConversation(convId, {
			currNode: nodeId
		});
	}

	/**
	 * Updates a message.
	 *
	 * @param id - Message ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the message is updated
	 */
	static async updateMessage(
		id: string,
		updates: Partial<Omit<DatabaseMessage, 'id'>>
	): Promise<void> {
		await db[IDXDB_TABLES.messages].update(id, updates);
	}

	/**
	 *
	 *
	 * Import
	 *
	 *
	 */

	/**
	 * Imports multiple conversations and their messages.
	 * Skips conversations that already exist.
	 *
	 * @param data - Array of { conv, messages } objects
	 */
	static async importConversations(
		data: ExportedConversation[]
	): Promise<{ imported: number; skipped: number }> {
		let importedCount = 0;
		let skippedCount = 0;

		return await db.transaction(
			'rw',
			[
				db[IDXDB_TABLES.conversations],
				db[IDXDB_TABLES.messages],
				db[IDXDB_TABLES.compactions],
				db[IDXDB_TABLES.compactionProjectionEvents],
				db[IDXDB_TABLES.archiveChunks],
				db[IDXDB_TABLES.archiveTerms],
				db[IDXDB_TABLES.retrievalTraces],
				db[IDXDB_TABLES.retrievalHitUsage]
			],
			async () => {
				for (const item of data) {
					const {
						conv,
						messages,
						compactions,
						compactionProjectionEvents,
						archiveChunks,
						retrievalTraces,
						retrievalHitUsage
					} = item;

					const existing = await db[IDXDB_TABLES.conversations].get(conv.id);
					if (existing) {
						console.warn(`Conversation "${conv.name}" already exists, skipping...`);
						skippedCount++;
						continue;
					}

					await db[IDXDB_TABLES.conversations].add(conv);
					for (const msg of messages) {
						await db[IDXDB_TABLES.messages].put(msg);
					}
					if (compactions?.length) await db[IDXDB_TABLES.compactions].bulkPut(compactions);
					if (compactionProjectionEvents?.length) {
						await db[IDXDB_TABLES.compactionProjectionEvents].bulkPut(compactionProjectionEvents);
					}
					if (archiveChunks?.length) {
						await db[IDXDB_TABLES.archiveChunks].bulkPut(archiveChunks);
						await db[IDXDB_TABLES.archiveTerms].bulkPut(
							archiveChunks.flatMap((chunk) =>
								chunk.terms.map((term) => ({
									id: `${chunk.id}:${term}`,
									conversationId: chunk.conversationId,
									chunkId: chunk.id,
									term
								}))
							)
						);
					}
					if (retrievalTraces?.length) {
						await db[IDXDB_TABLES.retrievalTraces].bulkPut(retrievalTraces);
					}
					if (retrievalHitUsage?.length) {
						await db[IDXDB_TABLES.retrievalHitUsage].bulkPut(retrievalHitUsage);
					}

					importedCount++;
				}

				return { imported: importedCount, skipped: skippedCount };
			}
		);
	}

	/**
	 *
	 *
	 * Forking
	 *
	 *
	 */

	/**
	 * Forks a conversation at a specific message, creating a new conversation
	 * containing all messages from the root up to (and including) the target message.
	 *
	 * @param sourceConvId - The source conversation ID
	 * @param atMessageId - The message ID to fork at (the new conversation ends here)
	 * @param options - Fork options (name and whether to include attachments)
	 * @returns The newly created conversation
	 */
	static async forkConversation(
		sourceConvId: string,
		atMessageId: string,
		options: { name: string; includeAttachments: boolean }
	): Promise<DatabaseConversation> {
		return await db.transaction(
			'rw',
			[
				db[IDXDB_TABLES.conversations],
				db[IDXDB_TABLES.messages],
				db[IDXDB_TABLES.compactions],
				db[IDXDB_TABLES.compactionProjectionEvents],
				db[IDXDB_TABLES.archiveChunks],
				db[IDXDB_TABLES.archiveTerms],
				db[IDXDB_TABLES.retrievalTraces]
			],
			async () => {
				const sourceConv = await db[IDXDB_TABLES.conversations].get(sourceConvId);
				if (!sourceConv) {
					throw new Error(`Source conversation ${sourceConvId} not found`);
				}

				const allMessages = await db[IDXDB_TABLES.messages]
					.where('convId')
					.equals(sourceConvId)
					.toArray();

				const pathMessages = filterByLeafNodeId(
					allMessages,
					atMessageId,
					true
				) as DatabaseMessage[];
				if (pathMessages.length === 0) {
					throw new Error(`Could not resolve message path to ${atMessageId}`);
				}

				const idMap = new Map<string, string>();

				for (const msg of pathMessages) {
					idMap.set(msg.id, uuid());
				}

				const newConvId = uuid();
				const clonedMessages: DatabaseMessage[] = pathMessages.map((msg) => {
					const newId = idMap.get(msg.id)!;
					const newParent = msg.parent ? (idMap.get(msg.parent) ?? null) : null;
					const newChildren = msg.children
						.filter((childId: string) => idMap.has(childId))
						.map((childId: string) => idMap.get(childId)!);

					return {
						...msg,
						id: newId,
						convId: newConvId,
						parent: newParent,
						children: newChildren,
						extra: options.includeAttachments ? msg.extra : undefined
					};
				});

				const lastClonedMessage = clonedMessages[clonedMessages.length - 1];
				const newConv: DatabaseConversation = {
					id: newConvId,
					name: options.name,
					lastModified: Date.now(),
					currNode: lastClonedMessage.id,
					forkedFromConversationId: sourceConvId,
					mcpServerOverrides: sourceConv.mcpServerOverrides
						? sourceConv.mcpServerOverrides.map((o: McpServerOverride) => ({
								serverId: o.serverId,
								enabled: o.enabled
							}))
						: undefined,
					memoryProject: sourceConv.memoryProject
				};

				await db[IDXDB_TABLES.conversations].add(newConv);

				for (const msg of clonedMessages) {
					await db[IDXDB_TABLES.messages].add(msg);
				}

				const sourceCompactions = await db[IDXDB_TABLES.compactions]
					.where('conversationId')
					.equals(sourceConvId)
					.toArray();
				const compactionIdMap = new Map<string, string>();
				for (const record of sourceCompactions) {
					if (record.status === 'ready' && record.sourceMessageIds.every((id) => idMap.has(id))) {
						compactionIdMap.set(record.id, uuid());
					}
				}
				for (const record of sourceCompactions) {
					const newId = compactionIdMap.get(record.id);
					if (!newId) continue;
					const sourceMessageIds = record.sourceMessageIds.map((id) => idMap.get(id)!);
					const sourceMessages = sourceMessageIds.map(
						(id) => clonedMessages.find((message) => message.id === id)!
					);
					const fingerprintData = JSON.stringify(
						sourceMessages.map((message) => ({
							id: message.id,
							role: message.role,
							content: message.content,
							reasoningContent: message.reasoningContent,
							toolCalls: message.toolCalls,
							toolCallId: message.toolCallId,
							extra: message.extra
						}))
					);
					const digest = await crypto.subtle.digest(
						'SHA-256',
						new TextEncoder().encode(fingerprintData)
					);
					const sourceFingerprint = Array.from(new Uint8Array(digest), (byte) =>
						byte.toString(16).padStart(2, '0')
					).join('');
					await db[IDXDB_TABLES.compactions].add({
						...record,
						id: newId,
						conversationId: newConvId,
						sourceMessageIds,
						deltaSourceMessageIds: record.deltaSourceMessageIds.map((id) => idMap.get(id)!),
						sourceFingerprint,
						previousCompactionId: record.previousCompactionId
							? compactionIdMap.get(record.previousCompactionId)
							: undefined
					});
				}

				const sourceEvents = await db[IDXDB_TABLES.compactionProjectionEvents]
					.where('conversationId')
					.equals(sourceConvId)
					.toArray();
				for (const event of sourceEvents) {
					const anchorMessageId = idMap.get(event.anchorMessageId);
					const compactionId = event.compactionId
						? compactionIdMap.get(event.compactionId)
						: undefined;
					if (!anchorMessageId || (event.action === 'apply' && !compactionId)) continue;
					await db[IDXDB_TABLES.compactionProjectionEvents].add({
						...event,
						id: uuid(),
						conversationId: newConvId,
						anchorMessageId,
						compactionId
					});
				}

				const sourceArchive = await db[IDXDB_TABLES.archiveChunks]
					.where('conversationId')
					.equals(sourceConvId)
					.toArray();
				const archiveIdMap = new Map<string, string>();
				for (const chunk of sourceArchive) {
					const compactionId = compactionIdMap.get(chunk.compactionId);
					if (!compactionId || !chunk.sourceMessageIds.every((id) => idMap.has(id))) continue;
					const newId = uuid();
					archiveIdMap.set(chunk.id, newId);
					const cloned = {
						...chunk,
						id: newId,
						conversationId: newConvId,
						compactionId,
						sourceMessageIds: chunk.sourceMessageIds.map((id) => idMap.get(id)!)
					};
					await db[IDXDB_TABLES.archiveChunks].add(cloned);
					await db[IDXDB_TABLES.archiveTerms].bulkAdd(
						cloned.terms.map((term) => ({
							id: `${newId}:${term}`,
							conversationId: newConvId,
							chunkId: newId,
							term
						}))
					);
				}

				const remapHitId = (id: string): string => {
					if (!id.startsWith('local:')) return id;
					const mapped = archiveIdMap.get(id.slice('local:'.length));
					return mapped ? `local:${mapped}` : id;
				};
				const sourceTraces = await db[IDXDB_TABLES.retrievalTraces]
					.where('conversationId')
					.equals(sourceConvId)
					.toArray();
				let clonedTraceCount = 0;
				for (const trace of sourceTraces) {
					const anchorMessageId = idMap.get(trace.anchorMessageId);
					const responseMessageId = trace.responseMessageId
						? idMap.get(trace.responseMessageId)
						: undefined;
					if (!anchorMessageId || (trace.responseMessageId && !responseMessageId)) continue;
					await db[IDXDB_TABLES.retrievalTraces].add({
						...trace,
						id: uuid(),
						conversationId: newConvId,
						anchorMessageId,
						responseMessageId,
						hits: trace.hits.map((hit) => {
							const provenance = hit.provenance ? { ...hit.provenance } : undefined;
							if (provenance) {
								if (typeof provenance.compactionId === 'string') {
									provenance.compactionId =
										compactionIdMap.get(provenance.compactionId) ?? provenance.compactionId;
								}
								if (typeof provenance.chunkId === 'string') {
									provenance.chunkId = archiveIdMap.get(provenance.chunkId) ?? provenance.chunkId;
								}
								if (Array.isArray(provenance.sourceMessageIds)) {
									provenance.sourceMessageIds = provenance.sourceMessageIds.map((id) =>
										typeof id === 'string' ? (idMap.get(id) ?? id) : id
									);
								}
							}
							return { ...hit, id: remapHitId(hit.id), provenance };
						}),
						injectedHitIds: trace.injectedHitIds.map(remapHitId)
					});
					clonedTraceCount++;
				}
				memoryDebug('retrieval.trace.forked', {
					sourceConversationId: sourceConvId,
					conversationId: newConvId,
					clonedTraceCount
				});

				return newConv;
			}
		);
	}
}
