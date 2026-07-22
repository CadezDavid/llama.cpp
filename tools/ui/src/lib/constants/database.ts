/**
 * Database-related constants (IndexedDB, Dexie).
 *
 * Centralized to ensure consistency across the app and simplify future
 * naming changes.
 */

import { STORAGE_APP_NAME } from './storage';

/** IndexedDB database name */
export const DB_NAME = STORAGE_APP_NAME;

/** IndexedDB store / table names */
export const IDXDB_TABLES = {
	conversations: 'conversations',
	messages: 'messages',
	compactions: 'compactions',
	compactionProjectionEvents: 'compactionProjectionEvents'
} as const;

/** IndexedDB store schemas */
export const IDXDB_STORE_SCHEMAS = {
	conversations: 'id, lastModified, currNode, name',
	messages: 'id, convId, type, role, timestamp, parent, children',
	compactions: 'id, conversationId, status, createdAt, [conversationId+createdAt]',
	compactionProjectionEvents:
		'id, conversationId, anchorMessageId, createdAt, [conversationId+anchorMessageId]'
} as const;

export const IDXDB_STORES_V1 = {
	[IDXDB_TABLES.conversations]: IDXDB_STORE_SCHEMAS.conversations,
	[IDXDB_TABLES.messages]: IDXDB_STORE_SCHEMAS.messages
} as const;

/** Combined Dexie stores definition — keys are table names, values are schemas */
export const IDXDB_STORES = {
	[IDXDB_TABLES.conversations]: IDXDB_STORE_SCHEMAS.conversations,
	[IDXDB_TABLES.messages]: IDXDB_STORE_SCHEMAS.messages,
	[IDXDB_TABLES.compactions]: IDXDB_STORE_SCHEMAS.compactions,
	[IDXDB_TABLES.compactionProjectionEvents]: IDXDB_STORE_SCHEMAS.compactionProjectionEvents
} as const;
