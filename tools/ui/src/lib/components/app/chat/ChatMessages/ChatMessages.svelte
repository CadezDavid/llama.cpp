<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import { beforeNavigate, afterNavigate } from '$app/navigation';
	import { ChatMessage, ChatMessageUserPending } from '$lib/components/app';
	import { setChatActionsContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';
	import { chatStore } from '$lib/stores/chat.svelte';
	import {
		chatPendingMessageContent,
		chatPendingMessageExtras,
		chatClearPendingMessage,
		chatInjectPendingMessage
	} from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { DatabaseService } from '$lib/services/database.service';
	import { config } from '$lib/stores/settings.svelte';
	import {
		agenticPendingSteeringMessageContent,
		agenticPendingSteeringMessageExtras,
		agenticClearSteeringMessage,
		agenticInjectSteeringMessage
	} from '$lib/stores/agentic.svelte';
	import {
		buildSiblingInfoMap,
		copyToClipboard,
		formatMessageForClipboard,
		hasAgenticContent,
		visibleTraceMessageId
	} from '$lib/utils';
	import type { DatabaseRetrievalTrace } from '$lib/types';

	interface Props {
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
		onMessagesReady?: (messageCount: number) => void;
	}

	let { messages = [], onUserAction, onMessagesReady }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);
	let retrievalTraces = $state<DatabaseRetrievalTrace[]>([]);
	let isVisible = $state(false);
	let previousConversationId = $state<string | null>(null);
	let previousRouteId = $state<string | null>(null);
	let traceRefreshTimer: ReturnType<typeof setTimeout> | null = null;

	const currentConfig = config();

	setChatActionsContext({
		copy: async (message: DatabaseMessage) => {
			const asPlainText = Boolean(currentConfig.copyTextAttachmentsAsPlainText);
			const clipboardContent = formatMessageForClipboard(
				message.content,
				message.extra,
				asPlainText
			);
			await copyToClipboard(clipboardContent, 'Message copied to clipboard');
		},

		delete: async (message: DatabaseMessage) => {
			await chatStore.deleteMessage(message.id);
			refreshAllMessages();
		},

		navigateToSibling: async (siblingId: string) => {
			await conversationsStore.navigateToSibling(siblingId);
		},

		editWithBranching: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			await chatStore.editMessageWithBranching(message.id, newContent, newExtras);
			refreshAllMessages();
		},

		editWithReplacement: async (
			message: DatabaseMessage,
			newContent: string,
			shouldBranch: boolean
		) => {
			onUserAction?.();
			await chatStore.editAssistantMessage(message.id, newContent, shouldBranch);
			refreshAllMessages();
		},

		editUserMessagePreserveResponses: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);
			refreshAllMessages();
		},

		regenerateWithBranching: async (message: DatabaseMessage, modelOverride?: string) => {
			onUserAction?.();
			await chatStore.regenerateMessageWithBranching(message.id, modelOverride);
			refreshAllMessages();
		},

		continueAssistantMessage: async (message: DatabaseMessage) => {
			onUserAction?.();
			await chatStore.continueAssistantMessage(message.id);
			refreshAllMessages();
		},

		forkConversation: async (
			message: DatabaseMessage,
			options: { name: string; includeAttachments: boolean }
		) => {
			await conversationsStore.forkConversation(message.id, options);
		}
	});

	function refreshAllMessages() {
		const conversation = activeConversation();

		if (conversation) {
			const conversationId = conversation.id;
			Promise.all([
				conversationsStore.getConversationMessages(conversationId),
				DatabaseService.getRetrievalTraces(conversationId)
			]).then(([loadedMessages, loadedTraces]) => {
				if (activeConversation()?.id !== conversationId) return;
				allConversationMessages = loadedMessages;
				retrievalTraces = loadedTraces;
			});
		} else {
			allConversationMessages = [];
			retrievalTraces = [];
		}
	}

	$effect(() => {
		const signature = messages
			.map((message) => `${message.id}:${message.content.length}:${message.toolCalls?.length ?? 0}`)
			.join('|');
		void signature;
		if (traceRefreshTimer) clearTimeout(traceRefreshTimer);
		traceRefreshTimer = setTimeout(() => {
			const conversationId = activeConversation()?.id;
			if (!conversationId) return;
			DatabaseService.getRetrievalTraces(conversationId).then((loaded) => {
				if (activeConversation()?.id === conversationId) retrievalTraces = loaded;
			});
		}, 250);
	});

	onDestroy(() => {
		if (traceRefreshTimer) clearTimeout(traceRefreshTimer);
	});

	// Track conversation changes to trigger transition even on same route
	$effect(() => {
		const conversation = activeConversation();
		const currentId = conversation?.id ?? null;

		if (currentId !== previousConversationId && previousConversationId !== null) {
			isVisible = false;
			requestAnimationFrame(() => {
				refreshAllMessages();
				previousConversationId = currentId;
				requestAnimationFrame(() => {
					isVisible = true;
				});
			});
		} else {
			previousConversationId = currentId;
			if (conversation) {
				refreshAllMessages();
			}
		}
	});

	$effect(() => {
		void allConversationMessages;

		onMessagesReady?.(displayMessages.length);
	});

	onMount(() => {
		requestAnimationFrame(() => {
			isVisible = true;
		});
	});

	beforeNavigate((navigation) => {
		isVisible = false;
		previousRouteId = navigation.from?.route.id ?? null;
	});

	afterNavigate(() => {
		requestAnimationFrame(() => {
			isVisible = true;
		});
	});

	let siblingInfoByMessageId = $derived(buildSiblingInfoMap(allConversationMessages));

	let displayMessages = $derived.by(() => {
		if (!messages.length) {
			return [];
		}

		const filteredMessages = currentConfig.showSystemMessage
			? messages
			: messages.filter((msg) => msg.type !== MessageRole.SYSTEM);

		// Build display entries, grouping agentic sessions into single entries.
		// An agentic session = assistant(with tool_calls) → tool → assistant → tool → ... → assistant(final)
		const result: Array<{
			message: DatabaseMessage;
			toolMessages: DatabaseMessage[];
			isLastAssistantMessage: boolean;
			isLastUserMessage: boolean;
			nextAssistantMessage: DatabaseMessage | null;
			retrievalTraces: DatabaseRetrievalTrace[];
			siblingInfo: ChatMessageSiblingInfo;
		}> = [];
		const visibleIds = new Set(filteredMessages.map((message) => message.id));
		const tracesByUserId = new SvelteMap<string, DatabaseRetrievalTrace[]>();
		for (const trace of retrievalTraces) {
			const relatedId = visibleTraceMessageId(trace, visibleIds);
			if (!relatedId) continue;
			const relatedIndex = filteredMessages.findIndex((message) => message.id === relatedId);
			let userId: string | null = null;
			for (let index = relatedIndex; index >= 0; index--) {
				if (filteredMessages[index].role === MessageRole.USER) {
					userId = filteredMessages[index].id;
					break;
				}
			}
			if (!userId) continue;
			const bucket = tracesByUserId.get(userId);
			if (bucket) bucket.push(trace);
			else tracesByUserId.set(userId, [trace]);
		}

		for (let i = 0; i < filteredMessages.length; i++) {
			const msg = filteredMessages[i];

			// Skip tool messages - they're grouped with preceding assistant
			if (msg.role === MessageRole.TOOL) continue;

			const toolMessages: DatabaseMessage[] = [];
			if (msg.role === MessageRole.ASSISTANT && hasAgenticContent(msg)) {
				let j = i + 1;

				while (j < filteredMessages.length) {
					const next = filteredMessages[j];

					if (next.role === MessageRole.TOOL) {
						toolMessages.push(next);

						j++;
					} else if (next.role === MessageRole.ASSISTANT) {
						toolMessages.push(next);

						j++;
					} else {
						break;
					}
				}

				i = j - 1;
			} else if (msg.role === MessageRole.ASSISTANT) {
				let j = i + 1;

				while (j < filteredMessages.length && filteredMessages[j].role === MessageRole.TOOL) {
					toolMessages.push(filteredMessages[j]);
					j++;
				}
			}

			const siblingInfo = siblingInfoByMessageId.get(msg.id) ?? {
				message: msg,
				siblingIds: [msg.id],
				currentIndex: 0,
				totalSiblings: 1
			};

			result.push({
				message: msg,
				toolMessages,
				isLastAssistantMessage: false,
				isLastUserMessage: false,
				nextAssistantMessage: null,
				retrievalTraces: tracesByUserId.get(msg.id) ?? [],
				siblingInfo
			});
		}

		let lastAssistantIdx = -1;
		for (let i = result.length - 1; i >= 0; i--) {
			if (result[i].message.role === MessageRole.ASSISTANT) {
				result[i].isLastAssistantMessage = true;
				lastAssistantIdx = i;
				break;
			}
		}

		if (lastAssistantIdx > 0 && result[lastAssistantIdx - 1].message.role === MessageRole.USER) {
			result[lastAssistantIdx - 1].isLastUserMessage = true;
		}

		for (let i = 0; i < result.length; i++) {
			if (result[i].message.role !== MessageRole.USER) continue;

			for (let j = i + 1; j < result.length; j++) {
				if (result[j].message.role === MessageRole.ASSISTANT) {
					result[i].nextAssistantMessage = result[j].message;
					break;
				}
			}
		}

		return result;
	});
</script>

<div
	class="transition-opacity duration-500 ease-out
		{isVisible ? 'opacity-100' : 'opacity-0'}
		{previousRouteId === '/(chat)/chat/[id]' ? '' : 'delay-300'}"
>
	{#each displayMessages as { message, toolMessages, isLastAssistantMessage, isLastUserMessage, nextAssistantMessage, retrievalTraces, siblingInfo } (message.id)}
		<ChatMessage
			class="mx-auto mt-12 w-full max-w-3xl"
			{message}
			{toolMessages}
			{isLastAssistantMessage}
			{isLastUserMessage}
			{nextAssistantMessage}
			{retrievalTraces}
			{siblingInfo}
		/>
	{/each}

	{#if activeConversation() && agenticPendingSteeringMessageContent(activeConversation()!.id)}
		{@const convId = activeConversation()!.id}
		{@const pendingContent = agenticPendingSteeringMessageContent(convId)}

		{#if pendingContent}
			<ChatMessageUserPending
				class="mx-auto mt-12 w-full max-w-[48rem]"
				content={pendingContent}
				extras={agenticPendingSteeringMessageExtras(convId)}
				onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
				onEdit={(newContent, extras) => agenticInjectSteeringMessage(convId, newContent, extras)}
				onDelete={() => agenticClearSteeringMessage(convId)}
			/>
		{/if}
	{:else if activeConversation() && chatPendingMessageContent(activeConversation()!.id)}
		{@const convId = activeConversation()!.id}
		{@const pendingContent = chatPendingMessageContent(convId)}

		{#if pendingContent}
			<ChatMessageUserPending
				class="mx-auto mt-12 w-full max-w-[48rem]"
				content={pendingContent}
				extras={chatPendingMessageExtras(convId)}
				onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
				onEdit={(newContent, extras) => chatInjectPendingMessage(convId, newContent, extras)}
				onDelete={() => chatClearPendingMessage(convId)}
			/>
		{/if}
	{/if}
</div>
