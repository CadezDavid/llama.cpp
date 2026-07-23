# Fork TODO

## Memory context visibility

- [x] Show the recalled context attached to each model request in the chat UI.
  Keep the normal transcript clean by displaying a collapsed indicator beneath
  the user message, such as `Memory context: 2 recalled items`.
- [x] In the expanded view, distinguish Spomin memories from excerpts recalled
  from compacted conversation history. Show the injected text, source, memory or
  fragment ID, relevance score, project when available, and token count.
- [x] Include compacted-context snippets that were inserted into the request,
  not only the compaction summary. Make it clear which text was sent to the
  model and which candidates were skipped.
- [x] Preserve request-specific history: the indicator must describe the exact
  context used for that message even if memories, thresholds, or the active
  compaction change later.
- [x] Keep retrieved context visually separate from the stored user message.
  Recalled text is temporary model context and must not appear as if the user
  wrote it.
- [x] Provide accessible collapsed and expanded states, avoid exposing secrets,
  and cover rendering, persistence, export/import, branching, and deleted-memory
  behavior with tests.
- [x] Make the "Compact conversation" popup better: the way the user should be able
  to select "compact through turn" should be a line from first message to last
  message. The lines should have points on it that represent points where we
  can "cut" the conversation and compact everything before it. Points should be
  spaced based on how much tokens of context is between them.The user then
  drags the line starting from the left and going to the point and compacts
  everything on the left. This is a more interactive and cleaner way to
  implement user-selected compaction. It also makes the user immediately aware
  of how much context they are compacting. Hovering points one the line should
  also display what message this is and perhaps how it starts, first 30 chars
  or something similar. You need to refine and develop this idea completely.
  Use what i wrote here as a base, not a finished design.
