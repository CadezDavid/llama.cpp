# About This Fork

This is a personal, experimental fork of
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). It combines a
regularly updated llama.cpp base with TurboQuant work and a set of WebUI
features for long-running conversations.

This repository is not the upstream llama.cpp project. For the official
project, documentation, releases, issues, and contribution process, use
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).

## What the fork is for

The main goal is to let local conversations continue beyond a model's context
window without silently discarding their history. The planned system separates
three things that are often treated as one:

- the complete conversation stored by the WebUI;
- the smaller context selected for the next model request;
- the server-side prompt and KV cache used to perform inference.

On top of that separation, the fork is adding:

- reversible conversation compaction;
- retrieval of exact excerpts from compacted conversation history;
- optional long-term memory retrieval through Spomin;
- automatic retrieval before inference, without requiring the model to call a
  tool;
- prompt and token diagnostics for understanding context use and cache costs;
- upload-time summarization and semantic access for large text attachments.

The detailed design is in
[`llama_cpp_context_compaction_and_spomin_design.md`](llama_cpp_context_compaction_and_spomin_design.md).

## Current status

The complete memory stack is implemented in the SvelteKit WebUI under
`tools/ui`. Normal chat, continuation, and agentic requests use one
context-preparation path. Compaction can be disabled, requested before sending,
or run automatically when the rendered prompt crosses a configurable percentage
of usable model input. The policy reserves output and safety capacity, targets a
smaller post-compaction prompt, and keeps a configurable recent tail literal.

From a conversation's menu, users can also measure old complete turns, generate
a structured compaction with the current model, preview the token savings, apply
it without changing the stored messages, inspect its sources and diagnostics,
recompact it, or restore the original history. Compaction metadata is retained
by conversation export, import, and compatible branch forks.

Compaction also writes its literal source messages into a local IndexedDB
archive in the same transaction that activates the summary. Before each model
request, the WebUI searches that archive with embeddings
while querying Spomin in parallel. It injects only the highest-scoring results
that pass configured thresholds and token budgets. Repetition cooldowns avoid
putting the same memory into consecutive requests unless the topic, score, or
compaction generation changes or the user explicitly asks to recall it.

The Memory settings page configures both providers, supports per-conversation
Spomin project overrides, offers explicit create/edit/delete controls, and
shows recent retrieval decisions. Automatic Spomin retrieval uses a two-second
fail-open deadline and never blocks a chat request when the provider is
unavailable. Spomin mutations remain explicit; automatic recall does not create
or modify long-term memories.

Each completed request displays a collapsed memory-context indicator below the
corresponding user turn. Its expanded view distinguishes Spomin memories from
compacted conversation excerpts, shows the exact text sent to the model, and
explains why other candidates were skipped. These are immutable request
snapshots, so later edits or deletions in the source memory do not rewrite chat
history.

Manual and confirmed compaction use a token-weighted timeline. Its points are
safe complete-turn boundaries, the selected left portion is compacted, and the
protected recent tail remains literal. Pointer, touch, and keyboard controls
all select from the same validated candidates used by automatic compaction.

Large text attachments are measured during upload. Attachments larger than 8%
of usable model input are summarized with the selected generation model and
indexed with Jina. The summarization request is transient: the existing
conversation can be restored from the server RAM cache without retaining the
attachment prompt. The generation model receives a synopsis and two conditional
browser-local tools instead of the complete document. Retrieval is semantic-only
and fails visibly.

## Branch guide

The branches are deliberately separated so that upstream updates, TurboQuant,
individual memory stages, and the two complete products remain easy to
distinguish.

| Branch                        | Purpose                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `main`                        | Complete local product containing all maintained features.                                      |
| `upstream`                    | Clean tracking branch for current upstream llama.cpp.                                            |
| `turboquant`                  | Exact TheTom TurboQuant development line.                                                        |
| `feature/compaction`          | Memory foundation plus conversation compaction implementation and tests.                         |
| `feature/spomin`              | Compaction stage plus Spomin configuration, storage integration, and explicit recall.            |
| `feature/automatic-retrieval` | Spomin stage plus automatic conversation and long-term memory retrieval before inference.        |
| `feature/memory`              | Complete memory product without TurboQuant, retained for isolated testing.                        |
| `feat/attachment-handling`    | Large attachment implementation before integration into `main`.                                  |

The `feature/*` branches are dependency-ordered intermediate builds, not
separate competing products. The `feature/memory` branch is the canonical
landing branch for the complete non-TurboQuant memory stack.

## Which branch to use

- Use `main` for normal operation.
- Use `upstream` for an unmodified upstream-oriented llama.cpp base.
- Use `turboquant` to inspect or update from TheTom's TurboQuant work.
- Use `feature/memory` for the complete memory product without TurboQuant.
- Use a `feature/*` branch only when developing or testing that intermediate
  stage.

The two product branches contain the completed memory stack. The feature
branches remain useful for focused testing and maintenance.

## Updating the fork

Upstream and feature history should remain easy to distinguish:

1. Fast-forward `upstream` from `ggml-org/llama.cpp`.
2. Fast-forward `turboquant` from TheTom's TurboQuant branch.
3. Integrate selected upstream, TurboQuant, and completed feature work into
   `main`.
4. Use the matching `feature/*` branch when an intermediate stage needs
   isolated development or testing.
This structure keeps external baselines reproducible without growing the
product branch name whenever another feature is added.

## Building and testing

Normal llama.cpp build and server instructions remain in the upstream
documentation:

- [`docs/build.md`](docs/build.md)
- [`tools/server/README.md`](tools/server/README.md)
- [`tools/ui/README.md`](tools/ui/README.md)

Memory-related WebUI changes should at minimum pass the UI type checks, lint,
unit tests, and production build. Changes merged into a TurboQuant integration
branch should also build the CUDA-enabled `llama-server` configuration used by
that branch.

## Debugging

llama-server uses `-lv 4` for trace logging, `-lv 5` for debug logging, and
`-v` for all available logs. Options passed through the local launcher belong
after `--`, for example:

```bash
~/scripts/start-llm.sh qwen -- --log-verbosity 5
```

Memory decisions happen in the browser as well as the server. Enable
`Memory diagnostics` under the WebUI Memory settings, save the settings, and
open the browser developer console. Structured `[Memory]` events cover
compaction policy and activation, recalled-context selection and cooldowns,
embedding failures, Spomin requests, prompt-budget trimming, and final context
insertion. These logs redact prompt text, memory text, retrieval queries,
embeddings, summaries, and credentials.

## Scope and support

This fork is intended for personal experimentation and local use. Its branch
names, incomplete features, and integration points may change as the memory
design is implemented. Problems that only exist in this fork should be tracked
and diagnosed here rather than reported as upstream llama.cpp issues.
