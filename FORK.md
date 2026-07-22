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
- prompt and token diagnostics for understanding context use and cache costs.

The detailed design is in
[`llama_cpp_context_compaction_and_spomin_design.md`](llama_cpp_context_compaction_and_spomin_design.md).

## Current status

The shared foundation is implemented in the SvelteKit WebUI under `tools/ui`.
It provides one context-preparation path for normal chat, continuation, and
agentic requests. It can create validated prompt projections, attach ephemeral
context without changing stored history, and measure the rendered prompt with
the existing llama-server template and tokenization endpoints.

The feature-specific stages are still under development:

- compaction policy, records, preview, restore, and automatic triggering;
- Spomin configuration and explicit retrieval;
- indexing and retrieval of literal conversation excerpts;
- automatic retrieval orchestration, ranking, and diagnostics.

## Branch guide

The branches are deliberately separated so that upstream updates, TurboQuant,
individual memory features, and combined builds can be maintained independently.

| Branch                          | Purpose                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `mainline`                      | Clean tracking branch for current upstream llama.cpp. Fork-specific features do not belong here. |
| `turboquant-original`           | Original TurboQuant development line, kept as a reference without the fork's mainline merges.    |
| `turboquant`                    | TurboQuant combined with newer llama.cpp changes and local TurboQuant-related fixes.             |
| `memory/foundation`             | Shared prompt-projection, ephemeral-context, request-building, and token-measurement foundation. |
| `memory/compaction`             | Conversation compaction implementation and tests.                                                |
| `memory/spomin`                 | Spomin configuration, storage integration, and explicit recall.                                  |
| `memory/automatic-retrieval`    | Automatic conversation and long-term memory retrieval before inference.                          |
| `integration/mainline-memory`   | Completed memory features integrated on the normal llama.cpp base.                               |
| `integration/turboquant-memory` | Completed memory features integrated with the maintained TurboQuant branch.                      |

The `memory/*` branches are development lines, not separate competing products.
Each feature should be developed and reviewed on its own branch, then combined
through the integration branches.

## Which branch to use

- Use `mainline` when you want an unmodified upstream-oriented llama.cpp base.
- Use `turboquant-original` to inspect or compare against the original
  TurboQuant work.
- Use `turboquant` when you want TurboQuant with the fork's newer llama.cpp
  baseline but without the memory feature stack.
- Use `integration/mainline-memory` for the combined memory work without
  TurboQuant.
- Use `integration/turboquant-memory` for the full combined fork.
- Use a `memory/*` branch only when developing or testing that specific stage.

Until the feature stages are complete, integration branches may contain
foundational or partially integrated functionality rather than a finished
end-user memory system.

## Updating the fork

Upstream and feature history should remain easy to distinguish:

1. Update `mainline` from `ggml-org/llama.cpp`.
2. Preserve `turboquant-original` as the original TurboQuant reference.
3. Merge current mainline changes into `turboquant` when needed.
4. Base feature work on `memory/foundation` and keep feature commits on the
   matching `memory/*` branch.
5. Combine completed memory stages in `integration/mainline-memory`.
6. Merge the combined memory work into `integration/turboquant-memory` and
   resolve TurboQuant-specific conflicts there.

This structure keeps the two external baselines reproducible and prevents
experimental memory work from becoming mixed into every branch.

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

## Scope and support

This fork is intended for personal experimentation and local use. Its branch
names, incomplete features, and integration points may change as the memory
design is implemented. Problems that only exist in this fork should be tracked
and diagnosed here rather than reported as upstream llama.cpp issues.
