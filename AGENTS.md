# Agent instructions

## Git workflow

- Use Git actively as a progress log, not only as a final delivery step.
- Check `git status` and review the relevant diff before editing and before every commit.
- Commit after each coherent, working milestone. Keep commits small enough to review and useful enough to restore; avoid both one large end-of-task commit and noisy commits for incomplete or trivial changes.
- Stage only the files or hunks that belong to the current change. Preserve unrelated work already present in the worktree.
- Run the relevant checks before committing when practical. If a milestone cannot be fully verified, record that clearly in the commit body or task handoff.
- Do not rewrite, squash, amend, revert, or discard existing commits or worktree changes unless the user explicitly asks.
- Treat local commits as part of normal implementation work. Push only when the user explicitly requests it.

### Commit messages

- Use a concise, imperative subject that names the affected area and the intent of the change.
- Explain why the change was needed in the body when the reason is not obvious from the diff.
- Mention important constraints, tradeoffs, or verification results when they will help the next agent continue the work.
- Avoid vague messages such as `updates`, `fix stuff`, or `work in progress`.

Examples:

```text
vegas: reuse the verification graph across draft tokens

Avoid rebuilding the graph for every token while preserving the existing
fallback path. Verified with the focused Vegas benchmark.
```

```text
cuda: guard sparse attention against empty masks
```
