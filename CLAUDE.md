## Agent skills

### Issue tracker

Local markdown: specs and tickets live under `.scratch/<feature>/` (gitignored). See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Coding standards

Review rules (LLM calls, trace format, turn state, tests) live in `CODING_STANDARDS.md`. Read it when reviewing a diff.
