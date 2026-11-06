# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body-file <file>`, or `--body` with a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --json number,title,body,labels,comments`.
- **List issues**: `gh issue list --state open --json number,title,labels` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Tick a checkbox**: edit the body with `gh issue edit <number> --body-file <file>`.
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v`; `gh` does this automatically when run inside a clone.

Triage state is a label on the issue (see `triage-labels.md` for the strings).

## Specs and tickets

- A spec is one issue, titled `Spec: <feature>`.
- A ticket is one issue per piece of work, never a single combined tickets issue. Its title starts with a two-digit number that continues across the repo: `11: Tutor gives a hint, not the answer`. That number is not the GitHub issue number, so refer to tickets by `#<issue number>`.
- A ticket's body has these sections, in order:
  - `## Parent`: `#<spec issue>`
  - `## What to build`
  - `## Acceptance criteria`: a checkbox list
  - `## Blocked by`: `- #<issue> <title>` per blocker, or `None (can start immediately)`
- A ticket is unblocked when every issue it lists is closed.

## When a ticket is done

Work closes through a pull request: its body says `Closes #<number>` for each ticket, so the ticket closes when the work reaches `main`. Before that, tick every checkbox on the ticket. Results a ticket asks to note, such as a live check, go in a comment.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

Each acceptance checkbox names what a test can observe: a return value, a trace row and its field, a stored record. A requirement in the ticket body ("every attempt is traced") gets its own checkbox naming where it shows ("the tool row lists each rephrased query").

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above. The user will normally pass the issue number directly.

Before starting, match every *What to build* requirement to a checkbox. For a requirement with no checkbox, ask the user how it should behave, then add the box to the ticket.

## Pull requests as a triage surface

**PRs as a request surface: no.** GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue: `gh issue create --parent <map> ...`, or `gh issue edit <map> --add-sub-issue <child>` afterwards (`gh` 2.94+). Where sub-issues aren't available, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's native issue dependencies. Add an edge with `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric database id (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`, not the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children, drop any with an open blocker or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
