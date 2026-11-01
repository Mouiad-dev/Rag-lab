# Development agent workflow

How this repository is built with Claude Code subagents while keeping token spend low. Agent definitions live in `.claude/agents/`.

## Team

| Agent | Model | Tools | Job |
|---|---|---|---|
| `planner` | Opus (`claude-opus-5-5`) | Read, Grep, Glob, Write (`docs/tasks/` only) | Split one roadmap step into small task specs |
| `coder` | Sonnet (`claude-sonnet-5-5`) | Read, Edit, Write, Grep, Glob, Bash | Implement one spec |
| `coder-light` | Haiku (`claude-haiku-4-5-20251001`) | Same as coder | Mechanical specs: boilerplate, config, renames, docs, simple tests |
| `reviewer` | Opus (`claude-opus-5-5`) | Read, Grep, Glob, Bash (read-only git and test commands) | Review specs before coding and diffs after coding |

Model notes: the plan originally asked for "Opus 4.8" and "Haiku 5.5", which are not current models. Opus 5.5 and Haiku 4.5 are used instead. The frontmatter uses the aliases `opus`, `sonnet`, `haiku`, which Claude Code resolves to the current model of each family; pin full model IDs in the frontmatter if you need exact reproducibility.

## Loop

```text
roadmap step ──► planner ──► task spec(s) in docs/tasks/
                                │
                     (optional) reviewer: plan review, 1 round
                                │
                 coder or coder-light ──► diff + test results
                                │
                       reviewer: spec + git diff
                     APPROVE ──► user commits; PROGRESS.md updated
                     CHANGES_REQUESTED ──► coder fixes (max 2 review rounds)
                     still open after round 2 ──► user decides
```

1. User or main session: "plan P2.1".
2. `planner` reads PROGRESS, SUMMARY, the roadmap step and relevant ADRs; writes 1–5 specs from [TASK.md](../templates/TASK.md).
3. Plan review is optional: use it for security-sensitive, schema or cross-package specs.
4. The main session runs `coder` or `coder-light` with the spec path, as the spec's `agent:` field says.
5. `reviewer` gets the spec path and reviews `git diff`.
6. After approval the main session updates `docs/PROGRESS.md` (3–6 lines) and the user commits.

## Token-budget rules

- **Small tasks.** One spec = one reviewable diff, target under ~300 changed lines and under ~8 files read.
- **Short context.** Agents read [SUMMARY.md](../architecture/SUMMARY.md) (under 150 lines), never the whole `PROJECT_ARCH.md`. A spec may cite one section like `PROJECT_ARCH.md §9`; read only that section.
- **Spec is the contract.** The coder reads the spec and the files listed in it, nothing else unless blocked (and then reports it).
- **Diff-only review.** The reviewer reads the spec and `git diff`; opens other files only to confirm a suspected defect.
- **Cheap model for cheap work.** `coder-light` (Haiku) for mechanical specs. It escalates instead of guessing.
- **Review cap.** At most 2 review rounds per spec. Then the user decides.
- **No re-reading.** Do not reread files already summarized in the spec. Do not paste whole files into reports.
- **Short reports.** Coder ≤ 15 lines, coder-light ≤ 10, reviewer ≤ 20, planner summary ≤ 5.
- **Focused checks.** Run the spec's acceptance commands and focused tests, not the full suite after every edit; CI runs everything.
- **Always-loaded files stay small.** `CLAUDE.md` stays short; most rules are path-scoped so they load only when matching files are touched.
- **Progress is the memory.** `docs/PROGRESS.md` holds the current step and last results so new sessions do not rediscover state.

## When to skip the loop

Typos, one-line doc fixes and pure formatting can be done directly by the main session or `coder-light` without a spec or review.

## Safety

No agent commits, pushes, publishes, calls paid APIs or downloads large models unless the user approved that specific action. Specs flag such needs with `needs_user_ok: yes`.
