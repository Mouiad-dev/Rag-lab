# ADR-0004: Open-source license

Date: 2026-10-26
Status: proposed (the maintainer decides)

## Context
The goal is a public repository that anyone can use and extend with plugins. The initial commit already contains a `LICENSE` file with the MIT License. No outside contributions have been accepted yet, so the license can still change without asking contributors. The project depends mostly on permissively licensed libraries; each dependency must still be checked when added.

## Options
| | MIT | Apache-2.0 |
|---|---|---|
| Length and simplicity | Very short, widely understood | Longer |
| Explicit patent grant | No | Yes, with patent-retaliation clause |
| Contribution terms | Implicit | Section 5: contributions are under the same license by default |
| NOTICE / change marking | No | NOTICE file and change statements required when redistributing |
| Compatibility | GPLv2 and v3 compatible | GPLv3 compatible, not GPLv2-only |
| Common in AI infrastructure | Yes | Yes (many ML frameworks and serving tools) |

## Recommendation
**Apache-2.0.** The project invites third-party plugins and contributions in an area (retrieval, ML systems) where patents exist; the explicit patent grant and contribution clause protect users and contributors at little cost. MIT remains a valid choice if maximum simplicity matters more.

## If accepted
- Replace `LICENSE` with the Apache-2.0 text, add a short `NOTICE`, and update README and CONTRIBUTING.
- Optionally require DCO sign-off (`git commit -s`) instead of a CLA.
- If MIT is kept, mark this ADR `accepted (MIT)` and change nothing else.

## Consequences
The license choice also affects which code can be copied into the repo. Third-party or proprietary code is never copied without a compatible license (rule in `.claude/rules/40-security.md`).
