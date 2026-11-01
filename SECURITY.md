# Security policy

## Supported versions

The project is in planning and has no release yet. Once releases start, the latest minor release receives security fixes.

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub private vulnerability reporting for this repository (Security tab, then "Report a vulnerability"). If that is unavailable, contact the maintainers at **[security contact to be added before public release]**.

Include the affected component and version or commit, reproduction steps, impact, and any suggested fix. We aim to acknowledge reports within 7 days.

## Scope notes

- Provider API keys must only come from environment variables or secret stores. A key appearing in logs, traces, configs or the web bundle is a vulnerability.
- Cross-tenant data access, prompt injection that bypasses authorization, sandbox escapes and SQL tool escapes are in scope.
- The Claude Code development hooks in `.claude/hooks/` are developer conveniences, not a security boundary.
