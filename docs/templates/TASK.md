# Task <STEP>-<nn>: <short title>

```yaml
roadmap_step: P0.0
agent: coder            # coder | coder-light
size: S                 # XS | S | M  (M is the maximum; split larger work)
needs_user_ok: no       # yes if paid calls, downloads, new dependencies, schema or security changes
catalogue_ids: []       # e.g. [C05, R02]
```

## Goal
One or two sentences: the observable outcome.

## Read only these files
- `path/to/file.py` (why)
- `docs/architecture/PROJECT_ARCH.md §N` (only if needed)

## Change
- Create `path/new.py`: what it contains.
- Edit `path/existing.py`: what changes.

## Constraints
- Interfaces, ADRs or rules that apply. Things not to touch.

## Acceptance
```bash
uv run pytest tests/path -q   # expected: N passed
```
- Behavioral checks a reviewer can confirm from the diff.

## Out of scope
- What a later task will do.

## Result (filled by coder)
- Files changed:
- Commands and results:
- Deviations:
