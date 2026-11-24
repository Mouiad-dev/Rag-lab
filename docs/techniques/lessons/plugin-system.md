# Plugin system v1 (not a catalogue technique)

This page explains what P7.1 proves and what it does not. It is not a technique lesson and adds no catalogue entry. The author guide is [PLUGINS.md](../PLUGINS.md); the decision is [ADR-0002](../../decisions/ADR-0002-technique-plugin-system.md).

## What v1 proves

- A third-party package can register a catalogue ID through the entry-point group `rag_foundry.techniques`.
- Discovery is explicit and sorted. It is all-or-nothing for the registry: if any candidate fails, nothing is registered and one error lists every failure.
- The caller registers the built-ins first (discovery does not); then a duplicate ID is an error, not an override.
- The sample plugin `examples/plugins/rag_foundry_sample_plugin` (C11, fidelity `simulation`) installs and is found with no change to core code. Evidence: [sample_plugin_install.md](../../evidence/P7.1/sample_plugin_install.md).
- The contract kit (`check_plugin_contract`, `check_chunker_run`) passes for all 47 built-in plugins.

## What v1 does not prove

- Discovery is not wired into the API or worker yet. Nothing discovers plugins at startup.
- There is no plugin signing and no allowlist. Any installed package that uses the group is trusted.
- There are no experimental local IDs. A plugin cannot add an ID that is missing from the catalogue.
- Kind protocols are frozen only for `chunker`, `retriever` and `fusion`.
- All-or-nothing applies to the registry only. Import side effects of loaded modules are not rolled back when discovery fails.
- Plugins are trusted in-process code. They are not a sandbox, and a simulation plugin is not evidence that a technique works.
