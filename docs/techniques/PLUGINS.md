# Technique plugins: author guide

Status: plugin system v1 (P7.1). The mechanism is tested; no third-party plugin is wired into the API or worker yet. The sample plugin `examples/plugins/rag_foundry_sample_plugin` is a simulation of C11, not an implementation.

Read [ADR-0002](../decisions/ADR-0002-technique-plugin-system.md) first. Plugins are trusted, in-process Python code. Never load untrusted or model-generated code as a plugin. Untrusted code runs only in the sandbox, which is a separate mechanism.

## 1. Packaging: the entry point

A plugin is an installable Python package. Register each technique under the entry-point group `rag_foundry.techniques`:

```toml
[project.entry-points."rag_foundry.techniques"]
C11 = "my_package.plugin:PageBlockChunker"
```

- The entry-point name must equal the catalogue ID (`C11`). A name that differs from the ID is a failure.
- The value points to the plugin class. The class carries the manifest: `technique_id`, `kind`, `version` (`X.Y.Z`), `fidelity` (`faithful`, `adaptation` or `simulation`) and `config_model` (a pydantic model that must set `extra="forbid"`).
- Install the package into the environment. The sample is installed with `uv pip install -e examples/plugins/rag_foundry_sample_plugin`; it is outside the uv workspace.

## 2. Catalogue IDs and kinds

- Plugins may only use IDs that already exist in `config/techniques.json` (all 72 entries). A new catalogue ID needs a human catalogue change; plugins and the Tech Scout cannot add one. The Scout only proposes (`proposed`); a human approves.
- `allowed_plugin_kinds` is a required top-level map in `config/techniques.json` (schema v2, additive; owner approved 2026-11-12). It maps each catalogue category to the plugin kinds it may use. A kind outside the map is rejected: `registry.check`/`register` raise `PluginKindError`, and discovery reports it as an `error_type` inside `PluginDiscoveryError`.
- Kind protocols are frozen only for `chunker`, `retriever` and `fusion`. The other kinds listed in ADR-0002 are not frozen in v1.

## 3. Discovery

Discovery is explicit. Nothing runs at import time.

```python
from pathlib import Path
from rag_foundry_chunking import register_builtin_plugins as register_chunking
from rag_foundry_indexing import register_builtin_plugins as register_indexing
from rag_foundry_techniques import TechniqueRegistry, discover_plugins, load_catalogue

registry = TechniqueRegistry(load_catalogue(Path("config/techniques.json")))
register_chunking(registry)  # built-ins first: discover_plugins does not register them
register_indexing(registry)
discover_plugins(registry)
```

Rules:

- The caller must register the built-in plugins before calling `discover_plugins`; discovery never registers them itself. With built-ins registered, a third-party plugin with the same ID as a built-in is an error, not an override. On an empty registry that check cannot fire.
- Two entry points with the same ID are both rejected (a duplicate between entry points is an error).
- Candidates are processed in sorted order.
- All-or-nothing: if any candidate fails, discovery raises `PluginDiscoveryError` and lists every failure. The registry is left unchanged.
- Every failure is collected as a `PluginDiscoveryFailure` in `PluginDiscoveryError.failures`, with this `error_type`:

| Failure | `error_type` |
|---|---|
| wrong entry-point group; value not a class; missing or invalid manifest; config model without `extra="forbid"`; entry-point name not equal to the ID | `TechniqueError` |
| ID not in the catalogue | `UnknownTechniqueError` |
| kind not allowed for the category | `PluginKindError` |
| duplicate of a registered plugin, or two entry points with the same ID | `DuplicatePluginError` |
| import error, or any other exception raised by the check | the exception's own class name (for example `ModuleNotFoundError`) |

- Not rolled back: `.load()` runs the plugin module's import side effects. These are not undone when discovery fails.

## 4. Contract kit

Use the kit before you publish a plugin:

- `check_plugin_contract(plugin_cls, catalogue, *, sample_config=None)`: checks a valid manifest (the fidelity value itself is validated by the `PluginManifest` model, not by the kit), the ID is in the catalogue, the kind is allowed, the config model sets `extra="forbid"` and produces a JSON schema, an empty registry raises `TechniqueUnavailableError` for the ID, and a second registration raises `DuplicatePluginError`. With `sample_config` (a raw mapping) it also checks that the config validates and that an unknown key is rejected.
- `check_chunker_run(plugin_cls, config, doc)`: validates the raw `config` mapping through the config model, runs the chunker on `doc` and checks that every result is a `ChunkSpan` inside the text, in non-decreasing start order.
- Both raise `PluginContractError` on failure. All 47 built-in plugins pass the kit.

The sample README lists the same calls: `examples/plugins/rag_foundry_sample_plugin/README.md`.

## 5. Scaffold

`scripts/new_technique.py` writes a plugin package skeleton for an existing catalogue ID:

```bash
uv run python scripts/new_technique.py C11 --kind chunker --fidelity simulation --out <dir>
```

`--fidelity` is required. `--kind` may be left out only when the category allows exactly one kind; otherwise leaving it out is a refusal. `--package` sets the package name (default `rag_foundry_plugin_<id>`), and `--catalogue` points at another catalogue file (default `config/techniques.json`; a catalogue that does not load is a refusal).

The skeleton has an entry point, a `Plugin` whose method raises `TechniqueUnavailableError` until you implement it, a README and a contract test. It never edits `config/techniques.json`.

Refusals (exit code 2, one error line on stderr):

- unknown ID (IDs are case-sensitive, so a lowercase ID is unknown);
- a kind not allowed for the category, or no `--kind` when the category allows more than one;
- an ID already `implemented` or `validated`;
- an ID that already has a built-in plugin;
- a non-empty output folder;
- a bad package name.

Limits:

- The script does not refuse an `--out` inside the repository. Write generated plugins outside the repository.
- Only the `chunker` and `graph_builder` templates are exercised by tests. The `retriever` and `fusion` templates are not tested.

## 6. Trust and boundaries

- Plugins run in the same process as the code that calls `discover_plugins` with full access. The API and worker do not call discovery yet. Review plugins as you would any dependency.
- Never run model-generated code as a plugin. Application code execution uses the designed sandbox port, not a plugin.
- Discovery does not check signatures and there is no allowlist yet.
- Do not claim a plugin is an implementation of a technique unless it meets the fidelity and evidence rules in the catalogue (`validated` needs evidence). A plugin that is a simulation must say so in its manifest.

## 7. Checklist

1. Choose an existing catalogue ID and check its category in `allowed_plugin_kinds`.
2. Scaffold outside the repository, implement the class and keep the fidelity honest.
3. Run `check_plugin_contract` and your contract test.
4. Install the package, register the built-ins, then call `discover_plugins` on a registry built from `config/techniques.json`.
5. Run `uv run python scripts/tasks.py check` in the repository if you change repository code.
