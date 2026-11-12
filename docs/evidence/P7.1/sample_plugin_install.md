# P7.1 sample plugin: real offline install and discovery

Run by the orchestrator on 2026-11-13, local machine (Windows 11, project `.venv`, Python 3.13.5), branch `p1/toolchain-and-skeleton`. Offline: no package was downloaded (the build backend `hatchling` was already in the uv cache). No paid calls.

This proves the roadmap's "done when" for P7.1: the sample external plugin in `examples/plugins/rag_foundry_sample_plugin` installs as its own distribution and is found and registered through the `rag_foundry.techniques` entry-point group, with no change to core code, the root `pyproject.toml`, `uv.lock` or `config/techniques.json`. The sample is a simulation (fidelity `simulation`); it is not an implementation of C11, which stays `planned` in the catalogue.

## 1. Install (editable, no dependencies, offline)

```text
$ uv pip install --offline --no-deps -e examples/plugins/rag_foundry_sample_plugin
Resolved 1 package in 23ms
   Building rag-foundry-sample-plugin @ file:///C:/Dev/import/Rag-lab/examples/plugins/rag_foundry_sample_plugin
      Built rag-foundry-sample-plugin @ file:///C:/Dev/import/Rag-lab/examples/plugins/rag_foundry_sample_plugin
Prepared 1 package in 2.59s
Installed 1 package in 17ms
 + rag-foundry-sample-plugin==0.1.0 (from file:///C:/Dev/import/Rag-lab/examples/plugins/rag_foundry_sample_plugin)
exit=0
```

## 2. Discover

```text
$ uv run --no-sync python -c "...discover_plugins..."
(DiscoveredPlugin(entry_point='C11', value='rag_foundry_sample_plugin:PageBlockChunker', distribution='rag-foundry-sample-plugin', distribution_version='0.1.0', technique_id='C11', kind='chunker', plugin_version='0.1.0'),)
('C11',)
exit=0
```

`discover_plugins` read the installed distribution's entry point and registered C11; the registry then holds only `C11` (no built-ins were registered in this check).

## 3. Remove again

The sample is not part of the uv workspace (owner decision 2026-11-12), so an exact sync removes it. The repo's usual full sync command was used instead of plain `uv sync --locked`, so the optional extras stay installed.

```text
$ uv sync --locked --all-packages --all-extras
Resolved 84 packages in 24ms
Uninstalled 1 package in 3ms
 - rag-foundry-sample-plugin==0.1.0 (from file:///C:/Dev/import/Rag-lab/examples/plugins/rag_foundry_sample_plugin)
exit=0
$ uv run --no-sync python -c "import importlib.util; print(importlib.util.find_spec(\"rag_foundry_sample_plugin\"))"
None
```

`git status` was clean after these steps (apart from the not yet committed P7.1-05 spec).
