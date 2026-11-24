# packages/techniques

Technique plugin package (see docs/decisions/ADR-0002-technique-plugin-system.md).

v0 API:
- `load_catalogue(path)` reads `config/techniques.json` (schema v2) into frozen typed entries.
  The loader never writes to it; the file is the canonical catalogue.
- `PluginManifest` (`technique_id`, `kind`, `version`, `fidelity`, `required_capabilities`,
  `config_model`), `PluginKind` (the 12 ADR-0002 kinds), `Fidelity`, `TechniquePlugin` protocol
  (`PROTOCOL_VERSION = "0"`). Kind-specific plugin methods are deferred.
- `TechniqueRegistry(catalogue)`: `register`, `entry`, `plugin`, `validate_config`,
  `registered_ids`. No plugin is registered by default; entry-point discovery arrives in P7.1.

Typed errors (all subclass `TechniqueError`; none subclass `KeyError` or `LookupError`):
- `UnknownTechniqueError`: the ID is not in the catalogue.
- `TechniqueUnavailableError`: the ID is catalogued but has no registered plugin.
- `DuplicatePluginError`: a second plugin for an already registered ID.
- `CatalogueError`: the catalogue file is unreadable or invalid.

There is no silent fallback. Plugins are trusted in-process code (ADR-0002); untrusted code
execution belongs in the sandbox, not here.
