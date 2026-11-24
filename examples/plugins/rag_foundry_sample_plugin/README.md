# rag-foundry-sample-plugin

A sample external technique plugin for RAG Foundry. It registers catalogue ID C11
through the `rag_foundry.techniques` entry-point group and chunks text by page blocks.

This is a sample of the plugin mechanism. It is NOT an implementation of C11: the
catalogue keeps C11 as `planned`, and the manifest fidelity is `simulation`.

Warning: plugins are trusted, in-process code (ADR-0002). Install only code you trust.

Install into your environment (outside the uv workspace):

    uv pip install -e examples/plugins/rag_foundry_sample_plugin

Then call `rag_foundry_techniques.discover_plugins(registry)` explicitly; nothing is
discovered at import time. Check it with `check_plugin_contract` and `check_chunker_run`.
