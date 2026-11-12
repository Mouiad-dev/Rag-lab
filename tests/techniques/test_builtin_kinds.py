from pathlib import Path

from rag_foundry_chunking import register_builtin_plugins as register_chunking
from rag_foundry_indexing import register_builtin_plugins as register_indexing
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

CATALOGUE = Path(__file__).resolve().parents[2] / "config" / "techniques.json"


def test_builtin_plugins_fit_their_category_kinds():
    reg = TechniqueRegistry(load_catalogue(CATALOGUE))
    register_chunking(reg)
    register_indexing(reg)
    ids = set(reg.registered_ids())
    assert len(ids) == 48  # every built-in plugin, so a quietly dropped one fails here
    assert {"R07", "R08", "R09", "R10", "R28"} <= ids  # kinds that differ from their category's main kind
