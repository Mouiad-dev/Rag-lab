"""Seed fixture dataset built from the P5.2 bilingual paired fixture."""

from rag_foundry_evaluation import EvalDataset

from tests.evaluation._corpus import build_dataset

ROWS = (
    ("en-fact", "en", (), "dev"),
    ("en-variant", "en", (), "dev"),
    ("paraphrase", "en", ("paraphrase",), "dev"),
    ("id-exact", "en", ("identifier",), "dev"),
    ("en-club", "en", (), "dev"),
    ("ar-fact", "ar", (), "validation"),
    ("mixed", "mixed", (), "validation"),
    ("unanswerable", "en", (), "validation"),
    ("id-ar", "ar", ("identifier",), "held_out"),
    ("ar-diacritic", "ar", ("diacritics",), "held_out"),
)


def seed_dataset() -> EvalDataset:
    base = build_dataset(ROWS)
    data = base.model_dump(mode="json")
    data.update(
        dataset_id="seed-fixture",
        dataset_version="1",
        description="Tiny offline fixture, not a benchmark. Built from the P5.2 bilingual paired fixture.",
    )
    return EvalDataset.model_validate(data)
