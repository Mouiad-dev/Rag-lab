import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing import (
    DedupConfig,
    EvidenceDedup,
    check_single_input,
    register_builtin_plugins,
)
from rag_foundry_techniques import RetrievalResult, RetrievedChunk, ScoreComponent


def hit(cid, rank, text, *, view=None, start=0, end=100):
    return RetrievedChunk(
        chunk_id=cid,
        extraction_view_id=view or f"view-{cid}",
        start=start,
        end=end,
        block_ids=("b",),
        text=text,
        score=1.0 / rank,
        rank=rank,
        score_components=(ScoreComponent(name="original", value=2.0),),
    )


def result(hits, *, grounding="corpus", version="v1"):
    return RetrievalResult(technique_id="R03", grounding=grounding, index_version_id=version, hits=tuple(hits))


def test_exact_text_drops_across_views_and_keeps_provenance():
    selected = EvidenceDedup(DedupConfig()).select(
        result([hit("a", 1, "same passage"), hit("b", 2, "other"), hit("c", 3, "same passage")])
    )
    assert [hit.chunk_id for hit in selected.result.hits] == ["a", "b"]
    dropped = selected.dropped[0]
    assert (dropped.chunk_id, dropped.extraction_view_id, dropped.start, dropped.end) == ("c", "view-c", 0, 100)
    assert (dropped.input_rank, dropped.kept_chunk_id, dropped.rule, dropped.similarity) == (3, "a", "exact_text", 1.0)
    component = next(c for c in selected.result.hits[0].score_components if c.name == "dedup_exact_text")
    assert component.source_rank == 3
    assert [h.rank for h in selected.result.hits] == [1, 2]


def test_span_overlap_boundaries_and_views():
    cfg = DedupConfig(rules=("span_overlap",))
    selected = EvidenceDedup(cfg).select(
        result(
            [
                hit("a", 1, "a", view="shared", start=0, end=100),
                hit("b", 2, "b", view="shared", start=50, end=150),
                hit("c", 3, "c", view="shared", start=80, end=180),
                hit("d", 4, "d", view="another", start=50, end=150),
            ]
        )
    )
    assert [(d.chunk_id, d.similarity) for d in selected.dropped] == [("b", 0.5)]
    assert [h.chunk_id for h in selected.result.hits] == ["a", "c", "d"]


def test_near_text_hand_calculation():
    first = "the tidal observatory records sea levels every hour along the northern harbour wall"
    second = "the tidal observatory records sea levels every hour along the northern harbour pier"
    selected = EvidenceDedup(DedupConfig(rules=("near_text",))).select(
        result([hit("a", 1, first), hit("b", 2, second)])
    )
    assert len(selected.dropped) == 1
    assert selected.dropped[0].similarity == pytest.approx(10 / 12, abs=1e-9)


def test_conflicting_numbers_are_protected_by_default():
    first = "the unit price of the blue box is 10 dollars in the spring catalogue"
    second = "the unit price of the blue box is 12 dollars in the spring catalogue"
    hits = [hit("a", 1, first), hit("b", 2, second)]
    protected = EvidenceDedup(DedupConfig(rules=("near_text",), near_threshold=0.55)).select(result(hits))
    assert [h.chunk_id for h in protected.result.hits] == ["a", "b"]
    unprotected = EvidenceDedup(DedupConfig(rules=("near_text",), near_threshold=0.55, protect_numbers=False)).select(
        result(hits)
    )
    assert unprotected.dropped[0].similarity == pytest.approx(9 / 15)


def test_swapped_numbers_are_not_merged():
    first = "the blue box costs 10 dollars and the red box costs 12 dollars in the catalogue"
    second = "the blue box costs 12 dollars and the red box costs 10 dollars in the catalogue"
    hits = [hit("a", 1, first), hit("b", 2, second)]
    protected = EvidenceDedup(DedupConfig(rules=("near_text",), near_threshold=0.5)).select(result(hits))
    assert [h.chunk_id for h in protected.result.hits] == ["a", "b"]
    unprotected = EvidenceDedup(DedupConfig(rules=("near_text",), near_threshold=0.5, protect_numbers=False)).select(
        result(hits)
    )
    assert [d.chunk_id for d in unprotected.dropped] == ["b"]


def test_k_cut_survivors_are_reported():
    hits = [
        hit("a", 1, "alpha facts about tides", view="v1", start=0, end=20),
        hit("b", 2, "beta facts about bridges", view="v2", start=0, end=20),
        hit("c", 3, "beta facts about bridges", view="v3", start=0, end=20),
    ]
    selected = EvidenceDedup(DedupConfig(k=1)).select(result(hits))
    assert [h.chunk_id for h in selected.result.hits] == ["a"]
    assert selected.truncated == ("b",)
    assert [(d.chunk_id, d.kept_chunk_id) for d in selected.dropped] == [("c", "b")]


def test_arabic_digit_folded_near_match_is_not_exact():
    first = "\u0633\u0639\u0631 \u0627\u0644\u0642\u0637\u0639\u0629 \u0661\u0660"  # Arabic-Indic 10
    second = "\u0633\u0639\u0631 \u0627\u0644\u0642\u0637\u0639\u0629 10"
    selected = EvidenceDedup(DedupConfig(rules=("exact_text", "near_text"))).select(
        result([hit("a", 1, first), hit("b", 2, second)])
    )
    assert selected.dropped[0].rule == "near_text"
    assert selected.dropped[0].similarity == 1.0


@pytest.mark.parametrize(
    "value",
    [
        result([hit("a", 1, "x"), hit("a", 2, "y")]),
        result([hit("a", 2, "x")]),
        result([], grounding="none", version=None),
    ],
)
def test_invalid_input_error_code(value):
    with pytest.raises(IndexingError) as exc:
        check_single_input(value)
    assert exc.value.code == "selection_invalid_input"


def test_selection_version_error_code_is_registered():
    assert IndexingError("selection_version_mismatch").code == "selection_version_mismatch"


def test_survivor_accounting_and_output_metadata():
    incoming = result([hit("a", 1, "same"), hit("b", 2, "same"), hit("c", 3, "unique")])
    selected = EvidenceDedup(DedupConfig(rules=("exact_text",))).select(incoming)
    seen = [h.chunk_id for h in selected.result.hits] + [d.chunk_id for d in selected.dropped]
    assert sorted(seen) == ["a", "b", "c"]
    assert len(seen) == len(set(seen))
    assert [h.rank for h in selected.result.hits] == list(range(1, len(selected.result.hits) + 1))
    assert selected.result.technique_id == "R10"
    assert selected.result.index_version_id == incoming.index_version_id
    assert [h.score for h in selected.result.hits] == [1.0, 1.0 / 3]
    assert selected.result.scoring == "dedup[R03]:exact_text"


def test_plugin_registry_config_validation(tmp_path):
    from pathlib import Path

    from rag_foundry_techniques import TechniqueRegistry, load_catalogue

    catalogue = load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json")
    registry = TechniqueRegistry(catalogue)
    register_builtin_plugins(registry)
    assert "R10" in registry.registered_ids()
    assert registry.validate_config("R10", {}) == DedupConfig()
    for config in ({"rules": []}, {"rules": ["semantic"]}, {"near_threshold": 0.3}):
        with pytest.raises(ValidationError):
            registry.validate_config("R10", config)
