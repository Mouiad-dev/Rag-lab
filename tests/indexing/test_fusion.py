# ruff: noqa: E501

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing import RrfConfig, RrfFusion, WeightedConfig, WeightedFusion
from rag_foundry_techniques import RetrievalResult, RetrievedChunk, ScoreComponent


def hit(cid, rank, score=1.0, components=()):
    return RetrievedChunk(
        chunk_id=cid,
        extraction_view_id="view",
        start=0,
        end=1,
        block_ids=("b",),
        text=cid,
        score=score,
        rank=rank,
        score_components=components,
    )


def result(tid, hits):
    return RetrievalResult(technique_id=tid, grounding="corpus", index_version_id="v1", hits=tuple(hits))


def test_rrf_formula_provenance_and_candidate_union():
    a = result(
        "R03", [hit("x", 1), hit("y", 2, components=(ScoreComponent(name="bm25_term", value=2.0),)), hit("z", 3)]
    )
    b = result("R02", [hit("y", 1), hit("w", 2)])
    fused = RrfFusion(RrfConfig()).fuse([a, b])
    assert [h.chunk_id for h in fused.hits] == ["y", "x", "w", "z"]
    y = fused.hits[0]
    assert y.score == pytest.approx(1 / 62 + 1 / 61)
    contributions = [c for c in y.score_components if c.name == "rrf_contribution"]
    assert len(contributions) == 2
    assert [(c.source, c.source_rank) for c in contributions] == [("R03", 2), ("R02", 1)]
    original = next(c for c in y.score_components if c.name == "bm25_term")
    assert (original.source, original.source_rank) == ("R03", 2)
    assert {h.chunk_id for h in fused.hits} <= {h.chunk_id for r in (a, b) for h in r.hits}
    assert_contributions_sum(fused, "rrf_contribution")


def test_rrf_tie_uses_chunk_id():
    fused = RrfFusion(RrfConfig()).fuse(
        [result("R03", [hit("x", 1), hit("y", 2)]), result("R02", [hit("y", 1), hit("x", 2)])]
    )
    assert [h.chunk_id for h in fused.hits] == ["x", "y"]


def test_weighted_hand_calculation_and_degenerate_lists():
    a = result("R03", [hit("x", 1, 8), hit("y", 2, 4), hit("z", 3, 2)])
    b = result("R02", [hit("y", 1, 0.9), hit("w", 2, 0.5)])
    fused = WeightedFusion(WeightedConfig()).fuse([a, b])
    assert [h.chunk_id for h in fused.hits] == ["y", "x", "w", "z"]
    assert fused.hits[0].score == pytest.approx(2 / 3)
    assert fused.hits[1].score == pytest.approx(0.5)
    assert [(h.chunk_id, h.score) for h in fused.hits[2:]] == [("w", 0.0), ("z", 0.0)]
    single = WeightedFusion(WeightedConfig()).fuse([result("R03", [hit("a", 1, 9)]), result("R02", [hit("b", 1, 0)])])
    assert next(c for c in single.hits[0].score_components if c.name == "normalization_degenerate").value == 1.0
    equal = WeightedFusion(WeightedConfig()).fuse([result("R03", [hit("c", 1, 4), hit("d", 2, 4)]), result("R02", [])])
    assert all(next(c for c in h.score_components if c.name == "min_max_score").value == 1.0 for h in equal.hits)
    for h in equal.hits:
        flag = next(c for c in h.score_components if c.name == "normalization_degenerate")
        assert flag.source == "R03"
    assert_contributions_sum(fused, "weighted_contribution")


def test_configured_weights_are_used_as_given():
    a = result("R03", [hit("x", 1, 8), hit("y", 2, 4), hit("z", 3, 2)])
    b = result("R02", [hit("y", 1, 0.9), hit("w", 2, 0.5)])
    fused = WeightedFusion(WeightedConfig(weights={"R03": 2, "R02": 1})).fuse([a, b])
    by_id = {h.chunk_id: h for h in fused.hits}
    assert by_id["x"].score == pytest.approx(2.0)
    assert by_id["y"].score == pytest.approx(2 * (1 / 3) + 1.0)
    assert_contributions_sum(fused, "weighted_contribution")


def assert_contributions_sum(fused, name):
    for h in fused.hits:
        assert sum(c.value for c in h.score_components if c.name == name) == pytest.approx(h.score)


@pytest.mark.parametrize(
    "case,code",
    [
        (lambda: RrfFusion(RrfConfig()).fuse([result("R03", [])]), "fusion_too_few_branches"),
        (lambda: RrfFusion(RrfConfig()).fuse([result("R03", []), result("R03", [])]), "fusion_duplicate_branch"),
        (
            lambda: RrfFusion(RrfConfig()).fuse(
                [
                    result("R03", []),
                    RetrievalResult(technique_id="R02", grounding="corpus", index_version_id="v2", hits=()),
                ]
            ),
            "fusion_version_mismatch",
        ),
        (
            lambda: RrfFusion(RrfConfig()).fuse(
                [
                    result("R03", []),
                    RetrievalResult(technique_id="R02", grounding="corpus", index_version_id=None, hits=()),
                ]
            ),
            "fusion_version_mismatch",
        ),
        (
            lambda: RrfFusion(RrfConfig()).fuse(
                [
                    result("R03", []),
                    RetrievalResult(technique_id="R01", grounding="none", index_version_id=None, hits=()),
                ]
            ),
            "fusion_version_mismatch",
        ),
        (
            lambda: RrfFusion(RrfConfig()).fuse([result("R03", [hit("a", 2)]), result("R02", [])]),
            "fusion_invalid_ranks",
        ),
        (
            lambda: RrfFusion(RrfConfig()).fuse([result("R03", [hit("a", 1), hit("a", 2)]), result("R02", [])]),
            "fusion_duplicate_candidate",
        ),
        (
            lambda: RrfFusion(RrfConfig()).fuse(
                [
                    result("R03", [hit("a", 1)]),
                    result(
                        "R02",
                        [
                            RetrievedChunk(
                                chunk_id="a",
                                extraction_view_id="other",
                                start=0,
                                end=1,
                                block_ids=("b",),
                                text="a",
                                score=1,
                                rank=1,
                            )
                        ],
                    ),
                ]
            ),
            "fusion_candidate_conflict",
        ),
        (
            lambda: WeightedFusion(WeightedConfig(weights={"R03": 1})).fuse([result("R03", []), result("R02", [])]),
            "fusion_weights_mismatch",
        ),
    ],
)
def test_each_fusion_error_code(case, code):
    with pytest.raises(IndexingError) as exc:
        case()
    assert exc.value.code == code
    assert IndexingError(code).code == code


def test_plugin_registry_config_validation():
    from pathlib import Path

    from rag_foundry_indexing import register_builtin_plugins
    from rag_foundry_techniques import TechniqueRegistry, load_catalogue

    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    assert {"R07", "R08"} <= set(registry.registered_ids())
    assert registry.validate_config("R07", {}) == RrfConfig()
    assert registry.validate_config("R08", {}) == WeightedConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("R08", {"bogus": 1})
