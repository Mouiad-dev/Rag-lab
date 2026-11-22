import uuid

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.graphs import GraphSpan
from rag_foundry_domain.summary_tree import SummaryTree, TreeNode, node_id_for, tree_key_for
from rag_foundry_domain.testing import make_scope

NEW_CODES = (
    "tree_scope_mismatch",
    "tree_invalid",
    "tree_exists",
    "tree_too_large",
    "tree_source_mismatch",
    "raptor_summary_invalid",
    "kag_schema_invalid",
    "kag_scope_mismatch",
    "kag_invalid",
    "kag_exists",
    "kag_too_large",
    "kag_source_mismatch",
)
VERSION = str(uuid.uuid4())
SCOPE = make_scope()
VIEW = str(uuid.uuid4())


def span(quote, start=0, view=VIEW):
    return GraphSpan(
        chunk_id=str(uuid.uuid4()), extraction_view_id=view, start=start, end=start + len(quote), quote=quote
    )


def leaf(quote, start=0, emb=(0.1, 0.2)):
    s = span(quote, start)
    return TreeNode(
        node_id=node_id_for(0, s.chunk_id), level=0, kind="leaf", text=quote, span=s, embedding=emb, synthetic=False
    )


def summary(children, level=1, text="sum", emb=(0.3, 0.4)):
    ids = [c.node_id if isinstance(c, TreeNode) else c for c in children]
    return TreeNode(
        node_id=node_id_for(level, ids),
        level=level,
        kind="summary",
        text=text,
        children=tuple(ids),
        embedding=emb,
        synthetic=True,
    )


def build(nodes, **over):
    args = {
        "schema": "rag_foundry.summary_tree/1",
        "tree_key": tree_key_for({"a": 1}),
        "tenant_id": SCOPE.tenant_id,
        "project_id": SCOPE.project_id,
        "index_version_id": VERSION,
        "embedding_key": "emb:x",
        "generator_json": "{}",
        "nodes": tuple(sorted(nodes, key=lambda n: (n.level, n.node_id))),
    }
    args.update(over)
    return SummaryTree(**args)


def two_level():
    leaves = [leaf("aa", 10), leaf("bb", 0), leaf("cc", 5)]
    mid1, mid2 = summary(leaves[:2], 1, "m1"), summary(leaves[2:], 1, "m2")
    root = summary([mid1, mid2], 2, "root")
    return leaves, build([*leaves, mid1, mid2, root]), root


@pytest.mark.parametrize("code", NEW_CODES)
def test_new_codes_registered(code):
    assert IndexingError(code).code == code


def test_ids_stable_and_order_insensitive():
    a, b = "rnd_" + "0" * 16, "rnd_" + "1" * 16
    assert node_id_for(1, [a, b]) == node_id_for(1, (b, a))
    assert node_id_for(1, [a, b]) != node_id_for(2, [a, b])
    assert node_id_for(0, "x").startswith("rnd_") and len(node_id_for(0, "x")) == 20
    assert len(tree_key_for({"a": 1})) == 64


def test_round_trip_lineage_and_helpers():
    leaves, tree, root = two_level()
    assert SummaryTree.from_json(tree.to_json()) == tree
    assert tree.roots() == (root.node_id,)
    assert tree.children_of(root.node_id) == root.children
    expected = sorted((n.span for n in leaves), key=lambda s: (s.extraction_view_id, s.start, s.end))
    assert tree.lineage(root.node_id) == tuple(expected)
    assert tree.leaf_ids(root.node_id) == tuple(sorted(n.node_id for n in leaves))
    assert [len(lv) for lv in tree.levels()] == [3, 2, 1]


def test_missing_child_rejected():
    l1 = leaf("aa")
    with pytest.raises(ValidationError):
        build([l1, summary([l1, "rnd_" + "f" * 16])])


def test_two_parents_rejected():
    l1, l2 = leaf("aa"), leaf("bb")
    with pytest.raises(ValidationError):
        build([l1, l2, summary([l1, l2], 1, "x"), summary([l1], 1, "y")])


def test_child_at_same_level_rejected():
    l1 = leaf("aa")
    s1 = summary([l1], 1, "x")
    with pytest.raises(ValidationError):
        build([l1, s1, summary([s1], 1, "y")])


def test_leaf_text_must_equal_quote():
    s = span("abc")
    with pytest.raises(ValidationError):
        TreeNode(node_id=node_id_for(0, s.chunk_id), level=0, kind="leaf", text="abx", span=s, synthetic=False)


def test_summary_text_cap():
    l1 = leaf("aa")
    assert summary([l1], text="x" * 4000)
    with pytest.raises(ValidationError):
        summary([l1], text="x" * 4001)
    with pytest.raises(ValidationError):
        summary([l1], text="")


def test_mixed_embedding_dimensions_rejected():
    l1, l2 = leaf("aa", emb=(0.1,)), leaf("bb", emb=(0.1, 0.2))
    with pytest.raises(ValidationError):
        build([l1, l2, summary([l1, l2], emb=(0.1,))])


def test_scope_checks():
    _, tree, _ = two_level()
    tree.check_scope(SCOPE, VERSION)
    with pytest.raises(IndexingError) as exc:
        tree.check_scope(make_scope(), VERSION)
    assert exc.value.code == "tree_scope_mismatch"
    with pytest.raises(IndexingError) as exc:
        tree.check_scope(SCOPE, str(uuid.uuid4()))
    assert exc.value.code == "tree_scope_mismatch"
    with pytest.raises(IndexingError) as exc:
        SummaryTree.from_json(tree.to_json().replace("summary_tree/1", "summary_tree/9"))
    assert exc.value.code == "tree_invalid"
    with pytest.raises(IndexingError) as exc:
        SummaryTree.from_json("{not json")
    assert exc.value.code == "tree_invalid"


def test_arabic_leaf_round_trip():
    quote = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
    l1 = leaf(quote)
    tree = build([l1, summary([l1], text="\u0645\u0644\u062e\u0635")])
    again = SummaryTree.from_json(tree.to_json())
    assert again == tree
    assert again.lineage(again.roots()[0])[0].quote == quote


def test_long_leaf_accepted():
    assert leaf("y" * 3000).text == "y" * 3000
