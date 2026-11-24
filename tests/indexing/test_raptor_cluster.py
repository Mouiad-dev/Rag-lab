import random

import pytest
from pydantic import ValidationError
from rag_foundry_indexing import ClusterConfig, ClusterItem, cluster_nodes

_AXES = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))


def _vector(group, index):
    base = list(_AXES[group])
    base[3] = 0.01 * (index + 1)
    return tuple(base)


def _items(ids=None, groups=3, per_group=4):
    items = []
    for number in range(groups * per_group):
        group, index = divmod(number, per_group)
        # interleave groups in key order so grouping is not just contiguous keys
        key = (index * groups + group, index * groups + group + 1)
        node_id = ids[number] if ids else f"n{number:02d}"
        items.append(ClusterItem(key=key, node_id=node_id, embedding=_vector(group, index)))
    return items


def _config(**overrides):
    return ClusterConfig(**{"target_cluster_size": 4, "max_cluster_size": 6, **overrides})


def _key_clusters(items, result):
    by_id = {item.node_id: item.key for item in items}
    return tuple(tuple(by_id[node_id] for node_id in cluster) for cluster in result.clusters)


def test_three_separated_groups_give_three_clusters():
    items = _items()
    result = cluster_nodes(items, _config())
    assert len(result.clusters) == 3
    expected = sorted(tuple(f"n{g * 4 + i:02d}" for i in range(4)) for g in range(3))
    assert sorted(tuple(sorted(c)) for c in result.clusters) == expected
    assert sorted(i for c in result.clusters for i in c) == sorted(item.node_id for item in items)


def test_shuffled_input_gives_identical_result():
    items = _items()
    baseline = cluster_nodes(items, _config())
    for seed in range(5):
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        assert cluster_nodes(shuffled, _config()) == baseline


def test_random_node_ids_do_not_change_clusters_by_key():
    base = _items()
    reference = _key_clusters(base, cluster_nodes(base, _config()))
    for seed in range(5):
        ids = [f"id-{random.Random(seed * 100 + i).random():.12f}-{i}" for i in range(12)]
        random.Random(seed).shuffle(ids)
        other = _items(ids=ids)
        assert _key_clusters(other, cluster_nodes(other, _config())) == reference


def test_node_id_order_does_not_override_key_order():
    # ids sort opposite to keys; members must follow key order
    ids = [f"z{99 - n:02d}" for n in range(6)]
    items = [ClusterItem(key=(n,), node_id=ids[n], embedding=(1.0, 0.0, 0.0, 0.0)) for n in range(6)]
    result = cluster_nodes(items, ClusterConfig(target_cluster_size=6, max_cluster_size=6))
    assert result.clusters == (tuple(ids),)


def test_max_cluster_size_splits_group_of_six():
    items = [ClusterItem(key=(n,), node_id=f"a{n}", embedding=_vector(0, n)) for n in range(6)] + [
        ClusterItem(key=(10 + n,), node_id=f"b{n}", embedding=_vector(1, n)) for n in range(2)
    ]
    result = cluster_nodes(items, ClusterConfig(target_cluster_size=3, max_cluster_size=3))
    assert all(len(c) <= 3 for c in result.clusters)
    assert sorted(i for c in result.clusters for i in c) == sorted(item.node_id for item in items)


def test_identical_vectors_split_by_sorted_order():
    items = [ClusterItem(key=(n,), node_id=f"r{5 - n}", embedding=(1.0, 1.0)) for n in range(6)]
    result = cluster_nodes(items, ClusterConfig(target_cluster_size=3, max_cluster_size=3))
    assert result.clusters == (("r5", "r4", "r3"), ("r2", "r1", "r0"))


def test_single_item_and_zero_vector():
    one = cluster_nodes([ClusterItem(key=(0,), node_id="a", embedding=(0.0, 0.0))], ClusterConfig())
    assert one.clusters == (("a",),)
    assert one.converged
    items = _items() + [ClusterItem(key=(99,), node_id="zero", embedding=(0.0, 0.0, 0.0, 0.0))]
    result = cluster_nodes(items, _config())
    assert sorted(i for c in result.clusters for i in c) == sorted(item.node_id for item in items)


def test_all_identical_vectors_form_valid_partition():
    items = [ClusterItem(key=(n,), node_id=f"i{n}", embedding=(1.0, 2.0)) for n in range(12)]
    result = cluster_nodes(items, ClusterConfig(target_cluster_size=4, max_cluster_size=5))
    assert all(len(c) <= 5 for c in result.clusters)
    assert sorted(i for c in result.clusters for i in c) == sorted(item.node_id for item in items)
    assert cluster_nodes(list(reversed(items)), ClusterConfig(target_cluster_size=4, max_cluster_size=5)) == result


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="at least one"):
        cluster_nodes([], ClusterConfig())
    duplicate = [
        ClusterItem(key=(0,), node_id="a", embedding=(1.0,)),
        ClusterItem(key=(1,), node_id="a", embedding=(1.0,)),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        cluster_nodes(duplicate, ClusterConfig())
    mixed = [
        ClusterItem(key=(0,), node_id="a", embedding=(1.0,)),
        ClusterItem(key=(1,), node_id="b", embedding=(1.0, 0.0)),
    ]
    with pytest.raises(ValueError, match="dimension"):
        cluster_nodes(mixed, ClusterConfig())


def test_config_validation():
    with pytest.raises(ValidationError):
        ClusterConfig(target_cluster_size=5, max_cluster_size=4)
    with pytest.raises(ValidationError):
        ClusterConfig(target_cluster_size=1)
    with pytest.raises(ValidationError):
        ClusterConfig(unknown=1)


def test_converged_and_max_iterations():
    result = cluster_nodes(_items(), _config())
    assert result.converged
    assert 1 <= result.iterations <= 20
    limited = cluster_nodes(_items(), _config(max_iterations=1))
    assert limited.iterations == 1
    assert not limited.converged
    assert sorted(i for c in limited.clusters for i in c) == sorted(item.node_id for item in _items())
    assert all(len(c) <= 6 for c in limited.clusters)


def test_farthest_first_tie_picks_first_in_sorted_order():
    vectors = {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (0.0, -1.0), "d": (1.0, 0.01)}
    items = [ClusterItem(key=(n,), node_id=name, embedding=vec) for n, (name, vec) in enumerate(vectors.items())]
    result = cluster_nodes(items, ClusterConfig(target_cluster_size=2, max_cluster_size=3))
    assert result.clusters == (("a", "c", "d"), ("b",))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_embedding_raises(bad):
    with pytest.raises(ValueError, match="non-finite"):
        cluster_nodes([ClusterItem(key=(0,), node_id="a", embedding=(bad, 1.0))], ClusterConfig())
