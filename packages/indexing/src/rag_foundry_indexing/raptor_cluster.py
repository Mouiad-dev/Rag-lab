"""Deterministic clustering for one RAPTOR level (catalogue R37).

This is an adaptation, not the paper's method. The RAPTOR paper reduces
embeddings with UMAP and fits a Gaussian mixture model whose component count is
chosen by BIC, with soft membership so a node may join several clusters. This
module uses none of that: no UMAP, no GMM, no BIC and no soft overlap. It runs
plain cosine k-means with a farthest-first start and hard assignment, so every
node belongs to exactly one cluster.

Determinism: items are sorted by ``(key, node_id)`` before any work. The caller
supplies ``key`` (for example the first lineage span), so random node IDs never
change the partition; ``node_id`` is only the last tie-break. Stdlib only.

``ClusterResult.iterations`` and ``converged`` describe only the top-level
k-means run, not the 2-means runs used inside oversized-cluster splits.
"""

from math import ceil, isfinite, sqrt

from pydantic import BaseModel, ConfigDict, Field, model_validator

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class ClusterConfig(BaseModel):
    model_config = _FROZEN

    target_cluster_size: int = Field(default=5, ge=2, le=20)
    max_cluster_size: int = Field(default=10, ge=2, le=50)
    max_iterations: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def _check_sizes(self) -> "ClusterConfig":
        if self.max_cluster_size < self.target_cluster_size:
            raise ValueError("max_cluster_size must be >= target_cluster_size")
        return self


class ClusterItem(BaseModel):
    model_config = _FROZEN

    key: tuple[int, ...]
    node_id: str
    embedding: tuple[float, ...]


class ClusterResult(BaseModel):
    model_config = _FROZEN

    clusters: tuple[tuple[str, ...], ...]
    iterations: int
    converged: bool


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    denominator = sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return 0.0 if denominator == 0 else sum(x * y for x, y in zip(left, right, strict=True)) / denominator


def _unit(vector: tuple[float, ...]) -> tuple[float, ...]:
    norm = sqrt(sum(x * x for x in vector))
    return vector if norm == 0 else tuple(x / norm for x in vector)


def _mean(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    units = [_unit(v) for v in vectors]
    return tuple(sum(column) / len(units) for column in zip(*units, strict=True))


def _farthest_first(vectors: list[tuple[float, ...]], k: int) -> list[tuple[float, ...]]:
    chosen = [0]
    chosen_set = {0}
    while len(chosen) < k:
        best_index, best_distance = -1, -1.0
        for index, vector in enumerate(vectors):
            if index in chosen_set:
                continue
            distance = min(1.0 - _cosine(vector, vectors[c]) for c in chosen)
            if distance > best_distance:  # strict: ties keep the first in sorted order
                best_index, best_distance = index, distance
        chosen.append(best_index)
        chosen_set.add(best_index)
    return [vectors[i] for i in chosen]


def _lloyd(vectors: list[tuple[float, ...]], k: int, max_iterations: int) -> tuple[list[int], int, bool]:
    centroids = _farthest_first(vectors, k)
    assignments: list[int] = [-1] * len(vectors)
    iterations = 0
    converged = False
    while iterations < max_iterations:
        iterations += 1
        updated = []
        for vector in vectors:
            best, best_similarity = 0, _cosine(vector, centroids[0])
            for index in range(1, k):
                similarity = _cosine(vector, centroids[index])
                if similarity > best_similarity:  # strict: ties go to the lowest index
                    best, best_similarity = index, similarity
            updated.append(best)
        if updated == assignments:
            converged = True
            break
        assignments = updated
        for index in range(k):
            members = [vectors[i] for i, a in enumerate(assignments) if a == index]
            if members:  # an empty cluster keeps its centroid and is dropped at the end
                centroids[index] = _mean(members)
    return assignments, iterations, converged


def _split(indices: list[int], vectors: list[tuple[float, ...]], config: ClusterConfig) -> list[list[int]]:
    if len(indices) <= config.max_cluster_size:
        return [indices]
    assignments, _, _ = _lloyd([vectors[i] for i in indices], 2, config.max_iterations)
    left = [i for i, a in zip(indices, assignments, strict=True) if a == 0]
    right = [i for i, a in zip(indices, assignments, strict=True) if a == 1]
    if not left or not right:  # e.g. identical vectors: split by sorted order
        half = (len(indices) + 1) // 2
        left, right = indices[:half], indices[half:]
    return _split(left, vectors, config) + _split(right, vectors, config)


def cluster_nodes(items: list[ClusterItem] | tuple[ClusterItem, ...], config: ClusterConfig) -> ClusterResult:
    if not items:
        raise ValueError("cluster_nodes requires at least one item")
    ids = [item.node_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node_id in cluster input")
    if len({len(item.embedding) for item in items}) != 1:
        raise ValueError("mixed embedding dimensions in cluster input")
    if not all(isfinite(x) for item in items for x in item.embedding):
        raise ValueError("non-finite embedding value in cluster input")
    ordered = sorted(items, key=lambda item: (item.key, item.node_id))
    vectors = [item.embedding for item in ordered]
    count = len(ordered)
    k = ceil(count / config.target_cluster_size)
    if k == 1:
        groups = [list(range(count))]
        iterations, converged = 0, True
    else:
        assignments, iterations, converged = _lloyd(vectors, k, config.max_iterations)
        groups = [[i for i, a in enumerate(assignments) if a == index] for index in range(k)]
        groups = [group for group in groups if group]
    final: list[list[int]] = []
    for group in groups:
        final.extend(_split(group, vectors, config))
    final.sort(key=lambda group: group[0])
    return ClusterResult(
        clusters=tuple(tuple(ordered[i].node_id for i in group) for group in final),
        iterations=iterations,
        converged=converged,
    )
