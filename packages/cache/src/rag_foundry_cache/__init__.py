"""Response caches (R50 exact, R51 semantic) and the prefix-cache probe (R49). See ADR-0021."""

from rag_foundry_cache.exact import CachedRun, CacheTrace, ExactCacheConfig, ExactResponseCache
from rag_foundry_cache.keys import (
    GenerationParams,
    ResponseKeyInputs,
    exact_key,
    partition_key,
    source_version_ids,
)
from rag_foundry_cache.prefix import (
    ArmSummary,
    PrefixCacheProbe,
    PrefixProbeConfig,
    PrefixProbeReport,
    ProbeCallRecord,
)
from rag_foundry_cache.prefix import register_builtin_plugins as _register_prefix
from rag_foundry_cache.revalidate import AllowAllSourceGate, SourceGate, revalidate


def register_builtin_plugins(registry) -> None:
    _register_prefix(registry)
    registry.register(ExactResponseCache)


__all__ = [
    "AllowAllSourceGate",
    "ArmSummary",
    "CacheTrace",
    "CachedRun",
    "ExactCacheConfig",
    "ExactResponseCache",
    "SourceGate",
    "GenerationParams",
    "PrefixCacheProbe",
    "PrefixProbeConfig",
    "PrefixProbeReport",
    "ProbeCallRecord",
    "ResponseKeyInputs",
    "exact_key",
    "partition_key",
    "register_builtin_plugins",
    "revalidate",
    "source_version_ids",
]
