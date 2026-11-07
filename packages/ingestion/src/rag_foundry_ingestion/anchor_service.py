"""Scoped anchor creation and bounded resolution over stored extraction views."""

from collections import defaultdict
from collections.abc import Sequence

from rag_foundry_contracts.resolve import AnchorResolution, ResolutionReason, ResolutionStatus
from rag_foundry_contracts.source import SourceAnchor, SourceRef
from rag_foundry_domain import ObjectStore, Scope
from rag_foundry_domain.errors import AnchorError, ExtractionError, NotFoundError, ObjectStoreError
from rag_foundry_domain.uow import UnitOfWorkFactory

from rag_foundry_ingestion.anchors import anchors_for_span as make_anchors_for_span
from rag_foundry_ingestion.anchors import make_anchor, resolve_anchor
from rag_foundry_ingestion.extraction import read_payload


class AnchorService:
    def __init__(
        self,
        store: ObjectStore,
        uow: UnitOfWorkFactory,
        *,
        provider_namespace: str,
        context: int = 32,
        max_views: int = 10,
    ) -> None:
        self._store = store
        self._uow = uow
        self._provider_namespace = provider_namespace
        self._context = context
        self._max_views = max_views

    async def anchor(self, scope: Scope, view_id: str, block_id: str, start: int, end: int) -> SourceAnchor:
        async with self._uow(read_only=True) as uow:
            view = await uow.extractions.get_view(scope, view_id)
        payload = await read_payload(self._store, scope, view)
        ref = SourceRef(
            provider_namespace=self._provider_namespace,
            document_id=view.document_id,
            document_version_id=view.document_version_id,
            extraction_view_id=view.id,
        )
        return make_anchor(payload, ref, block_id, start, end, context=self._context)

    async def anchors_for_span(self, scope: Scope, view_id: str, start: int, end: int) -> tuple[SourceAnchor, ...]:
        async with self._uow(read_only=True) as uow:
            view = await uow.extractions.get_view(scope, view_id)
        payload = await read_payload(self._store, scope, view)
        ref = SourceRef(
            provider_namespace=self._provider_namespace,
            document_id=view.document_id,
            document_version_id=view.document_version_id,
            extraction_view_id=view.id,
        )
        return make_anchors_for_span(payload, ref, start, end, context=self._context)

    async def resolve(self, scope: Scope, anchors: Sequence[SourceAnchor]) -> tuple[AnchorResolution, ...]:
        results: list[AnchorResolution | None] = [None] * len(anchors)
        grouped: dict[str, list[tuple[int, SourceAnchor]]] = defaultdict(list)
        for index, anchor in enumerate(anchors):
            if anchor.source.provider_namespace != self._provider_namespace:
                results[index] = AnchorResolution(
                    anchor_id=anchor.anchor_id,
                    status=ResolutionStatus.UNAVAILABLE,
                    reason=ResolutionReason.FOREIGN_NAMESPACE,
                )
            else:
                grouped[anchor.source.extraction_view_id].append((index, anchor))

        if len(grouped) > self._max_views:
            raise AnchorError("too_many_views")

        views = {}
        if grouped:
            async with self._uow(read_only=True) as uow:
                for view_id in grouped:
                    try:
                        views[view_id] = await uow.extractions.get_view(scope, view_id)
                    except NotFoundError:
                        views[view_id] = None

        for view_id, items in grouped.items():
            view = views[view_id]
            if view is None:
                for index, anchor in items:
                    results[index] = AnchorResolution(
                        anchor_id=anchor.anchor_id,
                        status=ResolutionStatus.UNAVAILABLE,
                        reason=ResolutionReason.NOT_FOUND,
                    )
                continue
            try:
                payload = await read_payload(self._store, scope, view)
            except (ExtractionError, ObjectStoreError):
                for index, anchor in items:
                    results[index] = AnchorResolution(
                        anchor_id=anchor.anchor_id,
                        status=ResolutionStatus.UNAVAILABLE,
                        reason=ResolutionReason.INTEGRITY_ERROR,
                    )
                continue
            ref = SourceRef(
                provider_namespace=self._provider_namespace,
                document_id=view.document_id,
                document_version_id=view.document_version_id,
                extraction_view_id=view.id,
            )
            for index, anchor in items:
                results[index] = resolve_anchor(payload, ref, anchor)

        assert all(result is not None for result in results)
        return tuple(results)
