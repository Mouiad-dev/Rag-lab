"""Static source reader for deterministic expansion tests."""

from rag_foundry_domain.errors import NotFoundError


class StaticSourceReader:
    def __init__(self, sources):
        self.sources = sources

    async def view_source(self, scope, view_id):
        key = (scope.tenant_id, scope.project_id, view_id)
        if key not in self.sources:
            raise NotFoundError()
        return self.sources[key]
