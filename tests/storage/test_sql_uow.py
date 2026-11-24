import asyncio

import pytest
from rag_foundry_domain.errors import StoreError
from rag_foundry_storage.sql.uow import SqlUnitOfWork
from sqlalchemy.exc import OperationalError


def test_commit_operational_error_maps_without_database_text():
    asyncio.run(_test_commit_operational_error_maps_without_database_text())


async def _test_commit_operational_error_maps_without_database_text():
    class Session:
        info = {}

        async def commit(self):
            raise OperationalError("statement", {}, Exception("secret database text"))

        async def close(self):
            pass

        async def rollback(self):
            pass

    uow = SqlUnitOfWork(lambda: Session(), read_only=False)
    async with uow:
        with pytest.raises(StoreError, match="store_unavailable") as caught:
            await uow.commit()
        assert caught.value.__cause__ is None
        assert "secret" not in str(caught.value)


def test_rollback_failure_does_not_mask_original_exception():
    asyncio.run(_test_rollback_failure_does_not_mask_original_exception())


async def _test_rollback_failure_does_not_mask_original_exception():
    class Session:
        info = {}

        async def rollback(self):
            raise RuntimeError("rollback failed")

        async def close(self):
            raise RuntimeError("close failed")

    uow = SqlUnitOfWork(lambda: Session(), read_only=False)
    with pytest.raises(ValueError, match="original"):
        async with uow:
            raise ValueError("original")


def test_savepoint_rollback_failure_does_not_mask_original_exception():
    asyncio.run(_test_savepoint_rollback_failure_does_not_mask_original_exception())


async def _test_savepoint_rollback_failure_does_not_mask_original_exception():
    from rag_foundry_storage.sql._tx import write

    class Savepoint:
        async def rollback(self):
            raise RuntimeError("savepoint rollback failed")

    class Session:
        info = {}

        async def begin_nested(self):
            return Savepoint()

    original = ValueError("original")
    with pytest.raises(ValueError) as caught:
        async with write(Session()):
            raise original
    assert caught.value is original
