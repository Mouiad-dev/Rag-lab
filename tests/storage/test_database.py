import asyncio
import sys

import pytest
from pydantic import SecretStr
from rag_foundry_storage.sql.database import Database, create_database, open_database, run_db_main
from sqlalchemy.exc import SQLAlchemyError


def test_create_database_does_not_connect():
    password = "distinctive-password-731"
    db = create_database(SecretStr(f"postgresql+psycopg://user:{password}@127.0.0.1:1/db?connect_timeout=1"))
    try:
        assert db.engine.pool.checkedout() == 0
    finally:
        run_db_main(db.dispose())


def test_database_engine_options():
    db = create_database(SecretStr("postgresql+psycopg://user:p@127.0.0.1:1/db"))
    try:
        assert db.engine.sync_engine.hide_parameters is True
        assert db.sessionmaker.kw["expire_on_commit"] is False
        assert db.engine.pool._pre_ping is True  # SQLAlchemy exposes no public pre-ping option.
    finally:
        run_db_main(db.dispose())


def test_database_password_is_absent_from_repr_url_and_connection_error():
    password = "distinctive-password-731"
    db = create_database(SecretStr(f"postgresql+psycopg://user:{password}@127.0.0.1:1/db?connect_timeout=1"))
    try:
        assert password not in repr(db)
        assert password not in str(db.engine.url)

        async def connect():
            with pytest.raises(SQLAlchemyError) as caught:
                async with db.engine.connect():
                    pass
            return str(caught.value)

        assert password not in run_db_main(connect())
    finally:
        run_db_main(db.dispose())


def test_open_database_disposes_on_normal_exit(monkeypatch):
    disposed = []

    async def dispose(self):
        disposed.append(self)

    monkeypatch.setattr(Database, "dispose", dispose)

    async def exercise():
        async with open_database(SecretStr("postgresql+psycopg://u:p@127.0.0.1:1/db")) as db:
            assert db.engine
        return db

    db = run_db_main(exercise())
    assert disposed == [db]


def test_open_database_disposes_on_exception(monkeypatch):
    disposed = []

    async def dispose(self):
        disposed.append(self)

    monkeypatch.setattr(Database, "dispose", dispose)

    async def exercise():
        async with open_database(SecretStr("postgresql+psycopg://u:p@127.0.0.1:1/db")):
            raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        run_db_main(exercise())
    assert len(disposed) == 1


def test_run_db_main_returns_coroutine_result():
    async def value():
        return "result"

    assert run_db_main(value()) == "result"


def test_run_db_main_uses_selector_loop_on_windows(monkeypatch):
    observed = []

    class ObservingSelector(asyncio.SelectorEventLoop):
        def __init__(self):
            observed.append(self)
            super().__init__()

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(asyncio, "SelectorEventLoop", ObservingSelector)

    async def loop_type():
        return asyncio.get_running_loop()

    loop = run_db_main(loop_type())
    assert isinstance(loop, ObservingSelector)
    assert observed == [loop]
