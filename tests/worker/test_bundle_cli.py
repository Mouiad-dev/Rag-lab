from types import SimpleNamespace

from rag_foundry_worker import bundle_cli
from rag_foundry_worker.bundle_cli import main


def test_prod_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(bundle_cli, "load_settings", lambda: SimpleNamespace(environment="prod"))
    assert main(["verify", "bundle"]) == 2
    assert "no auth context" in capsys.readouterr().err


def test_bad_directory_reports_bundle_code(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        bundle_cli,
        "load_settings",
        lambda: SimpleNamespace(environment="dev", database_url="db"),
    )
    assert main(["verify", str(tmp_path / "missing")]) == 1
    assert capsys.readouterr().err.strip() == "error: bad_path"


def test_verify_succeeds_without_database_url(monkeypatch, tmp_path, capsys):
    import asyncio

    from rag_foundry_ingestion.bundle import write_bundle_dir
    from test_source_bundle import _bundle

    _owner, files, _anchors = asyncio.run(_bundle(tmp_path))
    bundle_dir = tmp_path / "bundle"
    write_bundle_dir(files, bundle_dir)
    monkeypatch.setattr(bundle_cli, "load_settings", lambda: SimpleNamespace(environment="dev"))
    assert main(["verify", str(bundle_dir)]) == 0
    output = capsys.readouterr().out
    assert "bundle_id=" in output


def test_export_and_verify_through_main_on_a_seeded_dev_store(monkeypatch, tmp_path, capsys):
    import asyncio
    from contextlib import asynccontextmanager

    from rag_foundry_domain import LOCAL_DEV_SCOPE
    from test_bundle_collector import NS, world

    owner, store, uow, _anchors, _index, owned, _foreign, built = asyncio.run(world(tmp_path, LOCAL_DEV_SCOPE))

    @asynccontextmanager
    async def open_db(_url):
        yield SimpleNamespace(uow=uow)

    settings = SimpleNamespace(environment="dev", database_url="db", object_store_root=tmp_path, provider_namespace=NS)
    monkeypatch.setattr(bundle_cli, "load_settings", lambda: settings)
    monkeypatch.setattr(bundle_cli, "open_database", open_db)
    monkeypatch.setattr(bundle_cli, "LocalObjectStore", lambda *_: store)
    out = tmp_path / "out"
    assert main(["export", "--views", *owned, "--index", built.id, "--out", str(out)]) == 0
    exported = capsys.readouterr()
    assert exported.err == ""
    lines = exported.out.splitlines()
    assert [line.split("=")[0] for line in lines] == ["bundle_id", "files", "bytes", "anchors", "views"]
    assert "views=2" in lines and not any(line == "anchors=0" for line in lines)
    for secret in (str(owner.tenant_id), str(owner.project_id), "Second guide", "pump", str(tmp_path)):
        assert secret not in exported.out
    assert main(["verify", str(out)]) == 0
    verified = capsys.readouterr().out.splitlines()
    assert verified[0] == lines[0] and verified[3] == lines[3]
    assert main(["export", "--views", *owned, "--out", str(out)]) == 1
    assert capsys.readouterr().err.strip() == "error: target_exists"
