import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    import os

    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("RAG_FOUNDRY_")}
    env["RAG_FOUNDRY_API_PORT"] = str(port)
    log_path = tmp_path_factory.mktemp("api") / "server.log"
    log = log_path.open("wb")
    proc = subprocess.Popen(
        [sys.executable, "-c", "from rag_foundry_api.main import run; run()"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log.flush()
                output = log_path.read_text(errors="replace")
                pytest.fail(f"api server exited early:\n{output}")
            try:
                urllib.request.urlopen(url + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("api server did not start")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


def test_health_ok(base_url):
    with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
        assert r.status == 200
        assert json.loads(r.read()) == {"status": "ok", "version": "0.0.0"}


def test_unknown_route_404(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base_url + "/nope", timeout=5)
    assert e.value.code == 404


def test_create_app_registers_health_route():
    from rag_foundry_api.main import create_app
    from rag_foundry_core.settings import Settings

    app = create_app(Settings())
    assert "/health" in {r.path for r in app.routes}
