"""Block every non-loopback network connection in provider tests.

This is a test helper that catches accidental network use through the common Python
socket entry points. It is not an isolation or security boundary: it does not cover
other resolvers or C extensions. It also wraps `asyncio.BaseEventLoop.create_connection`,
which blocks non-loopback literal IPs on selector and proactor loops; hostnames are not
resolved here. Do not rely on it for containment.
"""

import asyncio
import ipaddress
import socket
from typing import Any

import pytest

_MESSAGE = "network blocked in provider tests"


def _is_loopback(address: Any) -> bool:
    # AF_UNIX paths are str/bytes; only tuple addresses can leave the host.
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(_MESSAGE)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if not _is_loopback(address):
            raise RuntimeError(_MESSAGE)
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if not _is_loopback(address):
            raise RuntimeError(_MESSAGE)
        return real_connect_ex(self, address)

    real_create_connection = asyncio.BaseEventLoop.create_connection

    async def guarded_create_connection(
        self: Any, protocol_factory: Any, host: Any = None, port: Any = None, **kw: Any
    ):
        if host is not None and not _is_loopback((host, port)):
            raise RuntimeError(_MESSAGE)
        return await real_create_connection(self, protocol_factory, host, port, **kw)

    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "gethostbyname", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
