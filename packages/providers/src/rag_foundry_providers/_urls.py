"""Base URL validation shared by provider config. Mirrors Settings._validate_ollama_base_url.

Messages name the field and never include the value, so a URL with user info is not leaked.
"""

import ipaddress
from urllib.parse import urlsplit


def validate_base_url(value: str, field: str) -> str:
    """Return the URL without one trailing slash, or raise ValueError naming ``field``."""
    # urlsplit silently drops whitespace and control characters; reject them so the
    # stored URL is exactly the URL that was checked.
    if any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        raise ValueError(f"{field} must not contain whitespace or control characters")
    parts = urlsplit(value)
    try:
        _ = parts.port
    except ValueError:
        raise ValueError(f"{field} has an invalid port") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{field} must be an http or https URL with a host")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError(f"{field} must not contain user info")
    if "?" in value or "#" in value:
        raise ValueError(f"{field} must not contain a query or fragment")
    return value[:-1] if value.endswith("/") else value


def is_loopback(url: str) -> bool:
    """True for 127.0.0.0/8, ::1 and localhost hosts."""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
