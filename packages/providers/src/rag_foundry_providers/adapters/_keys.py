"""API key reading shared by the HTTP adapters.

The key is read only from the environment and held as SecretStr. Errors name the variable,
never the value.
"""

import os
import re

from pydantic import SecretStr

from rag_foundry_providers.errors import MissingCredentialError

_KEY_CHARS = re.compile(r"[!-~]+")


def read_api_key(env_var: str = "ANTHROPIC_API_KEY") -> SecretStr:
    """Read the key from os.environ at call time.

    Empty, whitespace-only or malformed values (anything but printable ASCII without spaces
    after strip) raise MissingCredentialError naming only the variable, never the value.
    """
    value = os.environ.get(env_var, "").strip()
    if not value or _KEY_CHARS.fullmatch(value) is None:
        raise MissingCredentialError(env_var)
    return SecretStr(value)
