"""Reusable conformance checks for PromptTemplateStore implementations."""

from __future__ import annotations

from uuid import uuid4

from rag_foundry_domain.errors import PromptError
from rag_foundry_domain.prompts import (
    PromptPurpose,
    PromptTemplate,
    PromptTemplateStore,
    prompt_sha256,
)


def _template(family: str, purpose: PromptPurpose, version: int = 1) -> PromptTemplate:
    text = f"test prompt {family} INSUFFICIENT_EVIDENCE [S1]"
    return PromptTemplate(
        id=f"{family}-v{version}",
        family=family,
        version=version,
        purpose=purpose,
        system_text=text,
        content_sha256=prompt_sha256(text),
    )


async def _expect_error(awaitable: object, code: str) -> None:
    try:
        await awaitable  # type: ignore[misc]
    except PromptError as exc:
        if exc.code != code:
            raise AssertionError(f"expected prompt error {code}") from None
    else:
        raise AssertionError(f"expected prompt error {code}")


async def check_prompt_template_store(store: PromptTemplateStore) -> None:
    suffix = uuid4().hex[:12]
    grounded = _template(f"check-{suffix}-grounded", "grounded")
    grounded_v2 = _template(f"check-{suffix}-grounded", "grounded", 2)
    ungrounded = _template(f"check-{suffix}-ungrounded", "ungrounded")
    previous: dict[PromptPurpose, str | None] = {}
    for purpose in ("grounded", "ungrounded"):
        try:
            previous[purpose] = await store.active_id(purpose)
        except PromptError as exc:
            if exc.code != "no_active_template":
                raise
            previous[purpose] = None
    body_failed = True
    try:
        await store.create(grounded)
        if await store.get(grounded.id) != grounded:
            raise AssertionError("created prompt did not round-trip")
        await _expect_error(store.create(grounded), "template_exists")
        await store.create(grounded_v2)
        await store.create(ungrounded)
        await _expect_error(
            store.activate("ungrounded", grounded.id, activated_by="check", decision_ref="check"),
            "purpose_mismatch",
        )
        await store.activate("grounded", grounded.id, activated_by="check", decision_ref="check")
        await store.activate("grounded", grounded_v2.id, activated_by="check", decision_ref="check")
        if await store.active_id("grounded") != grounded_v2.id:
            raise AssertionError("latest activation did not win")
        rows = await store.activations("grounded")
        if tuple(row.seq for row in rows) != tuple(sorted(row.seq for row in rows)):
            raise AssertionError("activation history is not ordered")
        await _expect_error(store.get(f"missing-{suffix}-v1"), "template_not_found")
        body_failed = False
    finally:
        restore_error: Exception | None = None
        for purpose, template_id in previous.items():
            if template_id is not None:
                try:
                    await store.activate(
                        purpose,
                        template_id,
                        activated_by="prompt store check restore",
                        decision_ref="conformance check restore",
                    )
                except Exception as exc:
                    if restore_error is None:
                        restore_error = exc
        if restore_error is not None and not body_failed:
            raise restore_error


PROMPT_STORE_CHECKS = (check_prompt_template_store,)
