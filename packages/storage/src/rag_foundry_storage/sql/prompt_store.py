"""SQL implementation of the immutable prompt template store."""

from datetime import UTC
from typing import Any

from rag_foundry_domain.errors import PromptError
from rag_foundry_domain.prompts import (
    PromptActivation,
    PromptPurpose,
    PromptTemplate,
    template_from_row,
    validate_loaded,
)
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_foundry_storage.sql._tx import read, write


def _row_template(row: Any) -> PromptTemplate:
    return template_from_row(
        id=row["id"],
        family=row["family"],
        version=row["version"],
        purpose=row["purpose"],
        system_text=row["system_text"],
        content_sha256=row["content_sha256"].strip(),
    )


def _sqlstate(exc: IntegrityError) -> str | None:
    original = exc.orig
    return getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)


class SqlPromptTemplateStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, template_id: str) -> PromptTemplate:
        try:
            async with read(self._session) as session:
                result = await session.execute(
                    text(
                        "SELECT id, family, version, purpose, system_text, content_sha256 "
                        "FROM prompt_templates WHERE id = :id"
                    ),
                    {"id": template_id},
                )
                row = result.mappings().first()
        except (OperationalError, InterfaceError):
            raise PromptError("store_unavailable") from None
        if row is None:
            raise PromptError("template_not_found")
        template = _row_template(row)
        validate_loaded(template)
        return template

    async def active_id(self, purpose: PromptPurpose) -> str:
        try:
            async with read(self._session) as session:
                result = await session.execute(
                    text(
                        "SELECT template_id FROM prompt_activations WHERE purpose = :purpose ORDER BY seq DESC LIMIT 1"
                    ),
                    {"purpose": purpose},
                )
                row = result.first()
        except (OperationalError, InterfaceError):
            raise PromptError("store_unavailable") from None
        if row is None:
            raise PromptError("no_active_template")
        return str(row[0])

    async def create(self, template: PromptTemplate) -> None:
        validate_loaded(template)
        try:
            async with write(self._session) as session:
                await session.execute(
                    text(
                        """INSERT INTO prompt_templates
                        (id, family, version, purpose, system_text, content_sha256, created_by)
                        VALUES (:id, :family, :version, :purpose, :system_text, :content_sha256, 'application')"""
                    ),
                    {
                        "id": template.id,
                        "family": template.family,
                        "version": template.version,
                        "purpose": template.purpose,
                        "system_text": template.system_text,
                        "content_sha256": template.content_sha256,
                    },
                )
        except IntegrityError as exc:
            if _sqlstate(exc) == "23505":
                raise PromptError("template_exists") from None
            # Unmapped integrity errors would carry bound parameters (prompt text) in their message.
            raise PromptError("invalid_template") from None
        except (OperationalError, InterfaceError):
            raise PromptError("store_unavailable") from None

    async def activate(
        self, purpose: PromptPurpose, template_id: str, *, activated_by: str, decision_ref: str
    ) -> PromptActivation:
        try:
            async with write(self._session) as session:
                result = await session.execute(
                    text("""INSERT INTO prompt_activations (purpose, template_id, activated_by, decision_ref)
                    VALUES (:purpose, :template_id, :activated_by, :decision_ref)
                    RETURNING seq, purpose, template_id, activated_by, activated_at, decision_ref"""),
                    {
                        "purpose": purpose,
                        "template_id": template_id,
                        "activated_by": activated_by,
                        "decision_ref": decision_ref,
                    },
                )
                row = result.mappings().one()
        except IntegrityError as exc:
            code = _sqlstate(exc)
            if code == "23503":
                raise PromptError("template_not_found") from None
            if code == "23000" or "prompt activation purpose mismatch" in str(exc.orig):
                raise PromptError("purpose_mismatch") from None
            raise PromptError("invalid_template") from None
        except (OperationalError, InterfaceError):
            raise PromptError("store_unavailable") from None
        return PromptActivation(
            seq=row["seq"],
            purpose=row["purpose"],
            template_id=row["template_id"],
            activated_by=row["activated_by"],
            activated_at=row["activated_at"].astimezone(UTC),
            decision_ref=row["decision_ref"],
        )

    async def activations(self, purpose: PromptPurpose) -> list[PromptActivation]:
        try:
            async with read(self._session) as session:
                result = await session.execute(
                    text("""SELECT seq, purpose, template_id, activated_by, activated_at, decision_ref
                    FROM prompt_activations WHERE purpose = :purpose ORDER BY seq"""),
                    {"purpose": purpose},
                )
                rows = result.mappings().all()
        except (OperationalError, InterfaceError):
            raise PromptError("store_unavailable") from None
        return [
            PromptActivation(
                seq=row["seq"],
                purpose=row["purpose"],
                template_id=row["template_id"],
                activated_by=row["activated_by"],
                activated_at=row["activated_at"].astimezone(UTC),
                decision_ref=row["decision_ref"],
            )
            for row in rows
        ]
