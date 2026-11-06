import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient
from rag_foundry_answering.runner import LocalRunner
from rag_foundry_api.main import create_app
from rag_foundry_contracts.runs import Citation, Evidence, RunResponse
from rag_foundry_contracts.source import Precision, SourceAnchor, SourceRef, TextRange
from rag_foundry_core.settings import Settings
from rag_foundry_domain import LOCAL_DEV_SCOPE, NotFoundError
from rag_foundry_providers.errors import BudgetExceededError, ProviderUnavailableError

PROJECT = LOCAL_DEV_SCOPE.project_id
URL = f"/api/v1/projects/{PROJECT}/runs"


def _response(status="ungrounded"):
    anchor = SourceAnchor(
        anchor_id="anchor-1",
        source=SourceRef(
            provider_namespace="test", document_id="doc", document_version_id="version", extraction_view_id="view"
        ),
        block_id="b00000",
        text_range=TextRange(start=0, end=4),
        exact="fact",
        precision=Precision.EXACT_TEXT,
    )
    return RunResponse(
        run_id=str(uuid.uuid4()),
        status=status,
        abstention_reason=None,
        answer="fact [S1]" if status == "answered" else "an answer",
        citations=(Citation(marker="S1", anchors=(anchor,)),) if status == "answered" else (),
        evidence=(Evidence(marker="S1", rank=1, score=1.0, used=True, anchors=(anchor,)),)
        if status == "answered"
        else (),
        pins={
            "technique_id": "R02" if status == "answered" else "R01",
            "technique_version": "1.0.0",
            "index_version_id": str(uuid.uuid4()) if status == "answered" else None,
            "embedding_key": "0" * 64 if status == "answered" else None,
            "chat_profile_id": "local-chat",
            "chat_provider": "ollama",
            "chat_model": "gemma4:e4b",
            "prompt_template": "grounded-answer-v2" if status == "answered" else "ungrounded-v1",
            "min_score": 0.0 if status == "answered" else None,
            "k": 5 if status == "answered" else None,
        },
        usage={"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "estimated_cost_usd": "0"},
        warnings=(),
    )


class StubRunner:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.scopes = result or _response(), error, []

    async def run(self, scope, body):
        self.scopes.append(scope)
        if self.error:
            raise self.error
        return self.result


def _client(runner=None, settings=None):
    return TestClient(create_app(settings or Settings(environment="test"), run_service=runner))


class IndexDouble:
    def __init__(self, version=None, error=None):
        self.version, self.error = version, error

    async def get_index_version(self, scope, index_version_id):
        if self.error:
            raise self.error
        return self.version


def _local_runner(index):
    calls = []

    def factory(_ledger, _budget):
        calls.append("factory")
        raise AssertionError("factory must not be reached for this request")

    runner = LocalRunner(index, None, model_factory=factory)
    runner.model_calls = calls
    return runner


def _post(client, **extra):
    body = {"question": "Q", "technique_id": "R01"}
    body.update(extra)
    return client.post(URL, json=body)


def test_r01_run_returns_200_ungrounded():
    runner = StubRunner()
    response = _post(_client(runner))
    assert response.status_code == 200 and response.json()["status"] == "ungrounded"
    assert runner.scopes == [LOCAL_DEV_SCOPE]


def test_r02_answered_response_validates_as_run_response():
    result = _response("answered")
    # The service boundary returns its typed contract; route validation serializes it unchanged.
    response = _post(_client(StubRunner(result)), technique_id="R02", index_version_id=str(uuid.uuid4()), min_score=0.0)
    assert response.status_code == 200 and RunResponse.model_validate(response.json()).status == "answered"


def test_unknown_project_returns_404():
    response = _client(StubRunner()).post(f"/api/v1/projects/{uuid.uuid4()}/runs", json={"question": "Q"})
    assert response.status_code == 404


def test_r02_requires_index_version():
    runner = _local_runner(IndexDouble())
    response = _post(_client(runner), technique_id="R02")
    assert response.status_code == 422 and response.json()["error"]["code"] == "index_version_required"


def test_r01_rejects_index_version():
    response = _post(_client(_local_runner(IndexDouble())), index_version_id=str(uuid.uuid4()))
    assert response.status_code == 422 and response.json()["error"]["code"] == "index_version_not_allowed"


def test_unknown_technique_returns_422():
    response = _post(_client(_local_runner(IndexDouble())), technique_id="R03")
    assert response.status_code == 422 and response.json()["error"]["code"] == "unsupported_technique"


def test_out_of_scope_index_version_is_404():
    # The repository's NotFoundError is deliberately allowed to propagate to the standard 404 mapping.
    runner = _local_runner(IndexDouble(error=NotFoundError()))
    response = _post(_client(runner), technique_id="R02", index_version_id=str(uuid.uuid4()))
    assert response.status_code == 404
    assert runner.model_calls == []


def test_failed_version_maps_to_409_not_ready():
    runner = _local_runner(IndexDouble(version=SimpleNamespace(state="failed")))
    response = _post(_client(runner), technique_id="R02", index_version_id=str(uuid.uuid4()))
    assert response.status_code == 409 and response.json()["error"]["code"] == "not_ready"
    assert runner.model_calls == []


def test_provider_error_body_omits_exception_text():
    error = ProviderUnavailableError("local")
    error.args = ("sentinel-7731",)

    def factory(_ledger, _budget):
        raise error

    runner = LocalRunner(IndexDouble(), None, model_factory=factory)
    response = _post(_client(runner))
    assert response.status_code == 503 and response.json()["error"]["code"] == "provider_unavailable"
    assert "sentinel-7731" not in response.text


def test_budget_error_maps_to_409():
    response = _post(_client(StubRunner(error=BudgetExceededError("local", "calls", 1, 1, 0, 1))))
    assert response.status_code == 409 and response.json()["error"]["code"] == "budget_exceeded"


def test_long_question_validation_maps_to_422():
    response = _post(_client(StubRunner()), question="x" * 2001)
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


def test_prod_authentication_maps_to_401():
    response = _post(_client(StubRunner(), Settings(environment="prod")))
    assert response.status_code == 401


def test_missing_run_service_maps_to_503():
    response = _post(_client())
    assert response.status_code == 503 and response.json()["error"]["code"] == "storage_unavailable"
