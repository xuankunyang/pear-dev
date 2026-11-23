import asyncio
import json
import threading

import httpx
import openai
import pytest

from pear.deploy.station.agent_loop.action_parser import ActionParser
from pear.deploy.station.agent_loop.clients.client import ContextLimit
from pear.deploy.station.agent_loop.clients.openai_api import OpenAIAPIClient
from pear.deploy.station.agent_loop.clients.schemas import CameraImage, ModelEvent, ModelInput, ReferenceImage
from pear.deploy.station.control.robots.yam.formats import BIMANUAL_POSE
from pear.deploy.station.exceptions import OperatorPaused

ACTION = {part: dict.fromkeys(names, 0.0) for part, names in BIMANUAL_POSE.schema.items()}
ANSWER = "```json\n" + json.dumps(ACTION) + "\n```"


def completed(answer=ANSWER, refusal=False):
    content = (
        [{"type": "refusal", "refusal": "Request refused"}]
        if refusal
        else [{"type": "output_text", "text": answer, "annotations": []}]
    )
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6-astra",
        "status": "completed",
        "output": [
            {
                "id": "rs_test",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Hold still."}],
                "encrypted_content": "encrypted-reasoning",
            },
            {"id": "msg_test", "type": "message", "status": "completed", "role": "assistant", "content": content},
        ],
        "usage": {
            "input_tokens": 200_000,
            "output_tokens": 25,
            "total_tokens": 200_025,
            "input_tokens_details": {"cached_tokens": 120},
            "output_tokens_details": {"reasoning_tokens": 10},
        },
    }


class Stream(httpx.AsyncByteStream):
    def __init__(self, mode, cancellation):
        self.mode = mode
        self.cancellation = cancellation
        self.closed = False

    async def __aiter__(self):
        def event(kind, **kwargs):
            data = {"type": kind, "sequence_number": 0, **kwargs}
            return (f"event: {kind}\ndata: " + json.dumps(data) + "\n\n").encode()

        yield event(
            "response.output_text.delta",
            delta=ANSWER,
            item_id="msg_test",
            output_index=1,
            content_index=0,
        )
        if self.mode == "cancel":
            self.cancellation.set()
            await asyncio.sleep(0.1)
        if self.mode == "late_cancel":
            self.cancellation.set()
        if self.mode == "stall":
            await asyncio.sleep(1)
        if self.mode == "disconnect":
            return
        if self.mode == "stream_error":
            yield event("error", code="server_error", message="temporary failure", param=None)
            return
        if self.mode in ("incomplete", "failed", "overloaded", "stream_context", "secret_error"):
            response = completed()
            status = self.mode if self.mode in ("incomplete", "failed") else "failed"
            response["status"] = status
            if self.mode in ("overloaded", "stream_context", "secret_error"):
                response["error"] = {
                    "code": "context_length_exceeded" if self.mode == "stream_context" else "server_error",
                    "message": "upstream rejected test-key"
                    if self.mode == "secret_error"
                    else "Our servers are currently overloaded.",
                }
            yield event(f"response.{status}", response=response)
            return
        response = completed("no action" if self.mode == "invalid_action" else ANSWER, self.mode == "refusal")
        yield event("response.completed", response=response)

    async def aclose(self):
        self.closed = True


@pytest.fixture
def service(monkeypatch):
    calls, streams, modes = [], [], []
    cancellation = threading.Event()
    real_async, real_sync = openai.AsyncOpenAI, openai.OpenAI

    def handle(request):
        if request.method == "GET":
            assert request.url.path == "/v1/models/gpt-6-astra"
            return httpx.Response(
                200, json={"id": "gpt-6-astra", "object": "model", "created": 1, "owned_by": "openai"}
            )
        assert request.url.path == "/v1/responses"
        calls.append(json.loads(request.content))
        mode = modes.pop(0) if modes else "ok"
        if mode in ("429", "500", "context"):
            status = {"429": 429, "500": 500, "context": 400}[mode]
            return httpx.Response(
                status,
                json={
                    "error": {
                        "message": "test failure",
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded" if mode == "context" else "test_error",
                    }
                },
            )
        stream = Stream(mode, cancellation)
        streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-request-id": "req_test"},
            stream=stream,
        )

    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        lambda **kw: real_async(
            **kw,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ),
    )
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: real_sync(
            **kw,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ),
    )
    return calls, streams, modes, cancellation


@pytest.fixture
def model_input(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"test-jpeg")
    return ModelInput(
        "Measured state. Previous execution: unreachable target.",
        tuple(CameraImage(name, path) for name in ("head", "left_wrist", "right_wrist")),
        (ReferenceImage(path),),
    )


def new_client(**kwargs):
    return OpenAIAPIClient("station prompt", timeout_s=1, base_url="https://test.local/v1", **kwargs)


def infer(client, model_input, service):
    events = []
    output = client.infer(model_input, on_event=lambda *e: events.append(e), cancellation=service[3])
    return output, events


def test_responses_history_images_usage_and_no_context_cap(service, model_input, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = new_client()
    assert client.check()["status"] == "ok"
    output, events = infer(client, model_input, service)
    ActionParser.parse(output.answer, BIMANUAL_POSE)
    assert output.usage["input"] == 200_000  # Larger than the old PEAR 128k configuration.
    assert output.usage["context_limit"] == 1_050_000
    assert output.usage["reasoning_tokens"] == 10
    assert output.reasoning_kind == "summary"
    assert output.reasoning == "Hold still."
    first = service[0][0]
    assert first["store"] is False and first["stream"] is True
    assert first["truncation"] == "disabled" and first["instructions"] == "station prompt"
    assert first["max_output_tokens"] == 32_000
    assert first["reasoning"]["effort"] == "xhigh"
    assert not {"temperature", "top_k", "chat_template_kwargs", "context_limit"} & first.keys()
    assert sum(p["type"] == "input_image" for p in first["input"][0]["content"]) == 4
    assert all("uuid" not in p for p in first["input"][0]["content"])
    infer(client, model_input, service)
    second = service[0][1]["input"]
    assert [p["type"] for p in second[0]["content"]].count("input_image") == 4
    assert [p["type"] for p in second[-1]["content"]].count("input_image") == 3
    assert second[1]["encrypted_content"] == "encrypted-reasoning"
    assert second[2]["role"] == "assistant"
    assert "unreachable" in second[-1]["content"][0]["text"]
    assert any(kind == ModelEvent.REQUEST_METRICS and data.get("api_request_id") == "req_test" for kind, data in events)
    client.close()
    assert not client.history
    assert all(s.closed for s in service[1])


@pytest.mark.parametrize(
    "mode", ["disconnect", "incomplete", "failed", "refusal", "stream_error", "429", "500", "context"]
)
def test_failed_response_keeps_committed_history(service, model_input, monkeypatch, mode):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = new_client()
    infer(client, model_input, service)
    history = list(client.history)
    service[2].append(mode)
    exception = (
        ContextLimit if mode == "context" else (ValueError if mode not in ("429", "500") else openai.APIStatusError)
    )
    with pytest.raises(exception):
        infer(client, model_input, service)
    assert client.history == history
    assert len(service[0]) == 2  # No automatic retry.
    assert all(s.closed for s in service[1])


@pytest.mark.parametrize("mode", ["cancel", "late_cancel", "stall"])
def test_cancel_and_timeout_discard_partial_action(service, model_input, monkeypatch, mode):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = new_client()
    client.timeout_s = 0.05 if mode == "stall" else 1
    service[2].append(mode)
    events = []
    with pytest.raises(TimeoutError if mode == "stall" else OperatorPaused):
        client.infer(model_input, on_event=lambda *e: events.append(e), cancellation=service[3])
    assert client.history == []
    assert all(s.closed for s in service[1])
    assert events[-1][0] == ModelEvent.PERFORMANCE
    assert events[-1][1]["status"] == ("failed" if mode == "stall" else "interrupted")


def test_missing_key_and_key_file(service, model_input, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key_file"):
        infer(new_client(), model_input, service)
    key = tmp_path / "key"
    key.write_text("test-key\n")
    client = new_client(api_key_file=str(key))
    infer(client, model_input, service)
    assert service[0]


def test_invalid_action_is_rejected_by_station_parser(service, model_input, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    service[2].append("invalid_action")
    output, _ = infer(new_client(), model_input, service)
    with pytest.raises(ValueError, match="fenced json"):
        ActionParser.parse(output.answer, BIMANUAL_POSE)


@pytest.mark.parametrize("mode", ["overloaded", "stream_context", "secret_error"])
def test_terminal_error_preserves_provider_detail_and_redacts_key(service, model_input, monkeypatch, mode):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    service[2].append(mode)
    client = new_client()
    with pytest.raises(ContextLimit if mode == "stream_context" else ValueError) as caught:
        infer(client, model_input, service)
    message = str(caught.value)
    if mode == "overloaded":
        assert "server_error" in message and "overloaded" in message
    if mode == "secret_error":
        assert "test-key" not in message and "[redacted]" in message
    assert client.history == []
    assert len(service[0]) == 1
