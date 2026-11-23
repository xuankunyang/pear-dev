"""OpenAI Responses API adapter for the station's existing robot-use protocol."""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient, TurnProgress
from pear.deploy.station.agent_loop.clients.metrics import Performance, RequestStatus
from pear.deploy.station.agent_loop.clients.schemas import (
    EventHandler,
    HealthStatus,
    ModelEvent,
    ModelHealth,
    ModelInput,
    ModelOutput,
    ReasoningKind,
    Usage,
)
from pear.deploy.station.exceptions import OperatorPaused


@dataclass(kw_only=True)
class ResponsesProgress(TurnProgress):
    """Only a completed response may become the next turn's history."""

    response: Any = None
    summary: str = ""


class OpenAIAPIClient(ModelClient):
    """Streams Responses, replaying complete history without an application context cap."""

    LABEL = "OpenAI"
    REASONING_EFFORT = "xhigh"
    RELATIVE_PATH_ARGUMENTS = ("api_key_file",)
    CHECK_TIMEOUT_S = 10
    MODEL_CONTEXT_WINDOW = 1_050_000  # Display metadata for gpt-6-astra, not a local truncation threshold.

    def __init__(
        self,
        system_prompt: str,
        *,
        timeout_s: float,
        model: str = "gpt-6-astra",
        base_url: str = "https://api.openai.com/v1",
        api_key_file: str | None = None,
        reasoning_effort: str = REASONING_EFFORT,
        max_output_tokens: int = 32_000,
        model_context_window: int = MODEL_CONTEXT_WINDOW,
    ) -> None:
        super().__init__(system_prompt, timeout_s=timeout_s)
        if not model or type(model) is not str:
            raise ValueError("model must be a non-empty model ID")
        if reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported reasoning_effort")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if type(model_context_window) is not int or model_context_window <= 0:
            raise ValueError("model_context_window must be a positive integer")
        self.model = model
        self.base_url = self._service_url(base_url)
        self.api_key_file = api_key_file
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.model_context_window = model_context_window
        self.history: list[dict[str, Any]] = []

    def _api_key(self) -> str:
        key = (
            Path(self.api_key_file).read_text().strip()
            if self.api_key_file
            else os.getenv("OPENAI_API_KEY", "").strip()
        )
        if not key:
            raise ValueError("Set api_key_file or OPENAI_API_KEY for the OpenAI client")
        return key

    def check(self) -> ModelHealth:
        from openai import OpenAI

        with OpenAI(
            api_key=self._api_key(),
            base_url=self.base_url,
            timeout=self.CHECK_TIMEOUT_S,
            max_retries=0,
        ) as client:
            client.models.retrieve(self.model)
        return {
            "status": HealthStatus.OK.value,
            "message": f"OpenAI model accessible: {self.model}; Responses inference not yet checked",
        }

    async def _infer_async(
        self, model_input: ModelInput, on_event: EventHandler, cancellation: threading.Event
    ) -> ModelOutput:
        import openai

        progress = ResponsesProgress(on_event=on_event, cancellation=cancellation, start_time=time.monotonic())
        status = RequestStatus.FAILED
        try:
            user_item = self._input_item(model_input)
            pending = [*self.history, user_item]
            async with openai.AsyncOpenAI(
                api_key=self._api_key(), base_url=self.base_url, timeout=self.timeout_s, max_retries=0
            ) as client:
                await self._race(
                    self._generate(client, pending, progress),
                    cancellation,
                    timeout_s=self.timeout_s,
                )
            OperatorPaused.check_cancelled(cancellation)
            OperatorPaused.check_cancelled(self._abort_event)
            response = progress.response
            answer = response.output_text
            if not answer.strip():
                raise ValueError("OpenAI returned no final answer")
            # Store the API's replayable output, including encrypted reasoning; never synthesize hidden reasoning.
            self.history = [*pending, *(item.model_dump(mode="json", exclude_none=True) for item in response.output)]
            status = RequestStatus.COMPLETED
            return ModelOutput(
                answer=answer,
                reasoning=progress.summary,
                reasoning_kind=ReasoningKind.SUMMARY.value if progress.summary else ReasoningKind.NONE.value,
                usage=progress.usage,
            )
        except OperatorPaused:
            status = RequestStatus.INTERRUPTED
            raise
        except openai.APIStatusError as error:
            body = error.body if isinstance(error.body, dict) else {}
            detail = body.get("error", body)
            if isinstance(detail, dict) and detail.get("code") in (
                "context_length_exceeded",
                "max_context_length_exceeded",
            ):
                raise ContextLimit("OpenAI model context is full. History is preserved; start a new task.") from error
            raise
        finally:
            on_event(
                ModelEvent.PERFORMANCE,
                Performance.report(
                    "openai", progress.start_time, progress.first_output_s, progress.usage, status=status
                ),
            )

    def _input_item(self, model_input: ModelInput) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": model_input.text}]
        if not self.history:
            for image in model_input.reference_images:
                content.extend(
                    [
                        {"type": "input_text", "text": "Task reference image"},
                        {"type": "input_image", "image_url": "data:image/jpeg;base64," + image.base64()},
                    ]
                )
        for image in model_input.images:
            content.extend(
                [
                    {"type": "input_text", "text": image.camera_name},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64," + image.base64()},
                ]
            )
        return {"role": "user", "content": content}

    async def _generate(self, client: Any, pending: list[dict[str, Any]], progress: ResponsesProgress) -> None:
        progress.on_event(
            ModelEvent.REQUEST_METRICS,
            {
                "model": self.model,
                "image_uploads": sum(
                    part.get("type") == "input_image"
                    for item in pending
                    for part in item.get("content", [])
                    if isinstance(part, dict)
                ),
                "application_context_limit": None,
            },
        )
        stream = await client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=pending,
            reasoning={"effort": self.reasoning_effort, "summary": "auto"},
            include=["reasoning.encrypted_content"],
            max_output_tokens=self.max_output_tokens,
            store=False,
            stream=True,
            truncation="disabled",
        )
        async with stream:
            progress.on_event(
                ModelEvent.REQUEST_METRICS,
                {"api_request_id": stream.response.headers.get("x-request-id"), "model": self.model},
            )
            async for event in stream:
                OperatorPaused.check_cancelled(progress.cancellation)
                OperatorPaused.check_cancelled(self._abort_event)
                if event.type == "response.output_text.delta":
                    progress.note_output()
                    progress.on_event(ModelEvent.ANSWER_DELTA, {"text": event.delta})
                elif event.type == "response.reasoning_summary_text.delta":
                    progress.note_output()
                    progress.summary += event.delta
                    progress.on_event(
                        ModelEvent.REASONING_DELTA,
                        {"text": event.delta, "reasoning_kind": ReasoningKind.SUMMARY.value},
                    )
                elif event.type == "response.completed":
                    progress.response = event.response
                    progress.usage = self._usage(event.response)
                    progress.on_event(ModelEvent.USAGE, progress.usage)
                elif event.type == "response.output_item.done" and event.item.type == "reasoning":
                    # Summaries can also arrive without delta events.
                    if not progress.summary:
                        progress.summary = "\n".join(
                            part.text for part in event.item.summary if part.type == "summary_text"
                        )
                elif event.type in ("response.failed", "response.incomplete"):
                    progress.usage = self._usage(event.response)
                    progress.on_event(ModelEvent.USAGE, progress.usage)
                    error = event.response.error
                    code = getattr(error, "code", None) or event.type
                    message = getattr(error, "message", None) or "No provider error detail"
                    if self._api_key() in message:
                        message = message.replace(self._api_key(), "[redacted]")
                    if code in ("context_length_exceeded", "max_context_length_exceeded"):
                        raise ContextLimit("OpenAI model context is full. History is preserved; start a new task.")
                    raise ValueError(f"OpenAI response did not complete ({code}): {message}")
                elif event.type in ("response.refusal.delta", "response.refusal.done"):
                    raise ValueError("OpenAI refused the request; no action is valid")
                elif event.type == "error":
                    raise ValueError(f"OpenAI stream error: {event.code}")
        response = progress.response
        if response is None or response.status != "completed":
            raise ValueError("OpenAI stream ended without a completed response")
        for item in response.output:
            if item.type not in ("message", "reasoning"):
                raise ValueError(f"Unexpected OpenAI output item: {item.type}")
            if item.type == "message" and any(part.type == "refusal" for part in item.content):
                raise ValueError("OpenAI refused the request; no action is valid")
        summaries = [
            part.text
            for item in response.output
            if item.type == "reasoning"
            for part in item.summary
            if part.type == "summary_text"
        ]
        if summaries:
            progress.summary = "\n".join(summaries)
        progress.on_event(
            ModelEvent.REQUEST_METRICS,
            {"response_id": response.id, "model": response.model},
        )

    def _usage(self, response: Any) -> Usage:
        counts = response.usage
        if counts is None:
            return {"context_limit": self.model_context_window, "final": True, "partial": True}
        result: Usage = {
            "input": counts.input_tokens,
            "output": counts.output_tokens,
            "cached_input": getattr(counts.input_tokens_details, "cached_tokens", 0),
            "context_limit": self.model_context_window,
            "final": True,
        }
        # Keep provider-specific accounting alongside the common station counters.
        result["reasoning_tokens"] = getattr(counts.output_tokens_details, "reasoning_tokens", 0)
        return result

    def _release(self) -> None:
        self.history.clear()
