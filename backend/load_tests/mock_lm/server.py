"""OpenAI-compatible mock LM server for Skynet load tests.

Implements the small subset of the Chat Completions API that DSPy / LiteLLM
actually call against ``openai/*`` model identifiers. The server returns
deterministic canned completions so worker job lifecycle paths can be exercised
under load without burning provider quota.

Tunables (env):
    MOCK_LM_LATENCY_MS:   Per-call latency in milliseconds. Defaults to 5 ms
                          which is fast enough to stress the queue + DB while
                          still mimicking a real provider call shape.
    MOCK_LM_FAILURE_RATE: Float in [0, 1]. Fraction of completions to fail
                          with HTTP 500. Default 0 — set during chaos runs.
    MOCK_LM_PORT:         Listen port. Default 9000.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Skynet Mock LM", version="1.0")

_LATENCY_MS = float(os.getenv("MOCK_LM_LATENCY_MS", "5"))
_FAILURE_RATE = float(os.getenv("MOCK_LM_FAILURE_RATE", "0"))

_CALL_COUNTER = 0
_START_TIME = time.monotonic()


_PLAIN_OPTIONS = (
    "yes",
    "no",
    "true",
    "false",
    "the answer is 42",
    "a short canned reply",
)

# DSPy's prompt enumerates output fields as ``\d+. `field_name` (type)``;
# this matches when ``response_format`` is absent so we can still derive
# field names from the user message body.
_DSPY_OUTPUT_FIELD_RE = re.compile(r"^\s*\d+\.\s*`([^`]+)`", re.MULTILINE)
# ChatAdapter prompts include ``[[ ## fieldname ## ]]`` markers and end
# with "Respond with the corresponding output fields, starting with the
# field `[[ ## fieldname ## ]]`". The presence of that exact instruction
# means the LM must reply in bracketed form, not JSON.
_DSPY_CHAT_MARKER_RE = re.compile(r"Respond with the corresponding output fields")


def _stable_text(prompt_hash: int) -> str:
    """Return a deterministic short reply derived from the prompt hash.

    Args:
        prompt_hash: Stable hash of the incoming messages used to vary text.

    Returns:
        One of the canned strings, picked by hashing the prompt so a
        retry of the same request yields the same answer.
    """
    return _PLAIN_OPTIONS[prompt_hash % len(_PLAIN_OPTIONS)]


def _schema_field_names(body: dict[str, Any]) -> list[str]:
    """Extract output field names from a ``response_format`` JSON schema.

    DSPy's JSONAdapter sends ``response_format={'type':'json_schema',
    'json_schema': {'schema': {'properties': {...}}}}`` to ask the LM
    for a structured response. We honour that by reading the properties
    so the canned reply uses the same field names the adapter parses.

    Args:
        body: Parsed JSON request body.

    Returns:
        Output property names in declaration order; empty list when the
        request didn't supply a ``json_schema`` response format.
    """
    rf = body.get("response_format") or {}
    if rf.get("type") not in {"json_schema", "json_object"}:
        return []
    schema = (rf.get("json_schema") or {}).get("schema") or {}
    props = schema.get("properties") or {}
    return list(props.keys())


def _prompt_field_names(messages: list[dict[str, Any]]) -> list[str]:
    """Extract output field names by parsing DSPy's prompt instructions.

    Falls back from ``_schema_field_names`` when the adapter requests
    JSON mode without a structured schema. DSPy still enumerates the
    output schema in the user message; we regex it out and return the
    names in order.

    Args:
        messages: ``messages`` array from the chat-completions request.

    Returns:
        Output field names found between the "Your output fields are"
        marker and the next blank line. Empty when no such marker is
        present (e.g. plain chat without a Predict signature).
    """
    text = "\n".join(
        m.get("content", "") if isinstance(m.get("content"), str) else "" for m in messages
    )
    marker = "Your output fields are"
    idx = text.find(marker)
    if idx == -1:
        return []
    block = text[idx : idx + 2000]
    names: list[str] = []
    for match in _DSPY_OUTPUT_FIELD_RE.finditer(block):
        name = match.group(1).strip()
        if name and name not in names:
            names.append(name)
    return names


def _completion_content(body: dict[str, Any], prompt_hash: int) -> str:
    """Build the assistant ``content`` string for a chat-completion reply.

    Detects which DSPy adapter is asking and returns the matching shape:
    JSONAdapter (``response_format`` set) gets ``{"field": value}``;
    ChatAdapter (prompt has bracketed markers) gets
    ``[[ ## field ## ]] value [[ ## completed ## ]]``. Plain chat
    requests with no DSPy markers fall back to a canned string.

    Args:
        body: Parsed chat-completions request body.
        prompt_hash: Stable hash used to pick a deterministic value.

    Returns:
        The string assigned to ``choices[0].message.content``.
    """
    value = _stable_text(prompt_hash)
    json_fields = _schema_field_names(body)
    if json_fields:
        return json.dumps(dict.fromkeys(json_fields, value))
    messages = body.get("messages") or []
    prompt_text = "\n".join(
        m.get("content", "") if isinstance(m.get("content"), str) else "" for m in messages
    )
    chat_mode = bool(_DSPY_CHAT_MARKER_RE.search(prompt_text))
    field_names = _prompt_field_names(messages)
    if chat_mode and field_names:
        parts = [f"[[ ## {name} ## ]]\n{value}" for name in field_names]
        parts.append("[[ ## completed ## ]]")
        return "\n\n".join(parts)
    if field_names:
        return json.dumps(dict.fromkeys(field_names, value))
    return value


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Return uptime + call count for harness readiness probes.

    Returns:
        A small JSON payload the orchestrator polls to confirm the mock is
        ready before bringing up the backends.
    """
    return {
        "status": "ok",
        "uptime_seconds": time.monotonic() - _START_TIME,
        "calls": _CALL_COUNTER,
        "latency_ms": _LATENCY_MS,
        "failure_rate": _FAILURE_RATE,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """Mimic OpenAI's chat completions endpoint with a canned response.

    Args:
        request: Incoming HTTP request whose JSON body matches the
            OpenAI Chat Completions schema (only ``model`` and ``messages``
            are read; everything else is accepted and ignored).

    Returns:
        A JSON response shaped like ``openai.ChatCompletion`` with a single
        choice. HTTP 500 when the configured failure rate fires.

    Raises:
        HTTPException: 500 with the configured probability, simulating a
            transient provider outage for chaos scenarios.
    """
    global _CALL_COUNTER
    _CALL_COUNTER += 1

    if _LATENCY_MS > 0:
        await asyncio.sleep(_LATENCY_MS / 1000.0)

    if _FAILURE_RATE > 0 and random.random() < _FAILURE_RATE:
        raise HTTPException(status_code=500, detail="mock-lm: injected failure")

    body = await request.json()
    model = body.get("model", "mock-model")
    messages = body.get("messages", [])
    prompt_text = "".join(
        m.get("content", "") if isinstance(m.get("content"), str) else "" for m in messages
    )
    canned = _completion_content(body, hash(prompt_text) & 0xFFFFFF)

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": canned},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(len(prompt_text) // 4, 1),
                "completion_tokens": max(len(canned) // 4, 1),
                "total_tokens": max(len(prompt_text) // 4 + len(canned) // 4, 2),
            },
        }
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    """Mimic OpenAI's embeddings endpoint with deterministic nonzero vectors.

    Args:
        request: Incoming HTTP request whose body contains ``input`` (string
            or list) and ``model``. Everything else is ignored.

    Returns:
        A JSON response shaped like ``openai.CreateEmbeddingResponse`` with
        one stable 1536-dimensional vector per input element. Nonzero vectors
        are required because the production adapter rejects a zero-norm
        provider response before pgvector storage or query execution.
    """
    if _LATENCY_MS > 0:
        await asyncio.sleep(_LATENCY_MS / 1000.0)

    body = await request.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]

    def _vector(value: object) -> list[float]:
        """Expand a SHA-256 digest into a stable nonzero vector.

        Args:
            value: Input value to fingerprint.

        Returns:
            A deterministic 1536-dimensional vector.
        """
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        return [((digest[index % len(digest)] + index) % 255 + 1) / 255.0 for index in range(1536)]

    return JSONResponse(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": _vector(value)}
                for index, value in enumerate(inputs)
            ],
            "model": body.get("model", "mock-embedding"),
            "usage": {"prompt_tokens": max(len(inputs), 1), "total_tokens": max(len(inputs), 1)},
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("MOCK_LM_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
