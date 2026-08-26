"""Drive the live generalist agent to generate recorded trajectories.

Forges a backend session JWT (HS256 over ``settings.backend_auth_secret``),
POSTs turns to the running server's ``POST /optimizations/generalist-agent``
SSE endpoint, and threads ``conversation_id`` + ``chat_history`` +
``wizard_state`` across a multi-turn conversation. Each driven turn is
persisted to ``agent_messages`` — those rows are the training trajectories.

``wizard_state`` is threaded from the DB (the last turn's persisted
``wizard_state_after``) so the agent progresses through the wizard naturally.

CLI (one conversation): ``.venv/bin/python -m scripts.agent_driver \
  --user sampler1@skynet.local --messages-json '["hi","..."]' --trust yolo``
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

import httpx
from sqlalchemy import create_engine, text

from core.config import settings

SERVER = os.environ.get("SKYNET_SERVER", "http://localhost:8000")
_DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres@localhost:5432/skynet")


def _b64url(raw: bytes) -> str:
    """URL-safe base64 without padding (JWT segment encoding)."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _secret() -> str:
    """Return the raw backend auth secret string."""
    value = settings.backend_auth_secret
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def mint_token(name: str, *, ttl_seconds: int = 3600) -> str:
    """Forge a backend session JWT for ``name``.

    Args:
        name: The username/email to attribute the conversation to.
        ttl_seconds: Token lifetime.

    Returns:
        A signed HS256 JWT accepted by ``get_authenticated_user``.
    """
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    payload = _b64url(
        json.dumps(
            {
                "aud": "skynet-backend",
                "iss": "skynet-frontend",
                "sub": name,
                "name": name,
                "role": "user",
                "iat": now,
                "exp": now + ttl_seconds,
            }
        ).encode("utf-8")
    )
    signature = _b64url(
        hmac.new(_secret().encode("utf-8"), f"{header}.{payload}".encode("ascii"), hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{signature}"


def _parse_sse(resp: httpx.Response) -> list[dict[str, Any]]:
    """Parse an ``event:``/``data:`` SSE stream into ordered event dicts.

    Args:
        resp: An open streaming response.

    Returns:
        The events in arrival order, each ``{"event", "data"}``.
    """
    events: list[dict[str, Any]] = []
    name: str | None = None
    data_lines: list[str] = []
    for line in resp.iter_lines():
        if line.startswith("event:"):
            name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif line == "":
            if name is not None:
                blob = "\n".join(data_lines)
                try:
                    data: Any = json.loads(blob) if blob else {}
                except json.JSONDecodeError:
                    data = blob
                events.append({"event": name, "data": data})
            name, data_lines = None, []
    return events


def _latest_wizard_state_after(conversation_id: str | None) -> dict[str, Any]:
    """Read the most recent persisted ``wizard_state_after`` for a conversation."""
    if not conversation_id:
        return {}
    engine = create_engine(_DB_URL)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT wizard_state_after FROM agent_messages "
                "WHERE conversation_id = :cid AND role = 'assistant' "
                "ORDER BY created_at DESC, id DESC LIMIT 1"
            ),
            {"cid": conversation_id},
        ).first()
    return dict(row.wizard_state_after) if row and row.wizard_state_after else {}


def _conversation_tools(conversation_id: str | None) -> list[str]:
    """Return the tool names recorded across a conversation's assistant turns."""
    if not conversation_id:
        return []
    engine = create_engine(_DB_URL)
    tools: list[str] = []
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT tool_calls FROM agent_messages "
                "WHERE conversation_id = :cid AND role = 'assistant' AND tool_calls IS NOT NULL"
            ),
            {"cid": conversation_id},
        ).all()
    for r in rows:
        for call in r.tool_calls or []:
            name = call.get("tool") if isinstance(call, dict) else None
            if name:
                tools.append(name)
    return tools


def drive_turn(
    *,
    user_message: str,
    chat_history: list[dict[str, str]],
    wizard_state: dict[str, Any],
    trust_mode: str,
    conversation_id: str | None,
    token: str,
    timeout: float = 240.0,
) -> dict[str, Any]:
    """Drive one agent turn over the SSE endpoint.

    Args:
        user_message: The user's message for this turn.
        chat_history: Prior ``{role, content}`` turns.
        wizard_state: The wizard snapshot to expose tools for this turn.
        trust_mode: ``"yolo"`` (auto-approve), ``"auto_safe"``, or ``"ask"``.
        conversation_id: Existing conversation id, or ``None`` to start one.
        token: A bearer session token from :func:`mint_token`.
        timeout: Per-turn HTTP timeout.

    Returns:
        ``{conversation_id, assistant_message, event_types}`` for the turn.
    """
    body = {
        "user_message": user_message,
        "chat_history": chat_history,
        "wizard_state": wizard_state,
        "trust_mode": trust_mode,
        "conversation_id": conversation_id,
    }
    with httpx.Client(timeout=timeout) as client, client.stream(
        "POST",
        f"{SERVER}/optimizations/generalist-agent",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    ) as resp:
        resp.raise_for_status()
        events = _parse_sse(resp)

    conv_id = conversation_id
    assistant = ""
    for e in events:
        if e["event"] == "conversation_meta" and isinstance(e["data"], dict):
            conv_id = e["data"].get("conversation_id", conv_id)
        if e["event"] == "done" and isinstance(e["data"], dict):
            assistant = e["data"].get("assistant_message", assistant)
    return {
        "conversation_id": conv_id,
        "assistant_message": assistant,
        "event_types": [e["event"] for e in events],
    }


def drive_conversation(
    messages: list[str],
    *,
    user: str,
    wizard_state: dict[str, Any] | None = None,
    trust_mode: str = "yolo",
) -> dict[str, Any]:
    """Drive a full multi-turn conversation, threading state from the DB.

    Args:
        messages: The user messages to send, in order.
        user: The username to attribute the conversation to.
        wizard_state: The initial wizard snapshot (defaults to empty intake).
        trust_mode: Trust mode for tool execution.

    Returns:
        ``{conversation_id, turns, tools_recorded}`` summarising the run.
    """
    token = mint_token(user)
    conv_id: str | None = None
    chat_history: list[dict[str, str]] = []
    state: dict[str, Any] = dict(wizard_state or {})
    turns: list[dict[str, Any]] = []
    for msg in messages:
        result = drive_turn(
            user_message=msg,
            chat_history=chat_history,
            wizard_state=state,
            trust_mode=trust_mode,
            conversation_id=conv_id,
            token=token,
        )
        conv_id = result["conversation_id"]
        chat_history = [
            *chat_history,
            {"role": "user", "content": msg},
            {"role": "assistant", "content": result["assistant_message"]},
        ]
        state = _latest_wizard_state_after(conv_id) or state
        turns.append({"user": msg[:80], "assistant": result["assistant_message"][:120], "events": result["event_types"]})
    return {
        "conversation_id": conv_id,
        "turns": turns,
        "tools_recorded": _conversation_tools(conv_id),
    }


def main() -> int:
    """CLI entry: drive one conversation and print a JSON summary."""
    parser = argparse.ArgumentParser(description="Drive the generalist agent to record a trajectory.")
    parser.add_argument("--user", default="sampler@skynet.local")
    parser.add_argument("--messages-json", required=True, help="JSON array of user messages.")
    parser.add_argument("--wizard-state-json", default="{}")
    parser.add_argument("--trust", default="yolo", choices=["yolo", "auto_safe", "ask"])
    args = parser.parse_args()
    summary = drive_conversation(
        json.loads(args.messages_json),
        user=args.user,
        wizard_state=json.loads(args.wizard_state_json),
        trust_mode=args.trust,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
