"""Вызовы LLM только через официальный Gemini REST API (v1beta generateContent)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.ai.local_intent_model import local_intent_model
from app.core.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_TIMEOUT = 240.0
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


def _llm_trace(
    *,
    enabled: bool,
    used_provider: str | None,
    used_model: str | None,
    attempts: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    remote = any(isinstance(a, dict) and str(a.get("provider")) == "gemini" for a in attempts)
    return {
        "enabled": enabled,
        "used_provider": used_provider,
        "used_model": used_model,
        "attempts": attempts,
        "status": status,
        "remote_llm_invoked": remote,
    }


def _parse_json_payload(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned, _extract_first_json_object(cleaned)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
    return None


def _extract_first_json_object(value: str) -> str | None:
    match = re.search(r"\{.*\}", value, flags=re.DOTALL)
    return match.group(0) if match else None


def _gemini_response_text(data: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    meta: dict[str, Any] = {}
    pf = data.get("promptFeedback")
    if isinstance(pf, dict) and pf.get("blockReason"):
        meta["block_reason"] = pf.get("blockReason")
        meta["block_rating"] = pf.get("blockRating")
        return None, meta
    cands = data.get("candidates") or []
    if not cands:
        meta["empty_candidates"] = True
        return None, meta
    c0 = cands[0] if isinstance(cands[0], dict) else {}
    meta["finish_reason"] = c0.get("finishReason")
    parts = (c0.get("content") or {}).get("parts") or []
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            chunks.append(p["text"])
    raw = "".join(chunks).strip() or None
    return raw, meta


class GeminiLlmClient:
    @property
    def remote_configured(self) -> bool:
        s = get_settings()
        if s.llm_provider in ("disabled", "local"):
            return False
        return bool(s.gemini_api_key)

    @property
    def remote_enabled(self) -> bool:
        return self.remote_configured

    def extract_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        payload, _ = self.extract_json_with_trace(system_prompt, user_prompt)
        return payload

    def extract_json_with_trace(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_output_tokens: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        get_settings.cache_clear()
        attempts: list[dict[str, Any]] = []
        s = get_settings()

        if s.llm_provider == "disabled":
            return None, _llm_trace(
                enabled=False,
                used_provider=None,
                used_model=None,
                attempts=[],
                status="disabled",
            )

        max_tok = max_output_tokens if max_output_tokens is not None else s.llm_max_output_tokens
        model = (s.llm_model or s.gemini_model).removeprefix("models/")

        def _local_fallback() -> tuple[dict[str, Any] | None, dict[str, Any]]:
            lp, lt = local_intent_model.extract_json_with_trace(user_prompt)
            attempts.append(
                {
                    "provider": "local",
                    "model": lt.get("model") or "local-intent-v1",
                    "status": "ok" if lp else "no_candidate",
                    "parsed": bool(lp),
                }
            )
            if lp:
                return lp, _llm_trace(
                    enabled=True,
                    used_provider="local",
                    used_model=lt.get("model"),
                    attempts=attempts,
                    status="ok",
                )
            return None, _llm_trace(
                enabled=True,
                used_provider=None,
                used_model=None,
                attempts=attempts,
                status="fallback",
            )

        if s.llm_provider == "local" or not self.remote_configured:
            return _local_fallback()

        url = f"{GEMINI_API_ROOT.rstrip('/')}/models/{model}:generateContent"
        params = {"key": s.gemini_api_key}
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tok,
                "responseMimeType": "application/json",
            },
        }

        try:
            with httpx.Client(timeout=GEMINI_TIMEOUT) as client:
                r = client.post(url, params=params, json=body)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as exc:
            attempts.append(
                {
                    "provider": "gemini",
                    "model": model,
                    "status": "error",
                    "error": f"HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                }
            )
            logger.warning("Gemini HTTP error: status=%s", exc.response.status_code)
            return _local_fallback()
        except Exception as exc:
            attempts.append(
                {"provider": "gemini", "model": model, "status": "error", "error": str(exc)},
            )
            logger.warning("Gemini request failed: %s", type(exc).__name__)
            return _local_fallback()

        if isinstance(data.get("error"), dict):
            err = data["error"]
            attempts.append(
                {
                    "provider": "gemini",
                    "model": model,
                    "status": "error",
                    "error": str(err.get("message") or err.get("status") or err),
                }
            )
            return _local_fallback()

        raw, resp_meta = _gemini_response_text(data)
        if not raw:
            inv_empty: dict[str, Any] = {
                "provider": "gemini",
                "model": model,
                "status": "invalid_json",
                "finish_reason": resp_meta.get("finish_reason"),
                "raw_chars": 0,
            }
            if resp_meta.get("block_reason"):
                inv_empty["block_reason"] = resp_meta["block_reason"]
            if resp_meta.get("empty_candidates"):
                inv_empty["empty_candidates"] = True
            attempts.append(inv_empty)
            return _local_fallback()

        parsed = _parse_json_payload(raw)
        if parsed is not None:
            attempts.append({"provider": "gemini", "model": model, "status": "ok", "parsed": True})
            return (
                parsed,
                _llm_trace(
                    enabled=True,
                    used_provider="gemini",
                    used_model=model,
                    attempts=attempts,
                    status="ok",
                ),
            )

        inv_bad: dict[str, Any] = {
            "provider": "gemini",
            "model": model,
            "status": "invalid_json",
            "finish_reason": resp_meta.get("finish_reason"),
            "raw_preview": raw[:480],
            "raw_chars": len(raw),
        }
        if resp_meta.get("block_reason"):
            inv_bad["block_reason"] = resp_meta["block_reason"]
        attempts.append(inv_bad)
        return _local_fallback()


gemini_llm = GeminiLlmClient()
