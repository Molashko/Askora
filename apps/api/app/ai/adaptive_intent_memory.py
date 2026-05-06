from __future__ import annotations

from datetime import datetime, UTC
import json
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import settings


class AdaptiveIntentMemory:
    def __init__(self) -> None:
        self._lock = Lock()

    def record(
        self,
        *,
        question: str,
        payload: dict[str, Any],
        source: str,
        outcome: str,
    ) -> bool:
        if not settings.adaptive_intent_auto_learn_enabled:
            return False

        normalized_question = self._normalize(question)
        if len(normalized_question) < 3:
            return False

        path = self._resolve_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        clean_payload = {key: value for key, value in payload.items() if key != "question"}
        entry = {
            "question": question.strip(),
            "payload": clean_payload,
            "source": source,
            "outcome": outcome,
            "created_at": datetime.now(UTC).isoformat(),
        }

        with self._lock:
            data = self._read_payload(path)
            entries = data.get("entries", [])
            if not isinstance(entries, list):
                entries = []

            kept_entries: list[dict[str, Any]] = []
            replaced = False
            for item in entries:
                if not isinstance(item, dict):
                    continue
                existing_question = item.get("question")
                if isinstance(existing_question, str) and self._normalize(existing_question) == normalized_question:
                    if not replaced:
                        kept_entries.append(entry)
                        replaced = True
                    continue
                kept_entries.append(item)

            if not replaced:
                kept_entries.append(entry)

            max_entries = max(1, settings.adaptive_intent_max_entries)
            payload_to_write = {
                "version": 1,
                "name": "askora-adaptive-intent-memory",
                "description": "Examples learned from successful AI-assisted intent recovery.",
                "entries": kept_entries[-max_entries:],
            }
            path.write_text(json.dumps(payload_to_write, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    def _resolve_path(self) -> Path:
        path = Path(settings.adaptive_intent_memory_path)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    def _read_payload(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().replace("ё", "е").split())


adaptive_intent_memory = AdaptiveIntentMemory()
