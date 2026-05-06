import json
import tempfile
import unittest
from pathlib import Path

from app.ai.adaptive_intent_memory import adaptive_intent_memory
from app.ai.local_intent_model import local_intent_model
from app.core.config import settings


class TestAdaptiveIntentMemory(unittest.TestCase):
    def test_recorded_example_is_loaded_by_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_model_path = Path(temp_dir) / "local_model.json"
            adaptive_memory_path = Path(temp_dir) / "adaptive_memory.json"
            base_model_path.write_text(
                json.dumps({"version": 1, "entries": []}, ensure_ascii=False),
                encoding="utf-8",
            )

            original_model_path = settings.local_intent_model_path
            original_memory_path = settings.adaptive_intent_memory_path
            original_auto_learn = settings.adaptive_intent_auto_learn_enabled
            try:
                settings.local_intent_model_path = str(base_model_path)
                settings.adaptive_intent_memory_path = str(adaptive_memory_path)
                settings.adaptive_intent_auto_learn_enabled = True

                recorded = adaptive_intent_memory.record(
                    question="Покажи странный новый запрос по выручке",
                    payload={
                        "intent_type": "aggregation",
                        "metrics": ["total_revenue"],
                        "dimensions": [],
                        "filters": [],
                        "time_expression": "за вчера",
                        "time_range_override": None,
                        "multi_date": None,
                        "comparison": {"enabled": False, "mode": "none", "baseline_label": None},
                        "preferred_chart_type": None,
                        "sort": None,
                        "limit": 50,
                        "confidence": 0.88,
                        "ambiguity_reasons": [],
                        "clarification_questions": [],
                        "notes": ["learned"],
                    },
                    source="hybrid_llm_fallback",
                    outcome="executed",
                )
                self.assertTrue(recorded)

                local_intent_model.reload()
                payload, trace = local_intent_model.extract_json_with_trace("Покажи странный новый запрос по выручке")

                self.assertIsNotNone(payload)
                self.assertEqual(payload["metrics"], ["total_revenue"])
                self.assertEqual(payload["time_expression"], "за вчера")
                self.assertEqual(trace["status"], "ok")
            finally:
                settings.local_intent_model_path = original_model_path
                settings.adaptive_intent_memory_path = original_memory_path
                settings.adaptive_intent_auto_learn_enabled = original_auto_learn
                local_intent_model.reload()


if __name__ == "__main__":
    unittest.main()
