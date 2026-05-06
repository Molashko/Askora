"""
Прогон пакета сложных NL-запросов (complex_training_batch_cases):
- выполняет реальные запросы через QueryService;
- при успешном executed и effective_source=hybrid_llm_fallback пишет adaptive_intent_memory
  и перезагружает локальную модель в процессе;
- по флагу --rebuild-local-model пересобирает статический local_intent_model.json из регрессии.

Перед любыми импортами app: при --force-llm-fallback выставляется порог низкой уверенности,
чтобы чаще вызывался внешний LLM (нужен GEMINI_API_KEY и LLM_PROVIDER=gemini).
"""
from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter

if "--force-llm-fallback" in sys.argv:
    os.environ.setdefault("ADAPTIVE_INTENT_LOW_CONFIDENCE_THRESHOLD", "0.99")

from app.db.session import SessionLocal
from app.repositories.users import UserRepository
from app.schemas.query import QueryRequest
from app.scripts.query_regression import complex_training_batch_cases
from app.services.query_service import QueryService


def main(*, rebuild_local_model: bool, force_llm_fallback: bool) -> int:
    cases = complex_training_batch_cases()
    db = SessionLocal()
    try:
        user = UserRepository(db).get_by_email("business@demo.local")
        if not user:
            raise RuntimeError("Нужен пользователь business@demo.local (демо-сид).")

        service = QueryService(db)
        print(f"Кейсов в пакете: {len(cases)}")
        if force_llm_fallback:
            print("Режим --force-llm-fallback: ADAPTIVE_INTENT_LOW_CONFIDENCE_THRESHOLD=0.99 (через env до импортов).\n")
        else:
            print(
                "Без --force-llm-fallback локальная модель часто достаточна → LLM не вызывается "
                "→ adaptive memory не пополняется. Добавьте флаг при наличии ключа LLM.\n"
            )

        hybrid_hits = 0
        adaptive_recorded = 0
        executed = 0

        for i, case in enumerate(cases, start=1):
            started = perf_counter()
            result = service.run(QueryRequest(question=case.question, dry_run=False), user)
            ms = round((perf_counter() - started) * 1000, 1)

            extraction = result.processing_trace.get("extraction")
            extraction = extraction if isinstance(extraction, dict) else {}
            effective = extraction.get("effective_source", "?")
            llm_block = extraction.get("llm") if isinstance(extraction.get("llm"), dict) else {}
            llm_status = llm_block.get("status", "-")

            if effective == "hybrid_llm_fallback":
                hybrid_hits += 1

            adaptive = result.processing_trace.get("adaptive_learning")
            adaptive = adaptive if isinstance(adaptive, dict) else {}
            rec = adaptive.get("recorded")
            if rec:
                adaptive_recorded += 1

            if result.status == "executed":
                executed += 1

            short_q = case.question if len(case.question) <= 72 else case.question[:69] + "..."
            print(f"[{i:02d}] {ms:>6.1f} ms | {result.status:<20} | src={effective:<22} | llm={llm_status!s} | adaptive={rec!s}")
            print(f"      {short_q}\n")

        print("---")
        print(f"Выполнено (executed): {executed}/{len(cases)}")
        print(f"С эффективным hybrid_llm_fallback: {hybrid_hits}")
        print(f"Записей в adaptive memory (recorded=True): {adaptive_recorded}")
        print("")
        print(
            "Adaptive memory: только при executed и effective_source=hybrid_llm_fallback "
            "(интент принят после вызова внешнего LLM)."
        )

        if rebuild_local_model:
            from app.scripts.build_local_intent_model import run as rebuild_run

            print("\nПересборка local_intent_model.json из регрессии + hard benchmark...")
            rebuild_run()
        else:
            print("\nЧтобы пересобрать статическую локальную модель: python -m app.scripts.build_local_intent_model")
            print("Или запустите этот скрипт с флагом --rebuild-local-model")

        return 0
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Прогон сложного пакета + опциональная пересборка локальной модели.")
    parser.add_argument(
        "--rebuild-local-model",
        action="store_true",
        help="После прогона вызвать build_local_intent_model (парафразы и hard benchmark в JSON).",
    )
    parser.add_argument(
        "--force-llm-fallback",
        action="store_true",
        help="Понизить порог уверенности (до импорта app), чтобы чаще вызывался внешний LLM и писалась adaptive memory.",
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            rebuild_local_model=args.rebuild_local_model,
            force_llm_fallback=args.force_llm_fallback,
        )
    )
