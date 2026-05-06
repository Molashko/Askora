from __future__ import annotations

import json
from datetime import date
from typing import Any, Literal

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.ai.adaptive_intent_memory import adaptive_intent_memory
from app.ai.gemini_llm import gemini_llm
from app.ai.percent_change import is_percent_change_request
from app.ai.extractor import HybridIntentExtractor
from app.ai.local_intent_model import local_intent_model
from app.core.config import settings
from app.core.privacy import redact_payload
from app.data_sources.registry import data_source_registry
from app.models.report import QueryHistory, QueryStatus, UserQueryExample
from app.models.user import User
from app.query_engine.executor import query_executor
from app.query_engine.sql_builder import sql_builder
from app.repositories.reports import ReportRepository
from app.schemas.common import MessageResponse
from app.schemas.query import (
    QueryExampleCreateRequest,
    QueryRequest,
    QueryResult,
    TrustBadge,
    TrustOverlay,
    ValidationResult,
)
from app.semantic_layer.planner import VisualizationPlanner
from app.semantic_layer.resolver import SemanticResolver
from app.services.audit_service import AuditService
from app.services.metrics_service import metrics_service
from app.services.query_review_service import QueryReviewResult, QueryReviewService
from app.services.sql_review_service import SQLReviewService
from app.sql_guardrails.validator import sql_validator

_TRUST_GEMINI_RECHECK_THRESHOLD_AUTO = 89


class QueryService:
    def __init__(self, db: Session):
        self.db = db
        self.extractor = HybridIntentExtractor(db)
        self.reviewer = QueryReviewService(db)
        self.sql_reviewer = SQLReviewService(db)
        self.resolver = SemanticResolver(db)
        self.visualization = VisualizationPlanner()
        self.audit = AuditService(db)
        self.repo = ReportRepository(db)

    def run(self, payload: QueryRequest, user: User) -> QueryResult:
        execution_anchor = date.today() if payload.execution_context == "schedule" else None
        qmode = payload.query_mode
        intent, extraction_trace = self.extractor.extract_with_trace(payload.question, query_mode=qmode)
        query_plan = self.resolver.resolve(intent, user.role.value, anchor_date=execution_anchor)
        processing_trace: dict[str, object] = {
            "query_mode": qmode,
            "extraction": extraction_trace,
            "resolved_plan": self._summarize_plan(query_plan),
        }

        review = (
            QueryReviewResult(adjusted=False, intent=intent, notes=[])
            if qmode == "fast"
            else self.reviewer.review(payload.question, intent, query_plan)
        )
        processing_trace["intent_review"] = {
            "adjusted": review.adjusted,
            "notes": review.notes,
        }
        if review.adjusted:
            intent = review.intent
            query_plan = self.resolver.resolve(review.intent, user.role.value, anchor_date=execution_anchor)
            processing_trace["resolved_plan_after_review"] = self._summarize_plan(query_plan)
            self.audit.log(
                actor_user_id=user.id,
                event_type="query_reconciled",
                status="success",
                question=payload.question,
                interpretation_json=self._to_json(query_plan.model_dump(mode="json")),
                extra_json={"review_notes": review.notes, "trace": processing_trace},
            )

        visualization = self.visualization.choose(query_plan)
        processing_trace["visualization"] = visualization.model_dump(mode="json")
        source = data_source_registry.get_source(self.db, dataset_key=query_plan.dataset)

        if query_plan.needs_clarification:
            validation = ValidationResult(
                allowed=False,
                normalized_sql="",
                complexity_score=0,
                row_limit_applied=0,
                warnings=[],
                blocked_reasons=["Запрос требует уточнения."],
            )
            self._persist_history(
                user=user,
                payload=payload,
                query_plan=query_plan,
                sql_text="",
                validation_json=validation.model_dump(mode="json"),
                result_preview_json={},
                status=QueryStatus.needs_clarification,
                row_count=0,
                chart_type=visualization.chart_type,
            )
            self.audit.log(
                actor_user_id=user.id,
                event_type="query_interpreted",
                status="needs_clarification",
                question=payload.question,
                interpretation_json=self._to_json(query_plan.model_dump(mode="json")),
                extra_json={"trace": processing_trace},
            )
            self._track_query_outcome("needs_clarification", validation.blocked_reasons)
            self._remember_ai_assisted_intent(
                question=payload.question,
                intent=intent,
                outcome="needs_clarification",
                processing_trace=processing_trace,
            )
            return QueryResult(
                question=payload.question,
                query_plan=query_plan,
                generated_sql="",
                validation=validation,
                visualization=visualization,
                columns=[],
                rows=[],
                row_count=0,
                status="needs_clarification",
                user_message="Запрос требует уточнения. Система не стала выполнять его автоматически.",
                suggestions=query_plan.clarification_questions or query_plan.warnings,
                trust_overlay=self._finalize_trust_overlay(
                    question=payload.question,
                    query_plan=query_plan,
                    validation=validation,
                    processing_trace=processing_trace,
                    result_status="needs_clarification",
                ),
                processing_trace=processing_trace,
            )

        if source.allowed_roles and user.role.value not in source.allowed_roles:
            validation = ValidationResult(
                allowed=False,
                normalized_sql="",
                complexity_score=0,
                row_limit_applied=query_plan.limit,
                warnings=[],
                blocked_reasons=[f"У роли {user.role.value} нет доступа к источнику данных {source.name}."],
            )
            self._persist_history(
                user=user,
                payload=payload,
                query_plan=query_plan,
                sql_text="",
                validation_json=validation.model_dump(mode="json"),
                result_preview_json={},
                status=QueryStatus.blocked,
                row_count=0,
                chart_type=visualization.chart_type,
            )
            self.audit.log(
                actor_user_id=user.id,
                event_type="query_blocked",
                status="blocked",
                question=payload.question,
                blocked_reason=validation.blocked_reasons[0],
                interpretation_json=self._to_json(query_plan.model_dump(mode="json")),
                validation_json=self._to_json(validation.model_dump(mode="json")),
                extra_json={"trace": processing_trace},
            )
            self._track_query_outcome("blocked", validation.blocked_reasons)
            return QueryResult(
                question=payload.question,
                query_plan=query_plan,
                generated_sql="",
                validation=validation,
                visualization=visualization,
                columns=[],
                rows=[],
                row_count=0,
                status="blocked",
                user_message="Запрос заблокирован политикой доступа к источнику данных.",
                suggestions=validation.blocked_reasons,
                trust_overlay=self._finalize_trust_overlay(
                    question=payload.question,
                    query_plan=query_plan,
                    validation=validation,
                    processing_trace=processing_trace,
                    result_status="blocked",
                ),
                processing_trace=processing_trace,
            )

        sql_builder.catalog = self.resolver.catalog
        sql_validator.catalog = self.resolver.catalog
        sql_text, params = sql_builder.build(query_plan)
        processing_trace["sql_builder"] = {
            "sql_preview": sql_text,
            "params": self._to_json(params),
        }

        sql_review = self.sql_reviewer.review(
            question=payload.question,
            query_plan=query_plan,
            sql_text=sql_text,
            params=params,
        )
        processing_trace["sql_review"] = {
            "allowed": sql_review.allowed,
            "needs_clarification": sql_review.needs_clarification,
            "blocked_reasons": sql_review.blocked_reasons,
            "notes": sql_review.notes,
        }

        if sql_review.notes:
            query_plan.warnings = self._dedupe(query_plan.warnings + sql_review.notes)
            query_plan.confidence = max(0.1, round(query_plan.confidence - min(0.03 * len(sql_review.notes), 0.12), 2))

        if not sql_review.allowed:
            query_plan.confidence = min(query_plan.confidence, 0.55)
            query_plan.needs_clarification = sql_review.needs_clarification or query_plan.needs_clarification
            query_plan.clarification_questions = self._dedupe(query_plan.clarification_questions + sql_review.blocked_reasons)
            query_plan.warnings = self._dedupe(query_plan.warnings + sql_review.blocked_reasons)
            validation = ValidationResult(
                allowed=False,
                normalized_sql=sql_text,
                complexity_score=0,
                row_limit_applied=query_plan.limit,
                warnings=[],
                blocked_reasons=sql_review.blocked_reasons,
            )
            result_status = QueryStatus.needs_clarification if sql_review.needs_clarification else QueryStatus.blocked
            self._persist_history(
                user=user,
                payload=payload,
                query_plan=query_plan,
                sql_text=sql_text,
                validation_json=validation.model_dump(mode="json"),
                result_preview_json={},
                status=result_status,
                row_count=0,
                chart_type=visualization.chart_type,
            )
            self.audit.log(
                actor_user_id=user.id,
                event_type="query_alignment_blocked",
                status=result_status.value,
                question=payload.question,
                sql_text=sql_text,
                blocked_reason="; ".join(sql_review.blocked_reasons),
                interpretation_json=self._to_json(query_plan.model_dump(mode="json")),
                validation_json=self._to_json(validation.model_dump(mode="json")),
                extra_json={"review_notes": sql_review.notes, "trace": processing_trace},
            )
            self._track_query_outcome(
                "needs_clarification" if sql_review.needs_clarification else "blocked",
                sql_review.blocked_reasons,
            )
            if sql_review.needs_clarification:
                self._remember_ai_assisted_intent(
                    question=payload.question,
                    intent=intent,
                    outcome="needs_clarification",
                    processing_trace=processing_trace,
                )
            return QueryResult(
                question=payload.question,
                query_plan=query_plan,
                generated_sql=sql_text,
                validation=validation,
                visualization=visualization,
                columns=[],
                rows=[],
                row_count=0,
                status="needs_clarification" if sql_review.needs_clarification else "blocked",
                user_message=(
                    "Система остановила выполнение после финальной сверки смысла запроса и SQL."
                    if sql_review.needs_clarification
                    else "Запрос заблокирован на этапе финальной сверки."
                ),
                suggestions=sql_review.blocked_reasons,
                trust_overlay=self._finalize_trust_overlay(
                    question=payload.question,
                    query_plan=query_plan,
                    validation=validation,
                    processing_trace=processing_trace,
                    result_status="needs_clarification" if sql_review.needs_clarification else "blocked",
                ),
                processing_trace=processing_trace,
            )

        validation = sql_validator.validate(sql_text, query_plan.limit, query_plan.dataset, dialect=source.dialect)
        processing_trace["guardrails"] = validation.model_dump(mode="json")

        if validation.warnings:
            query_plan.confidence = max(0.1, round(query_plan.confidence - min(0.02 * len(validation.warnings), 0.08), 2))

        if not validation.allowed:
            query_plan.confidence = min(query_plan.confidence, 0.45)
            self._persist_history(
                user=user,
                payload=payload,
                query_plan=query_plan,
                sql_text=validation.normalized_sql,
                validation_json=validation.model_dump(mode="json"),
                result_preview_json={},
                status=QueryStatus.blocked,
                row_count=0,
                chart_type=visualization.chart_type,
            )
            self.audit.log(
                actor_user_id=user.id,
                event_type="query_blocked",
                status="blocked",
                question=payload.question,
                sql_text=validation.normalized_sql,
                blocked_reason="; ".join(validation.blocked_reasons),
                interpretation_json=self._to_json(query_plan.model_dump(mode="json")),
                validation_json=self._to_json(validation.model_dump(mode="json")),
                extra_json={"trace": processing_trace},
            )
            self._track_query_outcome("blocked", validation.blocked_reasons)
            return QueryResult(
                question=payload.question,
                query_plan=query_plan,
                generated_sql=validation.normalized_sql,
                validation=validation,
                visualization=visualization,
                columns=[],
                rows=[],
                row_count=0,
                status="blocked",
                user_message="Запрос заблокирован guardrails и не был выполнен.",
                suggestions=validation.blocked_reasons,
                trust_overlay=self._finalize_trust_overlay(
                    question=payload.question,
                    query_plan=query_plan,
                    validation=validation,
                    processing_trace=processing_trace,
                    result_status="blocked",
                ),
                processing_trace=processing_trace,
            )

        explain_plan = self._safe_explain_plan(validation.normalized_sql, params, query_plan.dataset)
        if explain_plan:
            estimated_cost, estimated_rows = self._extract_plan_estimates(explain_plan)
            validation.explain_plan_json = explain_plan
            validation.estimated_cost = estimated_cost
            validation.estimated_rows = estimated_rows
            processing_trace["explain_plan"] = {
                "estimated_cost": estimated_cost,
                "estimated_rows": estimated_rows,
            }
            if (
                settings.max_query_cost > 0
                and estimated_cost is not None
                and estimated_cost > settings.max_query_cost
            ):
                validation.allowed = False
                validation.blocked_reasons = self._dedupe(
                    validation.blocked_reasons
                    + [
                        (
                            "Запрос отклонён до выполнения: прогнозируемая стоимость "
                            f"{estimated_cost:.2f} превышает лимит {settings.max_query_cost:.2f}."
                        )
                    ]
                )
                query_plan.confidence = min(query_plan.confidence, 0.5)
                self._persist_history(
                    user=user,
                    payload=payload,
                    query_plan=query_plan,
                    sql_text=validation.normalized_sql,
                    validation_json=validation.model_dump(mode="json"),
                    result_preview_json={},
                    status=QueryStatus.blocked,
                    row_count=0,
                    chart_type=visualization.chart_type,
                )
                self.audit.log(
                    actor_user_id=user.id,
                    event_type="query_blocked",
                    status="blocked",
                    question=payload.question,
                    sql_text=validation.normalized_sql,
                    blocked_reason="; ".join(validation.blocked_reasons),
                    interpretation_json=self._to_json(query_plan.model_dump(mode="json")),
                    validation_json=self._to_json(validation.model_dump(mode="json")),
                    extra_json={"trace": processing_trace},
                )
                self._track_query_outcome("blocked", validation.blocked_reasons)
                return QueryResult(
                    question=payload.question,
                    query_plan=query_plan,
                    generated_sql=validation.normalized_sql,
                    validation=validation,
                    visualization=visualization,
                    columns=[],
                    rows=[],
                    row_count=0,
                    status="blocked",
                    user_message="Запрос заблокирован по прогнозной стоимости до запуска в БД.",
                    suggestions=validation.blocked_reasons,
                    trust_overlay=self._finalize_trust_overlay(
                        question=payload.question,
                        query_plan=query_plan,
                        validation=validation,
                        processing_trace=processing_trace,
                        result_status="blocked",
                    ),
                    processing_trace=processing_trace,
                )

        columns: list[str] = []
        rows: list[dict] = []
        row_count = 0
        status = QueryStatus.executed
        user_message = "Запрос успешно выполнен."

        try:
            if payload.dry_run:
                user_message = "SQL успешно прошел проверку и не был выполнен, потому что выбран режим dry-run."
            else:
                columns, rows, row_count = query_executor.execute(
                    self.db,
                    validation.normalized_sql,
                    params,
                    dataset_key=query_plan.dataset,
                )
        except Exception as exc:
            self.db.rollback()
            status = QueryStatus.failed
            user_message = f"Не удалось выполнить запрос: {exc}"

        processing_trace["execution"] = {
            "status": status.value,
            "row_count": row_count,
            "dry_run": payload.dry_run,
            "columns": columns,
        }

        comparison_summary = self._build_comparison_summary(rows, query_plan)
        preview = redact_payload(self._to_json({"rows": rows[:10], "columns": columns}))
        self._persist_history(
            user=user,
            payload=payload,
            query_plan=query_plan,
            sql_text=validation.normalized_sql,
            validation_json=validation.model_dump(mode="json"),
            result_preview_json=preview,
            status=status,
            row_count=row_count,
            chart_type=visualization.chart_type,
        )
        self.audit.log(
            actor_user_id=user.id,
            event_type="query_executed" if status == QueryStatus.executed else "query_failed",
            status=status.value,
            question=payload.question,
            sql_text=validation.normalized_sql,
            row_count=row_count,
            interpretation_json=self._to_json(query_plan.model_dump(mode="json")),
            validation_json=self._to_json(validation.model_dump(mode="json")),
            extra_json={"preview": preview, "trace": processing_trace},
        )
        self._track_query_outcome("executed" if status == QueryStatus.executed else "failed", validation.blocked_reasons)
        hybrid_recorded = False
        extraction_tail = processing_trace.get("extraction")
        hybrid_source = isinstance(extraction_tail, dict) and extraction_tail.get("effective_source") == "hybrid_llm_fallback"
        if status == QueryStatus.executed and hybrid_source:
            hybrid_recorded = self._record_adaptive_hybrid_fallback(
                question=payload.question,
                intent=intent,
                outcome="executed",
            )
        trust_overlay = self._finalize_trust_overlay(
            question=payload.question,
            query_plan=query_plan,
            validation=validation,
            processing_trace=processing_trace,
            result_status="executed" if status == QueryStatus.executed else "failed",
        )
        gem_recorded = False
        if status == QueryStatus.executed:
            gem_recorded = self._record_adaptive_gemini_trust(
                question=payload.question,
                intent=intent,
                trust_overlay=trust_overlay,
                hybrid_llm_used=hybrid_source,
            )
        if hybrid_recorded or gem_recorded:
            local_intent_model.reload()
        learning_trace: dict[str, Any] = {}
        if hybrid_recorded:
            learning_trace["hybrid_recorded"] = True
        if gem_recorded:
            learning_trace["gemini_trust_recorded"] = True
        if learning_trace:
            processing_trace["adaptive_learning"] = learning_trace

        interp_prompt = None
        if status == QueryStatus.executed:
            interp_prompt = self._build_interpretation_confirmation_prompt(
                query_plan=query_plan,
                trust_overlay=trust_overlay,
                processing_trace=processing_trace,
            )

        return QueryResult(
            question=payload.question,
            query_plan=query_plan,
            generated_sql=validation.normalized_sql,
            validation=validation,
            visualization=visualization,
            columns=columns,
            rows=rows,
            row_count=row_count,
            status="executed" if status == QueryStatus.executed else "failed",
            user_message=user_message,
            suggestions=query_plan.warnings + validation.warnings,
            comparison_summary=comparison_summary,
            trust_overlay=trust_overlay,
            processing_trace=processing_trace,
            interpretation_confirmation_prompt=interp_prompt,
        )

    def list_history(self, user: User):
        return self.repo.list_query_history(user.id)

    def delete_history_item(self, history_id, user: User) -> bool:
        item = self.repo.get_history_item(history_id, user.id)
        if not item:
            return False
        self.repo.delete_history_item(item)
        self.audit.log(
            actor_user_id=user.id,
            event_type="history_deleted",
            status="success",
            question=item.question,
            sql_text=item.sql_text,
        )
        return True

    def clear_history(self, user: User) -> int:
        deleted = self.repo.clear_history(user.id)
        self.audit.log(
            actor_user_id=user.id,
            event_type="history_cleared",
            status="success",
            row_count=deleted,
        )
        return deleted

    def list_examples(self, user: User) -> list[UserQueryExample]:
        return self.repo.list_query_examples(user.id)

    def create_example(self, payload: QueryExampleCreateRequest, user: User) -> UserQueryExample:
        existing = self.repo.find_query_example(user.id, payload.text)
        if existing:
            existing.is_pinned = payload.is_pinned
            return self.repo.save_query_example(existing)
        return self.repo.create_query_example(
            UserQueryExample(
                user_id=user.id,
                text=payload.text.strip(),
                is_pinned=payload.is_pinned,
            )
        )

    def delete_example(self, example_id, user: User) -> bool:
        item = self.repo.get_query_example(example_id, user.id)
        if not item:
            return False
        self.repo.delete_query_example(item)
        return True

    def _persist_history(
        self,
        *,
        user: User,
        payload: QueryRequest,
        query_plan,
        sql_text: str,
        validation_json: dict,
        result_preview_json: dict,
        status: QueryStatus,
        row_count: int,
        chart_type: str | None,
    ) -> QueryHistory:
        history = QueryHistory(
            user_id=user.id,
            question=payload.question,
            query_plan_json=self._to_json(query_plan.model_dump(mode="json")),
            sql_text=sql_text,
            validation_json=redact_payload(self._to_json(validation_json)),
            result_preview_json=redact_payload(self._to_json(result_preview_json)),
            chart_type=chart_type,
            confidence=query_plan.confidence,
            status=status,
            row_count=row_count,
        )
        return self.repo.create_query_history(history)

    def _build_comparison_summary(self, rows: list[dict], query_plan) -> dict | None:
        if (
            not query_plan.comparison.enabled
            or not rows
            or not query_plan.metrics
            or "period_label" not in rows[0]
            or is_percent_change_request(query_plan.question)
        ):
            return None

        metric_key = query_plan.metrics[0].key
        grouped: dict[str, dict[str, float]] = {}
        for row in rows:
            label_key = next((dimension.key for dimension in query_plan.dimensions if dimension.key in row), None)
            label_value = row.get(label_key, "Итого") if label_key else "Итого"
            grouped.setdefault(str(label_value), {})
            grouped[str(label_value)][str(row.get("period_label", "Период"))] = float(row.get(metric_key, 0) or 0)

        cur_key = query_plan.time_range.label or "Текущий период"
        prev_key = query_plan.comparison.baseline_label or "Предыдущий период"
        summary = []
        for label, periods in grouped.items():
            current = periods.get(cur_key, 0)
            previous = periods.get(prev_key, 0)
            delta = current - previous
            delta_pct = round((delta / previous * 100), 2) if previous else None
            summary.append(
                {
                    "label": label,
                    "current": current,
                    "previous": previous,
                    "delta": round(delta, 2),
                    "delta_pct": delta_pct,
                }
            )
        return {"items": summary, "metric": query_plan.metrics[0].label}

    def _to_json(self, payload):
        return jsonable_encoder(payload)

    def _build_trust_overlay(
        self,
        *,
        query_plan,
        validation: ValidationResult,
        processing_trace: dict[str, object],
        result_status: str,
    ) -> TrustOverlay:
        extraction = processing_trace.get("extraction") if isinstance(processing_trace.get("extraction"), dict) else {}
        intent_review = processing_trace.get("intent_review") if isinstance(processing_trace.get("intent_review"), dict) else {}
        sql_review = processing_trace.get("sql_review") if isinstance(processing_trace.get("sql_review"), dict) else {}
        llm_trace = extraction.get("llm") if isinstance(extraction, dict) and isinstance(extraction.get("llm"), dict) else {}

        source = str(extraction.get("effective_source") or "unknown")
        source_labels = {
            "hybrid_local": "Локальная модель + правила",
            "hybrid_llm_fallback": "LLM fallback + правила",
            "local": "Локальная модель",
            "rule_based": "Правила и семантический слой",
            "unknown": "Источник не определён",
        }
        source_label = source_labels.get(source, source)
        if llm_trace.get("remote_llm_invoked") and str(llm_trace.get("used_provider")) == "local":
            model = str(llm_trace.get("used_model") or "")
            tail = f" ({model})" if model else ""
            source_label = f"{source_label} — вызов Gemini был, итоговый JSON взялся из локальной модели{tail}"
        elif llm_trace.get("used_provider"):
            provider = str(llm_trace.get("used_provider"))
            model = str(llm_trace.get("used_model") or "")
            source_label = f"{source_label} ({provider}{f' / {model}' if model else ''})"

        score_percent = max(0, min(100, round(query_plan.confidence * 100)))
        confidence_level = "high" if score_percent >= 80 else "medium" if score_percent >= 60 else "low"
        review_adjusted = bool(intent_review.get("adjusted"))
        sql_allowed = bool(sql_review.get("allowed", True))
        autocorrects = [
            item
            for item in query_plan.warnings
            if isinstance(item, str) and item.startswith("Похоже, имелось в виду")
        ]
        raw_cautions = self._dedupe(
            [
                *[item for item in query_plan.warnings if item not in autocorrects],
                *validation.warnings,
                *validation.blocked_reasons,
                *query_plan.clarification_questions,
            ]
        )
        cautions = [item for item in raw_cautions if not self._is_low_signal_trust_note(item)][:5]

        evidence: list[str] = [f"Источник интерпретации: {source_label}."]
        if review_adjusted:
            evidence.append("Intent review скорректировал исходную интерпретацию перед построением SQL.")
        else:
            evidence.append("Intent review не потребовал смысловой корректировки.")
        if sql_allowed:
            evidence.append("Финальная сверка смысла запроса и SQL не нашла критичных расхождений.")
        if validation.allowed:
            evidence.append("SQL прошёл guardrails по безопасности и стоимости.")
        if autocorrects:
            evidence.append(f"Автовосстановление опечаток сработало для {len(autocorrects)} фрагм.")

        needs_manual_review = any(
            [
                result_status != "executed",
                query_plan.needs_clarification,
                score_percent < 72,
                review_adjusted,
                not sql_allowed,
                not validation.allowed,
                bool(cautions),
            ]
        )

        if result_status == "executed" and not needs_manual_review:
            summary = "Интерпретация выглядит устойчивой: смысл запроса, SQL и guardrails согласованы."
        elif result_status == "executed":
            summary = "Результат получен, но есть сигналы, которые стоит просмотреть вручную."
        elif result_status == "needs_clarification":
            summary = "Система осознанно остановилась до выполнения, потому что доверие к автоматической интерпретации недостаточно."
        elif result_status == "blocked":
            summary = "Интерпретация построена, но выполнение остановлено правилами безопасности или финальной сверкой."
        else:
            summary = "Во время выполнения возникла ошибка, поэтому доверять результату нельзя без повторной проверки."

        overlay = TrustOverlay(
            score_percent=score_percent,
            confidence_level=confidence_level,
            summary=summary,
            source=source,
            source_label=source_label,
            needs_manual_review=needs_manual_review,
            badges=[
                TrustBadge(
                    label="Уверенность",
                    value=f"{score_percent}%",
                    tone="success" if confidence_level == "high" else "warning" if confidence_level == "medium" else "danger",
                ),
                TrustBadge(
                    label="Intent review",
                    value="С корректировкой" if review_adjusted else "Без корректировки",
                    tone="warning" if review_adjusted else "success",
                ),
                TrustBadge(
                    label="SQL review",
                    value="Пройден" if sql_allowed else "Не пройден",
                    tone="success" if sql_allowed else "danger",
                ),
                TrustBadge(
                    label="Guardrails",
                    value="Пройдены" if validation.allowed else "Остановили запуск",
                    tone="success" if validation.allowed else "danger",
                ),
            ],
            evidence=evidence[:4],
            cautions=cautions,
            auto_corrections=autocorrects[:3],
        )
        processing_trace["trust_overlay"] = overlay.model_dump(mode="json")
        return overlay

    def _finalize_trust_overlay(
        self,
        *,
        question: str,
        query_plan,
        validation: ValidationResult,
        processing_trace: dict[str, object],
        result_status: str,
    ) -> TrustOverlay:
        overlay = self._build_trust_overlay(
            query_plan=query_plan,
            validation=validation,
            processing_trace=processing_trace,
            result_status=result_status,
        )
        overlay = self._maybe_enrich_trust_overlay_with_gemini(
            question=question,
            overlay=overlay,
            query_plan=query_plan,
            validation=validation,
            processing_trace=processing_trace,
            result_status=result_status,
        )
        processing_trace["trust_overlay"] = overlay.model_dump(mode="json")
        return overlay

    @staticmethod
    def _trust_confidence_level_for_score(score_percent: int) -> Literal["high", "medium", "low"]:
        if score_percent >= 80:
            return "high"
        if score_percent >= 60:
            return "medium"
        return "low"

    def _trust_overlay_full_mode_gemini_unavailable(
        self,
        overlay: TrustOverlay,
        *,
        detail: str,
        processing_trace: dict[str, object],
    ) -> TrustOverlay:
        """Комплексный режим без ответа Gemini: не меняем score, только предупреждение."""
        detail_short = (detail or "нет ответа").strip()[:400]
        msg = (
            "Режим «Комплексный» сейчас не выполнил дополнительную проверку: сервис (Gemini/LLM) недоступен "
            f"({detail_short}). Выберите «Автоматический» или «Быстрый», либо повторите позже."
        )
        processing_trace["trust_gemini_recheck"] = {
            "status": "full_mode_unavailable",
            "detail": detail_short,
        }
        cautions = self._dedupe([msg, *list(overlay.cautions)])
        evidence = self._dedupe(
            ["Комплексный режим: дополнительная проверка не получена — оценка доверия без пересчёта.", *list(overlay.evidence)]
        )
        badges = [
            *list(overlay.badges),
            TrustBadge(label="Комплексный режим", value="Проверка недоступна", tone="warning"),
        ]
        return overlay.model_copy(
            update={
                "cautions": cautions[:14],
                "evidence": evidence[:14],
                "badges": badges,
                "gemini_trust_second_pass": True,
                "trust_score_before_gemini": overlay.score_percent,
                "gemini_alignment_percent": None,
                "gemini_trust_verdict": "error",
                "gemini_trust_comment": detail_short,
                "summary": f"{overlay.summary} {msg}",
            },
        )

    @staticmethod
    def _trust_parse_failure_detail(llm_trace: dict[str, Any]) -> str:
        """Почему перепроверка доверия не получила JSON (для подписи пользователю и processing_trace)."""
        attempts = llm_trace.get("attempts")
        if not isinstance(attempts, list):
            return "пустой или невалидный ответ модели"
        for att in reversed(attempts):
            if not isinstance(att, dict) or att.get("status") != "invalid_json":
                continue
            fr = att.get("finish_reason")
            if fr in ("length", "MAX_TOKENS", "LENGTH"):
                return (
                    "ответ перепроверки обрезан по лимиту токенов; "
                    "повторите запрос или уменьшите объём SQL/подсказок в вопросе"
                )
            if att.get("refusal"):
                return f"модель отказалась ответить на перепроверку: {str(att.get('refusal'))[:300]}"
            if att.get("block_reason"):
                return f"запрос к Gemini заблокирован ({att.get('block_reason')})"
            if not att.get("raw_chars"):
                return "пустой ответ модели на перепроверку доверия (нет текста в ответе API)"
            return (
                "модель вернула текст, который не удалось разобрать как один JSON-объект с полями "
                "alignment_percent / verdict / comment (часто — лишний текст вне JSON или неверная структура)"
            )
        return "пустой или невалидный ответ модели"

    def _maybe_enrich_trust_overlay_with_gemini(
        self,
        *,
        question: str,
        overlay: TrustOverlay,
        query_plan,
        validation: ValidationResult,
        processing_trace: dict[str, object],
        result_status: str,
    ) -> TrustOverlay:
        qmode = str(processing_trace.get("query_mode") or "auto")
        if qmode == "fast":
            return overlay
        if qmode == "full":
            if not gemini_llm.remote_configured:
                return self._trust_overlay_full_mode_gemini_unavailable(
                    overlay,
                    detail="API отключён или не настроен",
                    processing_trace=processing_trace,
                )
        else:
            if overlay.score_percent >= _TRUST_GEMINI_RECHECK_THRESHOLD_AUTO:
                return overlay
            if not gemini_llm.remote_configured:
                return overlay

        context = {
            "question": question,
            "result_status": result_status,
            "trust_score_percent": overlay.score_percent,
            "query_mode": qmode,
            "summary": overlay.summary,
            "metrics": [m.label for m in query_plan.metrics],
            "dimensions": [d.label for d in query_plan.dimensions],
            "time_range": query_plan.time_range.model_dump(mode="json"),
            "comparison": query_plan.comparison.model_dump(mode="json"),
            "sql_preview": (validation.normalized_sql or "")[:2800],
            "cautions": list(overlay.cautions)[:10],
            "evidence": list(overlay.evidence)[:10],
        }
        if qmode == "full":
            system_prompt = (
                "Ты опытный аудитор NL2SQL: пользователь включил режим дополнительной проверки. "
                "Ответь одним JSON-объектом, без markdown и без текста вне JSON.\n"
                "Оцени свободно и честно: период и сравнение в вопросе, метрики, разрезы, фильтры, SQL и типичные "
                "упрощения NL2SQL. Можно учитывать нюансы формулировки на русском.\n"
                "Ключи:\n"
                "- alignment_percent: целое 0-100 — насколько по твоему мнению результат отвечает намерению пользователя;\n"
                "- verdict: consistent | uncertain | mismatch — как считаешь уместным;\n"
                "- comment: по-русски 2-6 предложений: что сходится, что спорно, что стоит уточнить;\n"
                "- extra_evidence: 0-6 строк — аргументы в пользу совпадения вопроса и SQL;\n"
                "- extra_cautions: 0-6 строк — где возможны недопонимания или пробелы;\n"
                "Не занижай оценку без причины и не завышай при явных рисках; uncertain — нормальный исход при тонких вопросах."
            )
        else:
            system_prompt = (
                "Ты аудитор NL2SQL. Ответь одним JSON-объектом, без markdown и без пояснений вне JSON.\n"
                "Ключи:\n"
                "- alignment_percent: целое 0-100 (насколько SQL и выбранные метрики/разрезы соответствуют вопросу);\n"
                "- verdict: одно из consistent | uncertain | mismatch;\n"
                "- comment: 1-2 коротких предложения по-русски;\n"
                "- extra_evidence: массив строк, 0-3 дополнительных аргумента за доверие;\n"
                "- extra_cautions: массив строк, 0-3 риска или несоответствия.\n"
                "Будь консервативен: mismatch только при явном логическом противоречии вопросу и SQL."
            )
        user_prompt = json.dumps(context, ensure_ascii=False)

        try:
            # Второй вызов LLM (аудит доверия): длинный контекст + развёрнутый JSON ответа — выше лимит токенов, чем у извлечения интента.
            trust_tokens = max(8192, int(settings.llm_max_output_tokens))
            payload, llm_trace = gemini_llm.extract_json_with_trace(
                system_prompt,
                user_prompt,
                max_output_tokens=trust_tokens,
            )
            processing_trace["trust_gemini_recheck"] = llm_trace
        except Exception as exc:  # pragma: no cover - сеть/API
            if qmode == "full":
                return self._trust_overlay_full_mode_gemini_unavailable(
                    overlay,
                    detail=f"ошибка вызова: {exc}",
                    processing_trace=processing_trace,
                )
            return overlay.model_copy(
                update={
                    "gemini_trust_second_pass": True,
                    "trust_score_before_gemini": overlay.score_percent,
                    "gemini_trust_verdict": "error",
                    "gemini_trust_comment": f"Ошибка перепроверки: {exc}",
                }
            )

        if not payload:
            if qmode == "full":
                detail_fail = self._trust_parse_failure_detail(llm_trace)
                return self._trust_overlay_full_mode_gemini_unavailable(
                    overlay,
                    detail=detail_fail,
                    processing_trace=processing_trace,
                )
            return overlay.model_copy(
                update={
                    "gemini_trust_second_pass": True,
                    "trust_score_before_gemini": overlay.score_percent,
                    "gemini_trust_verdict": "error",
                    "gemini_trust_comment": "LLM не вернул валидный JSON для перепроверки доверия.",
                }
            )

        raw_align = payload.get("alignment_percent")
        try:
            align = int(raw_align) if raw_align is not None else 0
        except (TypeError, ValueError):
            align = 0
        align = max(0, min(100, align))

        verdict = str(payload.get("verdict") or "uncertain").lower()
        if verdict not in {"consistent", "uncertain", "mismatch"}:
            verdict = "uncertain"

        baseline = overlay.score_percent
        if verdict == "consistent":
            new_score = max(baseline, align)
        elif verdict == "mismatch":
            new_score = min(baseline, align)
        else:
            new_score = min(100, max(baseline, (baseline + align) // 2))

        evidence = list(overlay.evidence)
        comment = payload.get("comment")
        if isinstance(comment, str) and comment.strip():
            evidence.insert(0, f"Gemini (перепроверка доверия): {comment.strip()}")
        for item in payload.get("extra_evidence") or []:
            if isinstance(item, str) and item.strip():
                evidence.append(item.strip())
        slice_cap = 16 if qmode == "full" else 10
        evidence = evidence[:slice_cap]

        cautions = list(overlay.cautions)
        for item in payload.get("extra_cautions") or []:
            if isinstance(item, str) and item.strip():
                cautions.append(item.strip())
        cautions = cautions[:slice_cap]

        badges = list(overlay.badges)
        tone: Literal["success", "warning", "danger", "neutral"] = (
            "success" if verdict == "consistent" else "danger" if verdict == "mismatch" else "warning"
        )
        gem_label = "Gemini: комплексный аудит" if qmode == "full" else "Gemini перепроверка"
        badges.append(
            TrustBadge(
                label=gem_label,
                value=f"{align}% · {verdict}",
                tone=tone,
            )
        )

        summary = overlay.summary
        if isinstance(comment, str) and comment.strip():
            summary = f"{summary} Перепроверка: {comment.strip()}"

        full_strict_review = qmode == "full" and verdict in ("uncertain", "mismatch")

        return TrustOverlay(
            score_percent=new_score,
            confidence_level=self._trust_confidence_level_for_score(new_score),
            summary=summary,
            source=overlay.source,
            source_label=overlay.source_label,
            needs_manual_review=overlay.needs_manual_review or verdict == "mismatch" or full_strict_review,
            badges=badges,
            evidence=evidence,
            cautions=cautions,
            auto_corrections=list(overlay.auto_corrections),
            gemini_trust_second_pass=True,
            trust_score_before_gemini=baseline,
            gemini_alignment_percent=align,
            gemini_trust_verdict=verdict,
            gemini_trust_comment=comment.strip() if isinstance(comment, str) else None,
        )

    def _is_low_signal_trust_note(self, note: str) -> bool:
        normalized = note.lower()
        return any(
            marker in normalized
            for marker in [
                "локальная переносимая модель классификации интента",
                "метрика выбрана по типу запроса",
                "период интерпретирован как",
            ]
        )

    def _record_adaptive_hybrid_fallback(self, *, question: str, intent, outcome: str) -> bool:
        return adaptive_intent_memory.record(
            question=question,
            payload=intent.model_dump(mode="json"),
            source="hybrid_llm_fallback",
            outcome=outcome,
        )

    def _record_adaptive_gemini_trust(
        self,
        *,
        question: str,
        intent,
        trust_overlay: TrustOverlay,
        hybrid_llm_used: bool,
    ) -> bool:
        if not trust_overlay.gemini_trust_second_pass:
            return False
        if trust_overlay.gemini_trust_verdict not in {"consistent", "uncertain"}:
            return False
        if hybrid_llm_used:
            # Уже записан полный intent из LLM; повторная запись тем же вопросом не даёт новой информации.
            return False
        return adaptive_intent_memory.record(
            question=question,
            payload=intent.model_dump(mode="json"),
            source="gemini_trust_recheck",
            outcome="executed",
        )

    def _remember_ai_assisted_intent(
        self,
        *,
        question: str,
        intent,
        outcome: str,
        processing_trace: dict[str, object],
    ) -> None:
        extraction = processing_trace.get("extraction")
        if not isinstance(extraction, dict):
            return
        if extraction.get("effective_source") != "hybrid_llm_fallback":
            return
        recorded = self._record_adaptive_hybrid_fallback(question=question, intent=intent, outcome=outcome)
        processing_trace["adaptive_learning"] = {
            "recorded": recorded,
            "outcome": outcome,
        }
        if recorded:
            local_intent_model.reload()

    def _build_interpretation_confirmation_prompt(
        self,
        *,
        query_plan,
        trust_overlay: TrustOverlay,
        processing_trace: dict[str, object],
    ) -> str | None:
        extraction = processing_trace.get("extraction")
        if not isinstance(extraction, dict):
            return None
        src = str(extraction.get("effective_source") or "")
        risky = (
            src == "hybrid_llm_fallback"
            or trust_overlay.gemini_trust_second_pass
            or trust_overlay.score_percent < 80
            or trust_overlay.needs_manual_review
        )
        if not risky:
            return None
        tr = query_plan.time_range
        metric_labels = ", ".join(m.label for m in query_plan.metrics[:6])
        dim_labels = ", ".join(d.label for d in query_plan.dimensions[:5])
        return (
            f"Период: «{tr.label}» ({tr.start_date} — {tr.end_date}). "
            f"Метрики: {metric_labels or '—'}. Разрезы: {dim_labels or '—'}. "
            "Это совпадает с тем, что вы хотели увидеть?"
        )

    @staticmethod
    def _intent_payload_from_resolved_plan(plan: dict[str, Any]) -> dict[str, Any]:
        metrics = [m["key"] for m in plan.get("metrics", []) if isinstance(m, dict) and m.get("key")]
        dimensions = [d["key"] for d in plan.get("dimensions", []) if isinstance(d, dict) and d.get("key")]
        filters_out: list[dict[str, Any]] = []
        for item in plan.get("filters", []) or []:
            if isinstance(item, dict) and item.get("key") is not None and item.get("operator") is not None:
                filters_out.append({"key": item["key"], "operator": item["operator"], "value": item.get("value")})
        time_range = plan.get("time_range")
        comparison = plan.get("comparison")
        if not isinstance(comparison, dict):
            comparison = {"enabled": False, "mode": "none", "baseline_label": None, "baseline_start_date": None, "baseline_end_date": None}
        return {
            "intent_type": plan.get("intent_type") or "aggregation",
            "metrics": metrics,
            "dimensions": dimensions,
            "filters": filters_out,
            "time_expression": None,
            "time_range_override": time_range if isinstance(time_range, dict) else None,
            "multi_date": plan.get("multi_date"),
            "comparison": comparison,
            "preferred_chart_type": plan.get("preferred_chart_type"),
            "sort": plan.get("sort"),
            "limit": plan.get("limit", 50),
            "confidence": float(plan.get("confidence") or 0.75),
            "ambiguity_reasons": [],
            "clarification_questions": [],
            "notes": [],
        }

    def record_interpretation_feedback(self, *, question: str, helpful: bool, user: User) -> MessageResponse:
        if not helpful:
            self.audit.log(
                actor_user_id=user.id,
                event_type="interpretation_feedback_negative",
                status="recorded",
                question=question,
                extra_json={"helpful": False},
            )
            return MessageResponse(message="Спасибо, учтём при улучшении интерпретации.")

        row = self.repo.get_latest_history_matching_question(user.id, question)
        if not row or not row.query_plan_json:
            raise LookupError("Не найдена недавняя запись истории с такой формулировкой вопроса.")

        plan = row.query_plan_json if isinstance(row.query_plan_json, dict) else {}
        payload = self._intent_payload_from_resolved_plan(plan)
        recorded = adaptive_intent_memory.record(
            question=question,
            payload=payload,
            source="user_confirmed",
            outcome="executed",
        )
        if recorded:
            local_intent_model.reload()
        self.audit.log(
            actor_user_id=user.id,
            event_type="interpretation_feedback_positive",
            status="recorded",
            question=question,
            interpretation_json=self._to_json(plan),
            extra_json={"helpful": True, "adaptive_recorded": recorded},
        )
        return MessageResponse(
            message="Сохранили подтверждённую интерпретацию для локальной модели." if recorded else "Подтверждение получено; автозапись отключена в настройках."
        )

    def _safe_explain_plan(self, sql_text: str, params: dict, dataset_key: str) -> dict | None:
        try:
            return query_executor.explain(self.db, sql_text, params, dataset_key=dataset_key)
        except Exception:
            self.db.rollback()
            return None

    def _extract_plan_estimates(self, explain_plan: dict) -> tuple[float | None, float | None]:
        plan_node = explain_plan.get("Plan") if isinstance(explain_plan, dict) else None
        if not isinstance(plan_node, dict):
            return None, None
        total_cost = plan_node.get("Total Cost")
        plan_rows = plan_node.get("Plan Rows")
        try:
            cast_cost = float(total_cost) if total_cost is not None else None
        except (TypeError, ValueError):
            cast_cost = None
        try:
            cast_rows = float(plan_rows) if plan_rows is not None else None
        except (TypeError, ValueError):
            cast_rows = None
        return cast_cost, cast_rows

    def _dedupe(self, items: list[str]) -> list[str]:
        result: list[str] = []
        for item in items:
            cleaned = item.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result

    def _summarize_plan(self, query_plan) -> dict[str, object]:
        return {
            "dataset": query_plan.dataset,
            "intent_type": query_plan.intent_type,
            "metrics": [metric.key for metric in query_plan.metrics],
            "dimensions": [dimension.key for dimension in query_plan.dimensions],
            "filters": [item.key for item in query_plan.filters],
            "time_range": query_plan.time_range.model_dump(mode="json"),
            "multi_date": query_plan.multi_date.model_dump(mode="json") if query_plan.multi_date else None,
            "comparison": query_plan.comparison.model_dump(mode="json"),
            "preferred_chart_type": query_plan.preferred_chart_type,
            "confidence": query_plan.confidence,
            "needs_clarification": query_plan.needs_clarification,
        }

    def _track_query_outcome(self, status: str, blocked_reasons: list[str]) -> None:
        metrics_service.observe_query_run(status)
        if status in {"blocked", "needs_clarification"}:
            reasons = blocked_reasons or ["unknown"]
            for reason in reasons:
                metrics_service.observe_query_blocked_reason(reason)
