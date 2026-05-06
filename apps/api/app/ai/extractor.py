from __future__ import annotations

from collections import defaultdict
import calendar
from datetime import date
import json
import re
from typing import Any

from sqlalchemy.orm import Session

from app.ai.gemini_llm import gemini_llm
from app.ai.percent_change import is_percent_change_request
from app.ai.prompts import build_extraction_system_prompt
from app.ai.local_intent_model import local_intent_model
from app.core.config import settings
from app.models.semantic import SemanticDictionaryEntry
from app.schemas.query import ComparisonSpec, QueryIntent, TimeRange
from app.semantic_layer.loader import semantic_loader
from app.semantic_layer.resolver import SemanticResolver
from app.semantic_layer.time_context import SemanticTimeContext


def _json_dumps_for_llm_prompt(payload: dict[str, Any]) -> str:
    """Сериализация в промпт LLM: date/datetime → ISO, прочие несериализуемые → str."""

    def _default(o: Any) -> Any:
        if isinstance(o, date):
            return o.isoformat()
        return str(o)

    return json.dumps(payload, ensure_ascii=False, indent=2, default=_default)


MONTHS = {
    "января": 1,
    "январь": 1,
    "январе": 1,
    "февраля": 2,
    "февраль": 2,
    "феврале": 2,
    "марта": 3,
    "март": 3,
    "марте": 3,
    "апреля": 4,
    "апрель": 4,
    "апреле": 4,
    "мая": 5,
    "май": 5,
    "мае": 5,
    "июня": 6,
    "июнь": 6,
    "июне": 6,
    "июля": 7,
    "июль": 7,
    "июле": 7,
    "августа": 8,
    "август": 8,
    "августе": 8,
    "сентября": 9,
    "сентябрь": 9,
    "сентябре": 9,
    "октября": 10,
    "октябрь": 10,
    "октябре": 10,
    "ноября": 11,
    "ноябрь": 11,
    "ноябре": 11,
    "декабря": 12,
    "декабрь": 12,
    "декабре": 12,
}

DESTRUCTIVE_PATTERNS = [
    "удали",
    "удалить",
    "удаление",
    "снеси",
    "снести",
    "очисти",
    "очистить",
    "drop",
    "delete",
    "truncate",
    "alter",
    "insert",
    "update",
    "обнови",
    "обновить",
    "измени",
    "изменить",
]

AMBIGUOUS_OUT_OF_DOMAIN_PATTERNS = [
    "айфон",
    "айфонов",
    "iphone",
    "товар",
    "товаров",
]

UNSUPPORTED_ANALYTICS_PATTERNS = {
    "канал": "В текущем датасете нет измерения по каналам. Сформулируйте вопрос по заказам, тендерам, отменам, городам, статусам или времени.",
    "каналы": "В текущем датасете нет измерения по каналам. Сформулируйте вопрос по заказам, тендерам, отменам, городам, статусам или времени.",
    "каналам": "В текущем датасете нет измерения по каналам. Сформулируйте вопрос по заказам, тендерам, отменам, городам, статусам или времени.",
    "прибыль": "В текущем датасете нет себестоимости и прибыли. Доступны выручка, средняя цена заказа, заказы, отмены и тендеры.",
    "прибыли": "В текущем датасете нет себестоимости и прибыли. Доступны выручка, средняя цена заказа, заказы, отмены и тендеры.",
    "маржа": "В текущем датасете нет себестоимости и прибыли. Доступны выручка, средняя цена заказа, заказы, отмены и тендеры.",
    "ltv": "В текущем датасете нет LTV и client lifetime-аналитики. Доступны метрики по заказам, тендерам, отменам, цене, дистанции и длительности.",
}

EXPLICIT_DIMENSION_PHRASES = {
    "по дням": "order_date",
    "по датам": "order_date",
    "по каждому дню": "order_date",
    "в разбивке по дням": "order_date",
    "по неделям": "order_week",
    "понедельно": "order_week",
    "в разбивке по неделям": "order_week",
    "по месяцам": "order_month",
    "помесячно": "order_month",
    "в разбивке по месяцам": "order_month",
    "по часам": "order_hour",
    "по часам дня": "order_hour",
    "по дням недели": "order_dow",
    "по городу": "city_id",
    "в разрезе по городам": "city_id",
    "по каждому городу": "city_id",
    "по статусам заказа": "order_status",
    "по статусу заказа": "order_status",
    "по статусам": "order_status",
    "по статусам тендера": "tender_status",
    "по статусу тендера": "tender_status",
    "по причинам отмен": "cancel_source",
    "по источникам отмен": "cancel_source",
    "по причине отмены": "cancel_source",
    "по городам": "city_id",
    "по водителям": "driver_id",
    "по водителю": "driver_id",
    "в разрезе по водителям": "driver_id",
    "по каждому водителю": "driver_id",
    "по пользователям": "user_id",
    "по user": "user_id",
    "в разрезе пользователей": "user_id",
    "по каждому пользователю": "user_id",
}

RANKING_DIMENSION_HINTS = [
    ("день недели", "order_dow"),
    ("дней недели", "order_dow"),
    ("деньнедели", "order_dow"),
    ("час", "order_hour"),
    ("часа", "order_hour"),
    ("часов", "order_hour"),
    ("город", "city_id"),
    ("города", "city_id"),
    ("городов", "city_id"),
    ("водитель", "driver_id"),
    ("водителя", "driver_id"),
    ("водителей", "driver_id"),
    ("статус заказа", "order_status"),
    ("статуса заказа", "order_status"),
    ("статусов заказа", "order_status"),
    ("статус тендера", "tender_status"),
    ("статуса тендера", "tender_status"),
    ("статусов тендера", "tender_status"),
    ("источник отмен", "cancel_source"),
    ("источника отмен", "cancel_source"),
    ("источников отмен", "cancel_source"),
    ("пользователь", "user_id"),
    ("пользователя", "user_id"),
    ("пользователей", "user_id"),
    ("день", "order_date"),
    ("дня", "order_date"),
    ("дней", "order_date"),
    ("дата", "order_date"),
    ("дат", "order_date"),
]

VAGUE_TERM_PATTERNS = ["дорог", "быстр", "плох", "хорош", "дешев"]

MONTH_PATTERN = "|".join(sorted({re.escape(month) for month in MONTHS}, key=len, reverse=True))
TEXTUAL_DATE_PATTERN = rf"\d{{1,2}}\s+(?:{MONTH_PATTERN})(?:\s+\d{{4}})?"
NUMERIC_DATE_PATTERN = r"\d{1,2}\.\d{1,2}(?:\.\d{2,4})?"


class HybridIntentExtractor:
    def __init__(self, db: Session):
        self.db = db
        self.catalog = semantic_loader.load_catalog_for_db(db)
        self.resolver = SemanticResolver(db)
        self.time_context = SemanticTimeContext(db)
        self._last_metric_typo_notes: list[str] = []

    def extract(self, question: str) -> QueryIntent:
        intent, _ = self.extract_with_trace(question, query_mode="auto")
        return intent

    def extract_with_trace(self, question: str, *, query_mode: str = "auto") -> tuple[QueryIntent, dict[str, Any]]:
        question_normalized = self._normalize_text(question)
        rule_based = self._rule_based_parse(question_normalized)
        local_based, local_trace = self._local_parse(question)
        merged = self._merge(rule_based, local_based, local_trace)
        merged["question"] = question
        llm_remote = settings.adaptive_intent_fallback_enabled and gemini_llm.remote_enabled
        llm_trace = {
            "enabled": query_mode != "fast" and llm_remote,
            "used_provider": None,
            "used_model": None,
            "attempts": [],
            "status": "disabled",
            "query_mode": query_mode,
        }
        llm_assisted = False
        if query_mode == "fast":
            llm_trace["status"] = "skipped_fast_mode"
        elif self._should_try_llm_fallback(query_mode, question_normalized, merged, local_trace):
            merged, llm_trace, llm_assisted = self._refine_with_llm_fallback(
                question=question,
                normalized_question=question_normalized,
                current_payload=merged,
            )
        elif llm_trace["enabled"]:
            llm_trace["status"] = "skipped"
        trace = {
            "normalized_question": question_normalized,
            "query_mode": query_mode,
            "rule_based": {
                "metrics": rule_based.get("metrics", []),
                "dimensions": rule_based.get("dimensions", []),
                "time_expression": rule_based.get("time_expression"),
                "time_range_override": rule_based.get("time_range_override"),
                "multi_date": rule_based.get("multi_date"),
                "comparison": rule_based.get("comparison"),
                "preferred_chart_type": rule_based.get("preferred_chart_type"),
                "ambiguity_reasons": rule_based.get("ambiguity_reasons", []),
                "notes": rule_based.get("notes", []),
            },
            "llm": llm_trace,
            "local_refinement": local_trace,
            "merged": {
                "metrics": merged.get("metrics", []),
                "dimensions": merged.get("dimensions", []),
                "time_expression": merged.get("time_expression"),
                "time_range_override": merged.get("time_range_override"),
                "multi_date": merged.get("multi_date"),
                "comparison": merged.get("comparison"),
                "preferred_chart_type": merged.get("preferred_chart_type"),
                "confidence": merged.get("confidence"),
            },
            "effective_source": self._effective_source(local_based, llm_assisted),
        }
        return QueryIntent.model_validate(merged), trace

    def _rule_based_parse(self, question: str) -> dict[str, Any]:
        metric_hits = self._match_metric_keys(question)
        metric_hits, metric_notes = self._align_metrics_with_request_type(question, metric_hits)
        metric_notes = [*self._last_metric_typo_notes, *metric_notes]
        dimension_hits = self._match_dimension_keys(question)
        filters, filter_notes = self._extract_filters(question)
        explicit_comparison, explicit_time_range, explicit_comparison_notes = self._extract_explicit_comparison_period(question)
        time_expression, time_range_override, discrete_dates, time_notes = self._extract_time_range(question)
        if explicit_time_range:
            time_expression = None
            time_range_override = explicit_time_range
        comparison = explicit_comparison or self._detect_comparison(question)
        preferred_chart_type, chart_notes = self._extract_chart_preference(question)
        ranking_sort, ranking_limit, ranking_dimension, ranking_notes = self._extract_ranking_preferences(question, metric_hits)
        if ranking_dimension and ranking_dimension not in dimension_hits:
            dimension_hits.append(ranking_dimension)
        if ranking_limit and not preferred_chart_type:
            preferred_chart_type = "bar"
        if len(discrete_dates) > 1:
            if "order_date" not in dimension_hits:
                dimension_hits.append("order_date")
            if comparison.enabled and not self._requests_percent_change(question):
                comparison = ComparisonSpec()
            if not preferred_chart_type:
                preferred_chart_type = "bar"

        intent_type = self._detect_intent_type(question, comparison.enabled, dimension_hits)
        ambiguity_reasons = self._detect_ambiguity(
            question,
            metric_hits=metric_hits,
            filters=filters,
            time_expression=time_expression,
            time_range_override=time_range_override,
            discrete_dates=discrete_dates,
            comparison=comparison,
        )

        suggestion_notes = self._suggest_correction_notes(
            question,
            low_confidence_hint=not metric_hits or bool(ranking_notes) or bool(filter_notes),
        )
        notes: list[str] = [
            *time_notes,
            *metric_notes,
            *filter_notes,
            *chart_notes,
            *explicit_comparison_notes,
            *ranking_notes,
            *suggestion_notes,
        ]
        if comparison.enabled:
            notes.append("Обнаружен сравнительный запрос между периодами.")
        if not metric_hits and not ambiguity_reasons:
            notes.append("Метрика не распознана явно, поэтому система может использовать безопасную метрику по умолчанию.")

        confidence = 0.35
        if metric_hits:
            confidence += 0.25
        if dimension_hits:
            confidence += 0.15
        if time_expression or time_range_override or discrete_dates:
            confidence += 0.15
        if comparison.enabled:
            confidence += 0.05
        if ambiguity_reasons:
            confidence -= 0.35

        clarification_questions: list[str] = []
        if ambiguity_reasons:
            clarification_questions.append(
                "Сформулируйте аналитический вопрос по заказам, тендерам, отменам, цене, длительности или дистанции."
            )

        return {
            "intent_type": intent_type,
            "metrics": metric_hits,
            "dimensions": dimension_hits,
            "filters": filters,
            "time_expression": time_expression,
            "time_range_override": time_range_override.model_dump(mode="json") if time_range_override else None,
            "multi_date": {"dates": [item.isoformat() for item in discrete_dates], "mode": "include"} if discrete_dates else None,
            "comparison": comparison.model_dump(),
            "preferred_chart_type": preferred_chart_type,
            "sort": ranking_sort,
            "limit": ranking_limit or 50,
            "confidence": max(0.1, min(confidence, 0.95)),
            "ambiguity_reasons": ambiguity_reasons,
            "clarification_questions": clarification_questions,
            "notes": notes,
        }

    def _align_metrics_with_request_type(self, question: str, metric_keys: list[str]) -> tuple[list[str], list[str]]:
        if len(metric_keys) > 1:
            return metric_keys, []

        requested_kind = self._detect_requested_metric_kind(question)
        if not requested_kind:
            return metric_keys, []

        if requested_kind == "sum" and self._requests_money_metric(question) and not self._has_money_metric():
            return [], ["В текущем датасете нет денежной метрики для выручки/дохода."]

        matching = [key for key in metric_keys if self._metric_kind(key) == requested_kind]
        if matching:
            return matching, []

        if metric_keys:
            fallback = self._default_metric_for_kind(question, requested_kind)
            if fallback:
                return [fallback], [f"Метрика приведена к типу запроса: {requested_kind}."]
            return metric_keys, []

        fallback = self._default_metric_for_kind(question, requested_kind)
        if fallback:
            return [fallback], [f"Метрика выбрана по типу запроса: {requested_kind}."]
        return metric_keys, []

    def _detect_requested_metric_kind(self, question: str) -> str | None:
        if any(token in question for token in ["средн", "в среднем"]):
            return "avg"
        if any(token in question for token in ["сумм", "выручк", "доход", "оборот", "денег", "деньгам", "касс"]):
            return "sum"
        if "сколько" in question and "сколько процентов" not in question:
            return "count"
        return None

    def _default_metric_for_kind(self, question: str, kind: str) -> str | None:
        if kind == "count":
            if "total_orders" in self.catalog.metrics:
                return "total_orders"
            if "rows_count" in self.catalog.metrics:
                return "rows_count"
            for key, metric in self.catalog.metrics.items():
                if self._metric_kind(key) == "count":
                    return key
            return None
        if kind == "sum":
            if "total_revenue" in self.catalog.metrics:
                return "total_revenue"
            for key, metric in self.catalog.metrics.items():
                if self._metric_kind(key) == "sum":
                    return key
            return None
        if kind == "avg":
            if "скорост" in question and "avg_speed_mps" in self.catalog.metrics:
                return "avg_speed_mps"
            if "avg_order_price" in self.catalog.metrics:
                return "avg_order_price"
            for key, metric in self.catalog.metrics.items():
                if self._metric_kind(key) == "avg":
                    return key
            return None
        return None

    def _requests_money_metric(self, question: str) -> bool:
        return any(token in question for token in ["выручк", "доход", "оборот", "денег", "деньгам", "касс", "продаж"])

    def _has_money_metric(self) -> bool:
        money_tokens = ("выруч", "доход", "оборот", "revenue", "sales", "amount", "price", "cost", "total")
        for key, metric in self.catalog.metrics.items():
            haystack = " ".join([key, metric.label, metric.description, *metric.synonyms]).lower()
            if any(token in haystack for token in money_tokens):
                return True
        return False

    def _metric_kind(self, metric_key: str) -> str | None:
        metric = self.catalog.metrics.get(metric_key)
        if not metric:
            return None
        expression = metric.sql.upper()
        if "COUNT(" in expression:
            return "count"
        if "AVG(" in expression:
            return "avg"
        if "SUM(" in expression:
            return "sum"
        return None

    def _local_parse(self, question: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if self.catalog.base_dataset != "order_tender_facts":
            return None, {"status": "skipped_dynamic_dataset", "entries": 0}
        return local_intent_model.extract_json_with_trace(question)

    def _merge(
        self,
        rule_based: dict[str, Any],
        local_based: dict[str, Any] | None,
        local_trace: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not local_based:
            return rule_based

        local_based = local_based or {}
        exact_local_match = bool(
            local_trace
            and local_trace.get("status") == "ok"
            and float(local_trace.get("similarity", 0.0)) >= 0.999
        )
        preserve_rule_disambiguation = self._should_preserve_rule_disambiguation(rule_based)
        merged_lists = defaultdict(list)
        for key in ["filters", "ambiguity_reasons", "clarification_questions", "notes"]:
            primary_source = local_based if exact_local_match else rule_based
            secondary_source = rule_based if exact_local_match else local_based
            merged_lists[key] = list(primary_source.get(key, []))
            local_items = secondary_source.get(key, [])
            if preserve_rule_disambiguation and key in {"ambiguity_reasons", "clarification_questions"}:
                local_items = []
            for item in local_items:
                if item not in merged_lists[key]:
                    merged_lists[key].append(item)

        rule_comparison = ComparisonSpec.model_validate(rule_based.get("comparison") or {})
        local_comparison = ComparisonSpec.model_validate(local_based.get("comparison") or {})

        if exact_local_match:
            merged_metrics = self._prefer_list(local_based.get("metrics"), rule_based.get("metrics"))
            merged_dimensions = self._prefer_list(local_based.get("dimensions"), rule_based.get("dimensions"))
        else:
            merged_metrics = self._prefer_list(rule_based.get("metrics"), local_based.get("metrics"))
            merged_dimensions = self._prefer_list(rule_based.get("dimensions"), local_based.get("dimensions"))
            if rule_based.get("sort") and rule_based.get("dimensions"):
                merged_dimensions = list(rule_based.get("dimensions") or [])

        return {
            "intent_type": (
                local_based.get("intent_type")
                if exact_local_match and local_based.get("intent_type") != "unknown"
                else (
                    rule_based.get("intent_type")
                    if rule_based.get("intent_type") != "unknown"
                    else (local_based.get("intent_type") or "unknown")
                )
            ),
            "metrics": merged_metrics,
            "dimensions": merged_dimensions,
            "filters": merged_lists["filters"],
            "time_expression": (
                local_based.get("time_expression") or rule_based.get("time_expression")
                if exact_local_match
                else (rule_based.get("time_expression") or local_based.get("time_expression"))
            ),
            "time_range_override": (
                local_based.get("time_range_override") or rule_based.get("time_range_override")
                if exact_local_match
                else (rule_based.get("time_range_override") or local_based.get("time_range_override"))
            ),
            "multi_date": (
                local_based.get("multi_date") or rule_based.get("multi_date")
                if exact_local_match
                else (rule_based.get("multi_date") or local_based.get("multi_date"))
            ),
            "comparison": (
                local_comparison.model_dump()
                if exact_local_match and local_comparison.enabled
                else (
                    rule_comparison.model_dump()
                    if rule_comparison.enabled
                    else local_comparison.model_dump()
                )
            ),
            "preferred_chart_type": (
                local_based.get("preferred_chart_type") or rule_based.get("preferred_chart_type")
                if exact_local_match
                else (rule_based.get("preferred_chart_type") or local_based.get("preferred_chart_type"))
            ),
            "sort": (
                local_based.get("sort") or rule_based.get("sort")
                if exact_local_match
                else (rule_based.get("sort") or local_based.get("sort"))
            ),
            "limit": (
                (local_based.get("limit") or rule_based.get("limit"))
                if exact_local_match
                else (
                    rule_based.get("limit")
                    if rule_based.get("sort")
                    else (local_based.get("limit") or rule_based.get("limit"))
                )
            ),
            "confidence": max(
                rule_based.get("confidence", 0),
                min(local_based.get("confidence", 0), 0.95),
            ),
            "ambiguity_reasons": merged_lists["ambiguity_reasons"],
            "clarification_questions": merged_lists["clarification_questions"],
            "notes": merged_lists["notes"],
        }

    def _effective_source(self, local_based: dict[str, Any] | None, llm_assisted: bool) -> str:
        if llm_assisted:
            return "hybrid_llm_fallback"
        if local_based:
            return "hybrid_local"
        return "rules_only"

    def _should_try_llm_fallback(
        self,
        query_mode: str,
        question: str,
        merged: dict[str, Any],
        local_trace: dict[str, Any],
    ) -> bool:
        if query_mode == "fast":
            return False
        if not settings.adaptive_intent_fallback_enabled or not gemini_llm.remote_enabled:
            return False
        if query_mode == "full":
            if self._has_hard_ambiguity_patterns(question):
                return False
            return True
        if self._has_hard_ambiguity_patterns(question):
            return False
        if merged.get("confidence", 0.0) < settings.adaptive_intent_low_confidence_threshold:
            return True
        if not merged.get("metrics"):
            return True
        if merged.get("ambiguity_reasons"):
            return True
        return local_trace.get("status") != "ok"

    def _refine_with_llm_fallback(
        self,
        *,
        question: str,
        normalized_question: str,
        current_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        user_prompt = self._build_llm_fallback_prompt(question, current_payload)
        llm_payload, llm_trace = gemini_llm.extract_json_with_trace(build_extraction_system_prompt(self.catalog), user_prompt)
        if not llm_payload:
            llm_trace["status"] = "no_candidate"
            return current_payload, llm_trace, False

        try:
            llm_candidate = QueryIntent.model_validate({**llm_payload, "question": question}).model_dump(mode="python")
        except Exception as exc:
            llm_trace["status"] = "invalid_candidate"
            llm_trace["validation_error"] = str(exc)
            return current_payload, llm_trace, False

        candidate_score = self._intent_quality_score(llm_candidate)
        current_score = self._intent_quality_score(current_payload)
        llm_trace["candidate_quality"] = round(candidate_score, 4)
        llm_trace["current_quality"] = round(current_score, 4)

        if not self._should_adopt_llm_candidate(current_payload, llm_candidate, current_score, candidate_score):
            llm_trace["status"] = "discarded"
            return current_payload, llm_trace, False

        llm_candidate = self._preserve_safety_signals(normalized_question, current_payload, llm_candidate)
        llm_trace["status"] = "adopted"
        return llm_candidate, llm_trace, True

    def _build_llm_fallback_prompt(self, question: str, current_payload: dict[str, Any]) -> str:
        compact_payload = {
            "intent_type": current_payload.get("intent_type"),
            "metrics": current_payload.get("metrics", []),
            "dimensions": current_payload.get("dimensions", []),
            "filters": current_payload.get("filters", []),
            "time_expression": current_payload.get("time_expression"),
            "time_range_override": current_payload.get("time_range_override"),
            "multi_date": current_payload.get("multi_date"),
            "comparison": current_payload.get("comparison"),
            "preferred_chart_type": current_payload.get("preferred_chart_type"),
            "confidence": current_payload.get("confidence"),
            "ambiguity_reasons": current_payload.get("ambiguity_reasons", []),
            "clarification_questions": current_payload.get("clarification_questions", []),
            "notes": current_payload.get("notes", []),
        }
        return (
            "Пользовательский вопрос:\n"
            f"{question}\n\n"
            "Текущая интерпретация локального парсера:\n"
            f"{_json_dumps_for_llm_prompt(compact_payload)}\n\n"
            "Это fallback-попытка: исправь только intent JSON, если видишь более точную интерпретацию. "
            "Если вопрос всё ещё неоднозначен, честно верни ambiguity_reasons и clarification_questions."
        )

    def _should_adopt_llm_candidate(
        self,
        current_payload: dict[str, Any],
        llm_candidate: dict[str, Any],
        current_score: float,
        candidate_score: float,
    ) -> bool:
        current_reasons = current_payload.get("ambiguity_reasons", [])
        candidate_reasons = llm_candidate.get("ambiguity_reasons", [])
        if not llm_candidate.get("metrics") and current_payload.get("metrics"):
            return False
        if candidate_score >= current_score + 0.08:
            return True
        if current_reasons and len(candidate_reasons) < len(current_reasons) and candidate_score > current_score:
            return True
        if (
            current_payload.get("confidence", 0.0) < settings.adaptive_intent_low_confidence_threshold
            and llm_candidate.get("confidence", 0.0) >= current_payload.get("confidence", 0.0) + 0.1
        ):
            return True
        if not current_payload.get("metrics") and llm_candidate.get("metrics"):
            return True
        return False

    def _intent_quality_score(self, payload: dict[str, Any]) -> float:
        score = float(payload.get("confidence", 0.0))
        if payload.get("metrics"):
            score += 0.18
        if payload.get("dimensions"):
            score += 0.08
        if payload.get("time_expression") or payload.get("time_range_override") or payload.get("multi_date"):
            score += 0.06
        if payload.get("filters"):
            score += 0.04
        score -= min(0.36, 0.12 * len(payload.get("ambiguity_reasons", [])))
        if not payload.get("metrics"):
            score -= 0.1
        return round(score, 4)

    def _preserve_safety_signals(
        self,
        question: str,
        current_payload: dict[str, Any],
        llm_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._has_hard_ambiguity_patterns(question):
            return llm_candidate

        candidate_reasons = list(llm_candidate.get("ambiguity_reasons", []))
        for reason in current_payload.get("ambiguity_reasons", []):
            if reason not in candidate_reasons:
                candidate_reasons.append(reason)
        llm_candidate["ambiguity_reasons"] = candidate_reasons

        candidate_questions = list(llm_candidate.get("clarification_questions", []))
        for item in current_payload.get("clarification_questions", []):
            if item not in candidate_questions:
                candidate_questions.append(item)
        llm_candidate["clarification_questions"] = candidate_questions
        llm_candidate["confidence"] = min(llm_candidate.get("confidence", 0.0), current_payload.get("confidence", 0.0))
        return llm_candidate

    def _has_hard_ambiguity_patterns(self, question: str) -> bool:
        if any(pattern in question for pattern in DESTRUCTIVE_PATTERNS):
            return True
        if self.catalog.base_dataset != "order_tender_facts":
            return False
        if any(pattern in question for pattern in AMBIGUOUS_OUT_OF_DOMAIN_PATTERNS):
            return True
        return any(pattern in question for pattern in UNSUPPORTED_ANALYTICS_PATTERNS)

    def _prefer_list(self, *sources: list[str] | None) -> list[str]:
        for source in sources:
            if source:
                return source
        return []

    def _should_preserve_rule_disambiguation(self, rule_based: dict[str, Any]) -> bool:
        if rule_based.get("ambiguity_reasons"):
            return False
        multi_date = rule_based.get("multi_date") or {}
        explicit_dates = multi_date.get("dates") or []
        has_explicit_structure = bool(rule_based.get("time_range_override")) or len(explicit_dates) >= 2
        return has_explicit_structure and bool(rule_based.get("metrics"))

    def _match_metric_keys(self, question: str) -> list[str]:
        matches: list[str] = list(self._match_compound_metrics(question))
        typo_notes: list[str] = []
        if not matches:
            stem_matches = self._match_metric_stems(question)
            matches.extend(stem_matches)
            if "cancelled_orders" in stem_matches:
                cancel_typo_match = re.search(r"\bатмен\w*\b", self._normalize_text(question))
                if cancel_typo_match:
                    typo_notes.append(
                        f"Похоже, имелось в виду «отменилось» вместо «{cancel_typo_match.group(0)}»."
                    )
        occupied_spans: list[tuple[int, int]] = []

        metric_aliases = sorted(self._collect_metric_aliases(), key=lambda item: len(item[0]), reverse=True)
        for alias, key in metric_aliases:
            for match in re.finditer(re.escape(alias), question):
                span = match.span()
                if any(not (span[1] <= taken[0] or span[0] >= taken[1]) for taken in occupied_spans):
                    continue
                occupied_spans.append(span)
                matches.append(key)
                break

        if not matches:
            typo_matches, typo_notes = self._match_metric_typos(question, metric_aliases)
            matches.extend(typo_matches)

        unique_matches: list[str] = []
        for key in matches:
            if key not in unique_matches:
                unique_matches.append(key)
        self._last_metric_typo_notes = typo_notes
        return unique_matches

    def _match_metric_stems(self, question: str) -> list[str]:
        if self.catalog.base_dataset != "order_tender_facts":
            return []
        matches: list[str] = []
        normalized = self._normalize_text(question)
        completed_orders_requested = bool(
            re.search(r"(?:выполненн\w+|завершенн\w+)\s+(?:заказ\w*|закз\w*|аказ\w*)", normalized)
        )
        if re.search(r"(?:в|ы)?ручк\w*|выучк\w*|оборот\w*", normalized):
            matches.append("total_revenue")
        if re.search(r"(?:количеств\w+|числ\w+)\s+(?:заказ\w*|закз\w*|аказ\w*)", normalized) and not completed_orders_requested:
            matches.append("total_orders")
        if completed_orders_requested:
            matches.append("completed_orders")
        if re.search(r"отмен\w*|атмен\w*", normalized):
            matches.append("cancelled_orders")

        unique_matches: list[str] = []
        for key in matches:
            if key not in unique_matches:
                unique_matches.append(key)
        return unique_matches

    def _match_metric_typos(self, question: str, aliases: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        matches: list[str] = []
        notes: list[str] = []
        seen_keys: set[str] = set()
        for token_match in re.finditer(r"\b[0-9a-zа-яё_-]{4,}\b", question):
            token = token_match.group(0)
            best_key: str | None = None
            best_alias: str | None = None
            best_distance: int | None = None
            best_alias_length = -1
            for alias, key in aliases:
                if " " in alias or len(alias) < 5 or alias == token:
                    continue
                if abs(len(alias) - len(token)) > 2:
                    continue
                if alias[-1] != token[-1]:
                    continue
                if not self._is_safe_single_typo(token, alias):
                    continue
                distance = self._levenshtein_distance(token, alias)
                if distance > 2:
                    continue
                if (
                    best_distance is None
                    or distance < best_distance
                    or (distance == best_distance and len(alias) > best_alias_length)
                ):
                    best_key = key
                    best_alias = alias
                    best_distance = distance
                    best_alias_length = len(alias)
            if best_key and best_key not in seen_keys:
                matches.append(best_key)
                seen_keys.add(best_key)
                if best_alias and token != best_alias:
                    notes.append(f"Похоже, имелось в виду «{best_alias}» вместо «{token}».")
        return matches, notes

    def _is_safe_single_typo(self, token: str, alias: str) -> bool:
        if token == alias:
            return False
        if self._levenshtein_distance(token, alias) <= 1:
            return True
        if len(token) >= 7 and len(alias) >= 7 and self._levenshtein_distance(token, alias) <= 2:
            return True
        if len(token) != len(alias):
            return False
        mismatches = [index for index, (left, right) in enumerate(zip(token, alias)) if left != right]
        if len(mismatches) != 2:
            return False
        first, second = mismatches
        return second == first + 1 and token[first] == alias[second] and token[second] == alias[first]

    def _levenshtein_distance(self, source: str, target: str) -> int:
        if source == target:
            return 0
        if not source:
            return len(target)
        if not target:
            return len(source)
        previous = list(range(len(target) + 1))
        for index, source_char in enumerate(source, start=1):
            current = [index]
            for target_index, target_char in enumerate(target, start=1):
                insertion = current[target_index - 1] + 1
                deletion = previous[target_index] + 1
                substitution = previous[target_index - 1] + (source_char != target_char)
                current.append(min(insertion, deletion, substitution))
            previous = current
        return previous[-1]

    def _match_compound_metrics(self, question: str) -> list[str]:
        matches: list[str] = []
        completion_rate_requested = bool(
            re.search(r"(процент|дол[яю]).*((выполненн|завершенн)\w*\s+заказ\w+)", question)
        )
        if completion_rate_requested:
            matches.append("order_completion_rate")
        if not completion_rate_requested and re.search(r"(выполненн|завершенн)\w*\s+заказ\w*", question):
            matches.append("completed_orders")
        if "отмен" in question and re.search(r"клиент\w+\s+и\s+водител\w+", question):
            matches.extend(["client_cancellations", "driver_cancellations"])
        if re.search(r"(выполненн\w+|поездк\w+)\s+и\s+отмен", question):
            matches.extend(["completed_orders", "cancelled_orders"])
        if re.search(r"отмен\w+\s+и\s+(выполненн\w+|поездк\w+)", question):
            matches.extend(["completed_orders", "cancelled_orders"])
        if re.search(r"выручк\w+.*\bи\b.*(выполненн\w+|поездк\w+)", question):
            matches.extend(["total_revenue", "completed_orders"])
        if re.search(r"(оборот\w*|выручк\w+).*\bи\b.*завершенн\w+", question):
            matches.extend(["total_revenue", "completed_orders"])
        if re.search(r"завершенн\w+\s+и\s+отмен", question):
            matches.extend(["completed_orders", "cancelled_orders"])
        if re.search(r"отмен\w+\s+и\s+завершенн\w+", question):
            matches.extend(["completed_orders", "cancelled_orders"])
        return matches

    def _collect_metric_aliases(self) -> list[tuple[str, str]]:
        aliases: list[tuple[str, str]] = []
        for key, metric in self.catalog.metrics.items():
            for alias in metric.synonyms:
                aliases.append((self._normalize_text(alias), key))
        for term, config in self.catalog.business_terms.items():
            if config.get("entity_type") != "metric":
                continue
            target_key = config.get("target_key")
            if target_key in self.catalog.metrics:
                aliases.append((self._normalize_text(term), target_key))
        for entry in self.db.query(SemanticDictionaryEntry).filter(SemanticDictionaryEntry.is_active.is_(True)).all():
            if entry.target_key not in self.catalog.metrics:
                continue
            aliases.append((self._normalize_text(entry.term), entry.target_key))
            for alias in entry.synonyms_json:
                aliases.append((self._normalize_text(alias), entry.target_key))
        return aliases

    def _match_dimension_keys(self, question: str) -> list[str]:
        matches: list[str] = []
        occupied_spans: list[tuple[int, int]] = []

        for phrase, key in sorted(EXPLICIT_DIMENSION_PHRASES.items(), key=lambda item: len(item[0]), reverse=True):
            for match in re.finditer(re.escape(phrase), question):
                span = match.span()
                if any(not (span[1] <= taken[0] or span[0] >= taken[1]) for taken in occupied_spans):
                    continue
                resolved_key = self._resolve_dimension_key(key)
                if not resolved_key:
                    continue
                occupied_spans.append(span)
                if resolved_key not in matches:
                    matches.append(resolved_key)
                break

        for alias, key in sorted(self._collect_dimension_aliases(), key=lambda item: len(item[0]), reverse=True):
            for match in re.finditer(re.escape(alias), question):
                span = match.span()
                if any(not (span[1] <= taken[0] or span[0] >= taken[1]) for taken in occupied_spans):
                    continue
                occupied_spans.append(span)
                if key not in matches:
                    matches.append(key)
                break

        if not matches:
            for key in self._match_dimension_typos(question, self._collect_dimension_aliases()):
                if key not in matches:
                    matches.append(key)

        return matches

    def _match_dimension_typos(self, question: str, aliases: list[tuple[str, str]]) -> list[str]:
        matches: list[str] = []
        seen_keys: set[str] = set()
        for token_match in re.finditer(r"\b[0-9a-zа-яё_-]{4,}\b", question):
            token = token_match.group(0)
            best_key: str | None = None
            best_distance: int | None = None
            best_alias_length = -1
            for alias, key in aliases:
                if " " in alias or len(alias) < 4 or alias == token:
                    continue
                if abs(len(alias) - len(token)) > 2:
                    continue
                if alias[-1] != token[-1]:
                    continue
                if not self._is_safe_single_typo(token, alias):
                    continue
                distance = self._levenshtein_distance(token, alias)
                if distance > 2:
                    continue
                if (
                    best_key is None
                    or distance < (best_distance or 99)
                    or (distance == best_distance and len(alias) > best_alias_length)
                ):
                    best_key = key
                    best_distance = distance
                    best_alias_length = len(alias)
            if best_key and best_key not in seen_keys:
                matches.append(best_key)
                seen_keys.add(best_key)
        return matches

    def _collect_dimension_aliases(self) -> list[tuple[str, str]]:
        aliases: list[tuple[str, str]] = []
        for key, dimension in self.catalog.dimensions.items():
            for alias in dimension.synonyms:
                normalized_alias = self._normalize_text(alias)
                if dimension.kind == "time" and not self._is_explicit_grouping_alias(normalized_alias):
                    continue
                aliases.append((normalized_alias, key))
        for term, config in self.catalog.business_terms.items():
            if config.get("entity_type") != "dimension":
                continue
            target_key = config.get("target_key")
            if target_key in self.catalog.dimensions:
                aliases.append((self._normalize_text(term), target_key))
        for entry in self.db.query(SemanticDictionaryEntry).filter(SemanticDictionaryEntry.is_active.is_(True)).all():
            if entry.target_key not in self.catalog.dimensions:
                continue
            aliases.append((self._normalize_text(entry.term), entry.target_key))
            for alias in entry.synonyms_json:
                aliases.append((self._normalize_text(alias), entry.target_key))
        return aliases

    def _resolve_dimension_key(self, key: str | None) -> str | None:
        if not key:
            return None
        if key in self.catalog.dimensions:
            return key
        prefixed = f"dim_{key}"
        if prefixed in self.catalog.dimensions:
            return prefixed
        normalized_key = self._normalize_text(key)
        for dimension_key, dimension in self.catalog.dimensions.items():
            candidates = [dimension_key, dimension.label, *dimension.synonyms]
            if normalized_key in {self._normalize_text(str(item)) for item in candidates if str(item).strip()}:
                return dimension_key
        return None

    def _is_explicit_grouping_alias(self, alias: str) -> bool:
        grouping_markers = ("по ", "разбив", "помесяч", "понедел", "по дня", "сгрупп")
        return any(marker in alias for marker in grouping_markers)

    def _extract_filters(self, question: str) -> tuple[list[dict[str, Any]], list[str]]:
        filters: list[dict[str, Any]] = []
        notes: list[str] = []
        city_match = re.search(r"(?:город(?:\s+id)?|city)\s*(\d+)", question)
        if city_match:
            filters.append({"key": "city_id", "operator": "eq", "value": city_match.group(1)})
        user_match = re.search(r"(?:пользователь|user(?:_?id)?)\s*([a-zA-Z0-9_-]+)", question)
        if user_match:
            filters.append({"key": "user_id", "operator": "eq", "value": user_match.group(1)})
        if any(token in question for token in ["выходн", "по выходным", "на выходных"]):
            filters.append({"key": "order_dow", "operator": "in", "value": [0, 6]})
        elif any(token in question for token in ["будн", "по будням", "в будни"]):
            filters.append({"key": "order_dow", "operator": "in", "value": [1, 2, 3, 4, 5]})

        hour_range_match = re.search(
            r"(?:с|между)\s*(\d{1,2})\s*(?:час(?:а|ов)?)?\s*(?:до|по|и)\s*(\d{1,2})\s*час",
            question,
        )
        if hour_range_match:
            start_hour = int(hour_range_match.group(1))
            end_hour = int(hour_range_match.group(2))
            if 0 <= start_hour <= 23 and 0 <= end_hour <= 23 and start_hour <= end_hour:
                filters.append({"key": "order_hour", "operator": "gte", "value": start_hour})
                filters.append({"key": "order_hour", "operator": "lte", "value": end_hour})

        duration_gt_match = re.search(r"длител\w+\s+(?:больш[е]|выше)\s+(\d+)\s*мин", question)
        if duration_gt_match:
            filters.append({"key": "duration_seconds", "operator": "gt", "value": int(duration_gt_match.group(1)) * 60})

        duration_gte_match = re.search(r"длител\w+\s+(?:не\s+менее|минимум)\s+(\d+)\s*мин", question)
        if duration_gte_match:
            filters.append({"key": "duration_seconds", "operator": "gte", "value": int(duration_gte_match.group(1)) * 60})

        duration_lt_match = re.search(r"длител\w+\s+(?:меньше|ниже)\s+(\d+)\s*мин", question)
        if duration_lt_match:
            filters.append({"key": "duration_seconds", "operator": "lt", "value": int(duration_lt_match.group(1)) * 60})

        duration_lte_match = re.search(r"длител\w+\s+(?:не\s+более|максимум)\s+(\d+)\s*мин", question)
        if duration_lte_match:
            filters.append({"key": "duration_seconds", "operator": "lte", "value": int(duration_lte_match.group(1)) * 60})

        if any(re.search(rf"\b{pattern}\w*\b", question) for pattern in VAGUE_TERM_PATTERNS):
            notes.append("Нечёткие качественные термины в фильтрах проигнорированы; применена базовая интерпретация без таких условий.")

        return filters, notes

    def _calendar_season_range(self, stem: str, year: int) -> tuple[date, date, str] | None:
        s = stem.lower()
        if s.startswith("весн"):
            return date(year, 3, 1), date(year, 5, 31), f"Весна {year}"
        if s.startswith("лет"):
            return date(year, 6, 1), date(year, 8, 31), f"Лето {year}"
        if s.startswith("осен"):
            return date(year, 9, 1), date(year, 11, 30), f"Осень {year}"
        if s.startswith("зим"):
            feb_last = calendar.monthrange(year + 1, 2)[1]
            return date(year, 12, 1), date(year + 1, 2, feb_last), f"Зима {year}—{year + 1}"
        return None

    def _clamp_range_to_data_calendar(self, start_date: date, end_date: date) -> tuple[date, date, list[str]]:
        bounds = self.time_context.get_bounds()
        anchor = self.time_context.get_anchor_date()
        notes: list[str] = []
        if end_date < bounds.min_date or start_date > bounds.max_date:
            notes.append("Период не пересекается с данными; использован весь доступный диапазон.")
            return bounds.min_date, bounds.max_date, notes
        if start_date < bounds.min_date:
            notes.append(f"Начало периода обрезано по минимальной дате в данных: {bounds.min_date.isoformat()}.")
            start_date = bounds.min_date
        if end_date > bounds.max_date:
            notes.append(f"Конец периода обрезан по максимальной дате в данных: {bounds.max_date.isoformat()}.")
            end_date = bounds.max_date
        if start_date > end_date:
            notes.append("После обрезки по данным период выродился; использован доступный диапазон.")
            start_date, end_date = bounds.min_date, bounds.max_date
        if end_date > anchor:
            notes.append(
                f"Конец периода ограничен последней доступной датой в данных: {anchor.isoformat()}."
            )
            end_date = anchor
        if start_date > end_date:
            start_date = min(bounds.min_date, end_date)
        return start_date, end_date, notes

    def _extract_season_pair_comparison(
        self, question: str
    ) -> tuple[ComparisonSpec | None, TimeRange | None, list[str]]:
        """«за осень и зиму», «осень 2024 и зима 2025» — два календарных сезона в одном сравнении."""
        q = question.lower()
        pair = re.search(
            r"(?:\b(?:за|в)\s+)?((?:весн|лет|осен|зим)\w*)(?:\s+(\d{4}))?\s+и\s+((?:весн|лет|осен|зим)\w*)(?:\s+(\d{4}))?",
            q,
        )
        if not pair:
            return None, None, []

        stem_a, year_a, stem_b, year_b = pair.group(1), pair.group(2), pair.group(3), pair.group(4)
        if stem_a[:4] == stem_b[:4]:
            return None, None, []

        ay = int(year_a) if year_a else None
        by = int(year_b) if year_b else None
        anchor = self.time_context.get_anchor_date()
        default_year = anchor.year if anchor.month >= 5 else anchor.year - 1
        if ay is not None and by is not None:
            y_a, y_b = ay, by
        elif ay is not None:
            y_a = y_b = ay
        elif by is not None:
            y_a = y_b = by
        else:
            y_a = y_b = default_year

        r_a = self._calendar_season_range(stem_a, y_a)
        r_b = self._calendar_season_range(stem_b, y_b)
        if not r_a or not r_b:
            return None, None, []

        s1, e1, lab1 = r_a
        s2, e2, lab2 = r_b
        s1, e1, n1 = self._clamp_range_to_data_calendar(s1, e1)
        s2, e2, n2 = self._clamp_range_to_data_calendar(s2, e2)
        notes = [
            f"Сравнение двух сезонов: текущий период — «{lab1}», базовый — «{lab2}».",
            *n1,
            *n2,
        ]

        return (
            ComparisonSpec(
                enabled=True,
                mode="previous_period",
                baseline_label=lab2,
                baseline_start_date=s2,
                baseline_end_date=e2,
            ),
            TimeRange(label=lab1, start_date=s1, end_date=e1, grain="day"),
            notes,
        )

    def _extract_time_range(self, question: str) -> tuple[str | None, TimeRange | None, list[date], list[str]]:
        range_match = re.search(
            rf"(?:с|за период с)\s+({TEXTUAL_DATE_PATTERN}|{NUMERIC_DATE_PATTERN})\s+по\s+({TEXTUAL_DATE_PATTERN}|{NUMERIC_DATE_PATTERN})",
            question,
        )
        if range_match:
            start_text = range_match.group(1)
            end_text = range_match.group(2)
            start_date, end_date, note = self._parse_date_range(start_text, end_text)
            label = f"С {self._pretty_date_label(start_text, start_date)} по {self._pretty_date_label(end_text, end_date)}"
            return (
                None,
                TimeRange(label=label, start_date=start_date, end_date=end_date, grain="day"),
                [],
                [note] if note else [],
            )

        explicit_dates, explicit_date_notes = self._extract_explicit_dates(question)
        if explicit_dates:
            if len(explicit_dates) >= 2:
                return (
                    None,
                    TimeRange(
                        label=f"Выбранные даты: {', '.join(item.isoformat() for item in explicit_dates)}",
                        start_date=min(explicit_dates),
                        end_date=max(explicit_dates),
                        grain="day",
                    ),
                    explicit_dates,
                    explicit_date_notes,
                )
            only_date = explicit_dates[0]
            return (
                None,
                TimeRange(
                    label=only_date.isoformat(),
                    start_date=only_date,
                    end_date=only_date,
                    grain="day",
                ),
                [],
                explicit_date_notes,
            )

        qlow = question.lower()
        season_year_match = re.search(r"\b((?:весн|лет|осен|зим)\w*)\s+(\d{4})\b", qlow)
        if not season_year_match:
            season_year_match = re.search(r"\b(\d{4})\s+((?:весн|лет|осен|зим)\w*)\b", qlow)
            if season_year_match:
                year = int(season_year_match.group(1))
                stem = season_year_match.group(2)
            else:
                stem = None
                year = None
        else:
            stem = season_year_match.group(1)
            year = int(season_year_match.group(2))

        if stem and year is not None:
            season_result = self._calendar_season_range(stem, year)
            if season_result:
                start_date, end_date, label = season_result
                bounds = self.time_context.get_bounds()
                anchor = self.time_context.get_anchor_date()
                notes: list[str] = []
                if end_date < bounds.min_date or start_date > bounds.max_date:
                    notes.append(
                        "Запрошенный сезон не пересекается с доступными датами в базе; использован весь доступный период."
                    )
                    start_date, end_date = bounds.min_date, bounds.max_date
                else:
                    if start_date < bounds.min_date:
                        notes.append(
                            f"Начало сезона обрезано по минимальной дате в данных: {bounds.min_date.isoformat()}."
                        )
                        start_date = bounds.min_date
                    if end_date > bounds.max_date:
                        notes.append(
                            f"Конец сезона обрезан по максимальной дате в данных: {bounds.max_date.isoformat()}."
                        )
                        end_date = bounds.max_date
                    if start_date > end_date:
                        notes.append("После обрезки по данным период выродился; показан доступный диапазон.")
                        start_date, end_date = bounds.min_date, bounds.max_date
                if end_date > anchor:
                    notes.append(
                        f"Сезон ещё не завершён в данных, поэтому конец периода ограничен последней доступной датой: {anchor.isoformat()}."
                    )
                    end_date = anchor
                if start_date > end_date:
                    start_date = min(bounds.min_date, end_date)
                return (
                    None,
                    TimeRange(label=label, start_date=start_date, end_date=end_date, grain="day"),
                    [],
                    notes,
                )

        month_match = re.search(rf"(?:за|в)\s+({MONTH_PATTERN})(?:\s+(\d{{4}}))?", question)
        if month_match:
            month_token = month_match.group(1)
            explicit_year = int(month_match.group(2)) if month_match.group(2) else None
            month = MONTHS.get(month_token)
            if month:
                start_date, end_date, is_partial = self.time_context.month_range(month, explicit_year)
                notes: list[str] = []
                if explicit_year is None:
                    notes.append(
                        f"Период интерпретирован как {start_date.isoformat()} - {end_date.isoformat()}, потому что год в запросе не указан."
                    )
                if is_partial:
                    notes.append(
                        f"Месяц ещё не завершён в данных, поэтому использован период по последнюю доступную дату: {end_date.isoformat()}."
                    )
                return (
                    None,
                    TimeRange(
                        label=self._build_month_label(month_token, start_date.year),
                        start_date=start_date,
                        end_date=end_date,
                        grain="day",
                    ),
                    [],
                    notes,
                )

        if "относительно прошлого месяца" in question or "к прошлому месяцу" in question or "относительно предыдущего месяца" in question:
            return "текущий месяц", None, [], []

        all_time_match = re.search(r"(?:за\s+)?(?:все|всё)\s+время|за\s+весь\s+период", question)
        if all_time_match:
            start_date, end_date = self.time_context.all_time_range()
            return (
                None,
                TimeRange(label="Всё время", start_date=start_date, end_date=end_date, grain="day"),
                [],
                [f"Использован весь доступный период данных: {start_date.isoformat()} - {end_date.isoformat()}."],
            )

        explicit_year_match = re.search(r"(?:за|в)\s+(\d{4})(?:\s+г(?:од|ода|оду|одом)?)?", question)
        if explicit_year_match:
            year = int(explicit_year_match.group(1))
            start_date, end_date, is_partial = self.time_context.calendar_year_range(year)
            notes: list[str] = []
            if is_partial:
                notes.append(
                    f"Год ещё не завершён в данных, поэтому использован период по последнюю доступную дату: {end_date.isoformat()}."
                )
            return (
                None,
                TimeRange(label=f"{year} год", start_date=start_date, end_date=end_date, grain="day"),
                [],
                notes,
            )

        rolling_year_match = re.search(r"(?:за|в)\s+(?:последний\s+год|год)\b", question)
        if rolling_year_match:
            start_date, end_date = self.time_context.rolling_year_range()
            return (
                None,
                TimeRange(label="Последние 12 календарных месяцев", start_date=start_date, end_date=end_date, grain="day"),
                [],
                [f"Период интерпретирован как последние 12 календарных месяцев: {start_date.isoformat()} - {end_date.isoformat()}."],
            )

        for phrase in sorted(self.catalog.time_mappings, key=len, reverse=True):
            if phrase in question:
                return phrase, None, [], []

        return None, None, [], []

    def _extract_explicit_comparison_period(
        self,
        question: str,
    ) -> tuple[ComparisonSpec | None, TimeRange | None, list[str]]:
        season_pair = self._extract_season_pair_comparison(question)
        if season_pair[0] is not None:
            return season_pair

        month_comparison = re.search(
            rf"(?:за|в)\s+({MONTH_PATTERN})(?:\s+(\d{{4}}))?\s+(?:относительно|по сравнению с|сравнительно с|к)\s+({MONTH_PATTERN})(?:\s+(\d{{4}}))?",
            question,
        )
        if not month_comparison:
            return None, None, []

        current_month_token = month_comparison.group(1)
        current_year = int(month_comparison.group(2)) if month_comparison.group(2) else None
        baseline_month_token = month_comparison.group(3)
        baseline_year = int(month_comparison.group(4)) if month_comparison.group(4) else None

        current_month = MONTHS.get(current_month_token)
        baseline_month = MONTHS.get(baseline_month_token)
        if not current_month or not baseline_month:
            return None, None, []

        current_start, current_end, current_partial = self.time_context.month_range(current_month, current_year)
        baseline_start, baseline_end, baseline_partial = self.time_context.month_range(baseline_month, baseline_year)

        notes: list[str] = []
        if current_year is None:
            notes.append(f"Для текущего месяца автоматически выбран год {current_start.year}.")
        if baseline_year is None:
            notes.append(f"Для базового месяца автоматически выбран год {baseline_start.year}.")
        if current_partial:
            notes.append(f"Текущий месяц в данных неполный, поэтому период ограничен датой {current_end.isoformat()}.")
        if baseline_partial:
            notes.append(f"Базовый месяц в данных неполный, поэтому период ограничен датой {baseline_end.isoformat()}.")

        return (
            ComparisonSpec(
                enabled=True,
                mode="previous_period",
                baseline_label=self._build_month_label(baseline_month_token, baseline_start.year),
                baseline_start_date=baseline_start,
                baseline_end_date=baseline_end,
            ),
            TimeRange(
                label=self._build_month_label(current_month_token, current_start.year),
                start_date=current_start,
                end_date=current_end,
                grain="day",
            ),
            notes,
        )

    def _extract_chart_preference(self, question: str) -> tuple[str | None, list[str]]:
        notes: list[str] = []
        if "не линейн" in question or "не лини" in question:
            notes.append("По формулировке запроса линейный график заменён на столбчатый.")
            return "bar", notes
        if "столб" in question or "гистограмм" in question or "bar" in question:
            return "bar", notes
        if "линейн" in question:
            return "line", notes
        if "кругов" in question or "pie" in question:
            return "pie", notes
        if "таблиц" in question or "без граф" in question:
            return "table", notes
        if "карточк" in question or "kpi" in question:
            return "kpi", notes
        return None, notes

    def _extract_ranking_preferences(
        self,
        question: str,
        metric_hits: list[str],
    ) -> tuple[str | None, int | None, str | None, list[str]]:
        metric_key = metric_hits[0] if metric_hits else None
        if not metric_key:
            return None, None, None, []

        sort_direction: str | None = None
        limit: int | None = None
        notes: list[str] = []

        top_match = re.search(r"\bтоп[- ]?(\d{1,2})\b", question)
        anti_top_match = re.search(r"\bантитоп[- ]?(\d{1,2})\b", question)
        better_plural_match = re.search(r"\bлучш(?:ие|их)\b(?:\s+(\d{1,2}))?", question)
        worse_plural_match = re.search(r"\bхудш(?:ие|их)\b(?:\s+(\d{1,2}))?", question)
        better_single_match = re.search(r"\bлучш(?:ий|его|ая|ее)\b", question)
        worse_single_match = re.search(r"\bхудш(?:ий|его|ая|ее)\b", question)

        if top_match:
            sort_direction = "DESC"
            limit = max(1, min(int(top_match.group(1)), 20))
            notes.append(f"Запрос интерпретирован как топ-{limit} по метрике {metric_key}.")
        elif anti_top_match:
            sort_direction = "ASC"
            limit = max(1, min(int(anti_top_match.group(1)), 20))
            notes.append(f"Запрос интерпретирован как антитоп-{limit} по метрике {metric_key}.")
        elif better_plural_match:
            sort_direction = "DESC"
            limit = max(1, min(int(better_plural_match.group(1) or 3), 20))
            notes.append(f"Запрос интерпретирован как список лучших значений по метрике {metric_key}.")
        elif worse_plural_match:
            sort_direction = "ASC"
            limit = max(1, min(int(worse_plural_match.group(1) or 3), 20))
            notes.append(f"Запрос интерпретирован как список худших значений по метрике {metric_key}.")
        elif better_single_match:
            sort_direction = "DESC"
            limit = 1
            notes.append(f"Запрос интерпретирован как поиск лучшего значения по метрике {metric_key}.")
        elif worse_single_match:
            sort_direction = "ASC"
            limit = 1
            notes.append(f"Запрос интерпретирован как поиск худшего значения по метрике {metric_key}.")
        elif any(
            marker in question
            for marker in ["самой большой", "самый большой", "наибольш", "максимальн", "больше всего", "самый высокий"]
        ):
            sort_direction = "DESC"
            limit = 1
            notes.append(f"Запрос интерпретирован как поиск максимума по метрике {metric_key}.")
        elif any(
            marker in question
            for marker in ["самой малень", "самый малень", "наименьш", "минимальн", "меньше всего", "самый низкий"]
        ):
            sort_direction = "ASC"
            limit = 1
            notes.append(f"Запрос интерпретирован как поиск минимума по метрике {metric_key}.")

        if not sort_direction:
            return None, None, None, []

        dimension_key = self._infer_ranking_dimension(question)
        if not dimension_key:
            inferred_dimensions = self._match_dimension_keys(question)
            dimension_key = inferred_dimensions[0] if inferred_dimensions else None
        return f"{metric_key} {sort_direction}", limit, dimension_key, notes

    def _infer_ranking_dimension(self, question: str) -> str | None:
        for marker, dimension_key in RANKING_DIMENSION_HINTS:
            if marker in question:
                return self._resolve_dimension_key(dimension_key)
        typo_dimension = self._infer_ranking_dimension_from_typos(question)
        if typo_dimension:
            return self._resolve_dimension_key(typo_dimension)
        return None

    def _infer_ranking_dimension_from_typos(self, question: str) -> str | None:
        ranking_aliases = [
            ("день", "order_date"),
            ("дня", "order_date"),
            ("дней", "order_date"),
            ("дата", "order_date"),
            ("датам", "order_date"),
            ("час", "order_hour"),
            ("часа", "order_hour"),
            ("часов", "order_hour"),
            ("город", "city_id"),
            ("города", "city_id"),
            ("городов", "city_id"),
            ("статус", "order_status"),
            ("источник", "cancel_source"),
            ("источника", "cancel_source"),
            ("источников", "cancel_source"),
        ]
        tokens = re.findall(r"\b[0-9a-zа-яё_-]{3,}\b", question)
        best_match: tuple[int, int, str] | None = None
        for token in tokens:
            for alias, dimension_key in ranking_aliases:
                if abs(len(alias) - len(token)) > 2:
                    continue
                distance = self._levenshtein_distance(token, alias)
                if distance > 2:
                    continue
                candidate = (distance, -len(alias), dimension_key)
                if best_match is None or candidate < best_match:
                    best_match = candidate
        if best_match:
            return best_match[2]
        return None

    def _suggest_correction_notes(self, question: str, *, low_confidence_hint: bool) -> list[str]:
        if not low_confidence_hint:
            return []

        suggestions: list[str] = []
        seen_pairs: set[tuple[str, str]] = set()
        skipped_tokens = {"покажи", "скажи", "можешь", "хочу", "понять", "пожалуйста"}
        candidate_terms = self._candidate_correction_terms()
        candidate_term_set = set(candidate_terms)
        for token in re.findall(r"\b[0-9a-zа-яё_-]{4,}\b", question):
            normalized_token = self._normalize_text(token)
            if normalized_token in skipped_tokens:
                continue
            if normalized_token in candidate_term_set:
                continue
            for alias in candidate_terms:
                if alias == normalized_token:
                    continue
                if abs(len(alias) - len(normalized_token)) > 2:
                    continue
                distance = self._levenshtein_distance(normalized_token, alias)
                if distance > 2:
                    continue
                pair = (normalized_token, alias)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                suggestions.append(f"Похоже, имелось в виду «{alias}» вместо «{normalized_token}».")
                break
            if len(suggestions) >= 2:
                break
        return suggestions

    def _candidate_correction_terms(self) -> list[str]:
        terms: list[str] = []
        for alias, _ in self._collect_metric_aliases():
            terms.append(alias)
        for alias, _ in self._collect_dimension_aliases():
            terms.append(alias)
        for config in self.catalog.filters.values():
            terms.extend(self._normalize_text(item) for item in config.synonyms)
        for item in self.catalog.time_mappings:
            terms.append(self._normalize_text(item))
        for item in ["лучший", "лучшие", "худший", "худшие", "топ", "антитоп"]:
            terms.append(item)

        unique: list[str] = []
        for item in terms:
            if len(item) < 4:
                continue
            if item not in unique:
                unique.append(item)
        return unique

    def _requests_percent_change(self, question: str) -> bool:
        return is_percent_change_request(question)

    def _detect_comparison(self, question: str) -> ComparisonSpec:
        if "этот год" in question and "прошлый год" in question:
            return ComparisonSpec(enabled=True, mode="year_over_year", baseline_label="Прошлый год")
        if any(token in question for token in ["сравни", "сравнение", "по сравнению", "в сравнении"]):
            return ComparisonSpec(enabled=True, mode="previous_period", baseline_label="Предыдущий период")
        if re.search(r"\b(текущ\w+|эт\w+)\b.*\bи\s+(прошл\w+|предыдущ\w+)\b", question):
            return ComparisonSpec(enabled=True, mode="previous_period", baseline_label="Предыдущий период")
        if is_percent_change_request(question):
            return ComparisonSpec(enabled=True, mode="previous_period", baseline_label="Предыдущий период")
        return ComparisonSpec()

    def _detect_intent_type(self, question: str, comparison_enabled: bool, dimension_hits: list[str]) -> str:
        if comparison_enabled:
            return "comparison"
        if any(token in question for token in ["динамика", "тренд", "по дням", "по неделям", "по месяцам", "по часам"]):
            return "trend"
        if dimension_hits:
            return "aggregation"
        return "aggregation"

    def _detect_ambiguity(
        self,
        question: str,
        *,
        metric_hits: list[str],
        filters: list[dict[str, Any]],
        time_expression: str | None,
        time_range_override: TimeRange | None,
        discrete_dates: list[date],
        comparison: ComparisonSpec,
    ) -> list[str]:
        reasons: list[str] = []

        if any(pattern in question for pattern in DESTRUCTIVE_PATTERNS):
            reason = "Запрос похож на команду изменения или удаления данных. Платформа работает только в режиме чтения и не выполняет такие действия."
            if reason not in reasons:
                reasons.append(reason)

        if self.catalog.base_dataset == "order_tender_facts":
            if any(pattern in question for pattern in AMBIGUOUS_OUT_OF_DOMAIN_PATTERNS):
                reason = "Текущий датасет описывает заказы такси, тендеры, отмены, цену, длительность и дистанцию. Такой запрос требует другой витрины данных."
                if reason not in reasons:
                    reasons.append(reason)

            for pattern, reason in UNSUPPORTED_ANALYTICS_PATTERNS.items():
                if pattern in question and reason not in reasons:
                    reasons.append(reason)

        multiple_dates_reason = self._detect_multiple_discrete_dates(question)
        if multiple_dates_reason and multiple_dates_reason not in reasons:
            reasons.append(multiple_dates_reason)

        normalized = self._normalize_text(question)
        if self.catalog.base_dataset != "order_tender_facts" and self._requests_money_metric(normalized) and not self._has_money_metric():
            reasons.append("В текущем CSV-датасете нет денежной метрики: выручки, дохода, оборота или суммы продаж.")

        has_time_context = bool(time_expression or time_range_override or discrete_dates)
        has_metric_context = bool(metric_hits)
        change_tokens = ("падени", "упал", "упали", "упало", "просел", "просела", "просели", "снижен", "снизил", "хуже")
        asks_for_reason = "почему" in normalized
        describes_change = any(token in normalized for token in change_tokens) or is_percent_change_request(normalized)
        if describes_change and not has_metric_context and not has_time_context and not filters:
            reasons.append("Уточните, что именно упало или изменилось и за какой период нужно это анализировать.")
        elif asks_for_reason and describes_change and not has_time_context and not comparison.enabled:
            reasons.append("Чтобы объяснить падение, укажите период или базу сравнения. Например: «почему продажи упали в выходные за прошлый месяц» или «сравни выходные этой недели и прошлой».")

        return reasons

    def _detect_multiple_discrete_dates(self, question: str) -> str | None:
        if self._has_explicit_date_range(question):
            return None
        explicit_dates, _ = self._extract_explicit_dates(question)
        if len(explicit_dates) >= 2:
            return None

        unique_matches = self._extract_discrete_dates(question)
        if len(unique_matches) >= 2:
            return "В вопросе указано несколько отдельных дат. Уточните диапазон через «с ... по ...» или выберите одну дату."
        return None

    def _has_explicit_date_range(self, question: str) -> bool:
        return bool(
            re.search(
                rf"(?:с|за период с)\s+({TEXTUAL_DATE_PATTERN}|{NUMERIC_DATE_PATTERN})\s+по\s+({TEXTUAL_DATE_PATTERN}|{NUMERIC_DATE_PATTERN})",
                question,
            )
        )

    def _extract_discrete_dates(self, question: str) -> list[str]:
        matches = re.findall(rf"{TEXTUAL_DATE_PATTERN}|{NUMERIC_DATE_PATTERN}", question)
        unique_matches: list[str] = []
        for item in matches:
            value = item.strip()
            if value and value not in unique_matches:
                unique_matches.append(value)
        return unique_matches

    def _extract_two_date_pair(self, question: str) -> tuple[date, date, str | None] | None:
        pair_match = re.search(
            rf"({TEXTUAL_DATE_PATTERN}|{NUMERIC_DATE_PATTERN})\s+и\s+({TEXTUAL_DATE_PATTERN}|{NUMERIC_DATE_PATTERN})",
            question,
        )
        if not pair_match:
            return None

        first_date, first_note = self._parse_single_date(pair_match.group(1))
        second_date, second_note = self._parse_single_date(pair_match.group(2))
        if not first_date or not second_date:
            return None

        notes = [item for item in [first_note, second_note] if item]
        note = " ".join(notes) if notes else "Запрос с двумя датами интерпретирован как период от первой даты до второй."
        return first_date, second_date, note

    def _extract_explicit_dates(self, question: str) -> tuple[list[date], list[str]]:
        collected: list[date] = []
        notes: list[str] = []

        shared_month_match = re.search(
            rf"(\d{{1,2}})\s*(?:,|и)\s*(\d{{1,2}})\s+({MONTH_PATTERN})(?:\s+(\d{{4}}))?",
            question,
        )
        if shared_month_match:
            day_a = int(shared_month_match.group(1))
            day_b = int(shared_month_match.group(2))
            month_token = shared_month_match.group(3)
            explicit_year = int(shared_month_match.group(4)) if shared_month_match.group(4) else None
            month = MONTHS.get(month_token)
            if month:
                year = explicit_year or date.today().year
                for day in (day_a, day_b):
                    parsed = self._safe_date(year, month, day)
                    if parsed and parsed not in collected:
                        collected.append(parsed)
                if explicit_year is None:
                    notes.append(f"Для дат без года использован текущий год: {year}.")

        for token in self._extract_discrete_dates(question):
            parsed_date, note = self._parse_single_date(token)
            if not parsed_date:
                continue
            if parsed_date not in collected:
                collected.append(parsed_date)
            if note and note not in notes:
                notes.append(note)

        collected.sort()
        return collected, notes

    def _parse_date_range(self, start_text: str, end_text: str) -> tuple[date, date, str | None]:
        anchor = date.today()
        start_parts = self._parse_date_parts(start_text)
        end_parts = self._parse_date_parts(end_text)

        if not start_parts or not end_parts:
            return anchor, anchor, None

        start_day, start_month, start_year = start_parts
        end_day, end_month, end_year = end_parts

        if start_year is not None and end_year is not None:
            start_date = self._safe_date(start_year, start_month, start_day)
            end_date = self._safe_date(end_year, end_month, end_day)
        elif end_year is not None:
            end_date = self._safe_date(end_year, end_month, end_day)
            inferred_start_year = end_year if (start_month, start_day) <= (end_month, end_day) else end_year - 1
            start_date = self._safe_date(inferred_start_year, start_month, start_day)
        elif start_year is not None:
            start_date = self._safe_date(start_year, start_month, start_day)
            inferred_end_year = start_year if (end_month, end_day) >= (start_month, start_day) else start_year + 1
            end_date = self._safe_date(inferred_end_year, end_month, end_day)
        else:
            base_year = date.today().year
            end_date = self._safe_date(base_year, end_month, end_day)
            inferred_start_year = base_year if (start_month, start_day) <= (end_month, end_day) else base_year - 1
            start_date = self._safe_date(inferred_start_year, start_month, start_day)

        if not start_date or not end_date:
            return anchor, anchor, None

        if end_date < start_date:
            shifted_end = self._safe_date(start_date.year + 1, end_date.month, end_date.day)
            end_date = shifted_end or end_date

        note = None
        if start_year is None and end_year is None:
            note = f"Период интерпретирован как {start_date.isoformat()} - {end_date.isoformat()} с использованием текущего года."
        elif start_year is None or end_year is None:
            note = f"Период интерпретирован как {start_date.isoformat()} - {end_date.isoformat()} с учётом указанной части даты."

        return start_date, end_date, note

    def _parse_single_date(self, text: str) -> tuple[date | None, str | None]:
        parts = self._parse_date_parts(text)
        if not parts:
            return None, None

        day, month, explicit_year = parts
        if explicit_year is not None:
            parsed_date = self._safe_date(explicit_year, month, day)
            return parsed_date, None

        current_year = date.today().year
        parsed_date = self._safe_date(current_year, month, day)
        if not parsed_date:
            return None, None
        return parsed_date, f"Период интерпретирован как {parsed_date.isoformat()} с использованием текущего года."

    def _parse_date_parts(self, text: str) -> tuple[int, int, int | None] | None:
        stripped = text.strip()

        numeric_match = re.fullmatch(r"(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?", stripped)
        if numeric_match:
            day = int(numeric_match.group(1))
            month = int(numeric_match.group(2))
            year = self._normalize_year_token(numeric_match.group(3))
            return day, month, year

        match = re.fullmatch(rf"(\d{{1,2}})\s+({MONTH_PATTERN})(?:\s+(\d{{4}}))?", stripped)
        if not match:
            return None

        day = int(match.group(1))
        month_token = match.group(2)
        month = MONTHS.get(month_token)
        year = int(match.group(3)) if match.group(3) else None
        if not month:
            return None

        return day, month, year

    def _normalize_year_token(self, token: str | None) -> int | None:
        if not token:
            return None
        if len(token) == 2:
            return 2000 + int(token)
        return int(token)

    def _safe_date(self, year: int, month: int, day: int) -> date | None:
        try:
            return date(year, month, day)
        except ValueError:
            return None

    def _pretty_date_label(self, text: str, parsed_date: date | None = None) -> str:
        normalized = text[:1].upper() + text[1:]
        if re.search(r"\d{4}", text):
            return normalized
        if parsed_date is None:
            return normalized
        return f"{normalized} {parsed_date.year}"

    def _build_month_label(self, month_token: str, year: int) -> str:
        return f"{month_token[:1].upper() + month_token[1:]} {year}"

    def _normalize_text(self, value: str) -> str:
        normalized = value.lower().replace("ё", "е")
        normalized = re.sub(r"[,;:!\?\(\)\[\]\{\}\"'`]+", " ", normalized)
        normalized = re.sub(r"\b(пожалуйста|плиз|будьте добры|будь добра|будь добр|мне надо|мне нужно|мне бы|покажи мне|посмотри|глянь|скажи)\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()
