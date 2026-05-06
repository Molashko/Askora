from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.models.user import User
from app.models.data_source import DataSource
from app.schemas.common import MessageResponse
from app.schemas.query import (
    DatasetContext,
    InterpretationFeedbackRequest,
    QueryExampleCreateRequest,
    QueryExampleSummary,
    QueryHistoryItem,
    QueryRequest,
    QueryResult,
    QueryTemplateSummary,
)
from app.semantic_layer.loader import semantic_loader
from app.services.query_service import QueryService
from app.services.rate_limit_service import query_rate_limiter
from app.services.metrics_service import metrics_service

router = APIRouter()


@router.get("/dataset-context", response_model=DatasetContext)
def dataset_context(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DatasetContext:
    _ = user
    source = (
        db.query(DataSource)
        .filter(DataSource.is_active.is_(True), DataSource.is_default.is_(True))
        .order_by(DataSource.updated_at.desc())
        .first()
    )
    catalog = semantic_loader.load_catalog_for_db(db)
    capabilities = source.capabilities_json if source and isinstance(source.capabilities_json, dict) else {}
    guidance = capabilities.get("query_guidance") if isinstance(capabilities.get("query_guidance"), dict) else {}
    metric_items = sorted(catalog.metrics.items(), key=lambda item: _metric_sort_key(item[0]))
    dimension_items = sorted(catalog.dimensions.items(), key=lambda item: _dimension_sort_key(item[0], item[1].label))
    metrics = [item.label for _, item in metric_items]
    dimensions = [item.label for _, item in dimension_items]
    columns = [
        str(item.get("original_name") or item.get("name"))
        for item in capabilities.get("columns", [])
        if isinstance(item, dict) and (item.get("original_name") or item.get("name"))
    ]
    quick_fragments = _guidance_list(guidance, "quick_fragments") or _build_quick_fragments(metrics, dimensions)
    quick_questions = _guidance_list(guidance, "quick_questions") or _build_quick_questions(metrics, dimensions)
    composing_hints = _guidance_list(guidance, "composing_hints") or _build_composing_hints(metrics, dimensions, columns)
    return DatasetContext(
        key=catalog.base_dataset,
        name=source.name if source else catalog.base_dataset,
        filename=capabilities.get("uploaded_filename") if capabilities else ("train.csv" if catalog.base_dataset == "order_tender_facts" else None),
        row_count=capabilities.get("row_count") if capabilities else None,
        is_uploaded_csv=capabilities.get("kind") == "uploaded_csv",
        metrics=metrics[:20],
        dimensions=dimensions[:20],
        columns=columns[:40],
        quick_fragments=quick_fragments[:14],
        quick_questions=quick_questions[:8],
        composing_hints=composing_hints[:4],
        llm_guidance_used=bool(guidance.get("llm_used")),
    )


def _guidance_list(guidance: dict, key: str) -> list[str]:
    value = guidance.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _build_quick_fragments(metrics: list[str], dimensions: list[str]) -> list[str]:
    metric_items = [_plain_metric_label(item) for item in metrics[:6]]
    dimension_items = [f"по {_plain_dimension_label(item)}" for item in dimensions[:5]]
    periods = ["за вчера", "за прошлую неделю", "за текущий месяц"]
    return _dedupe([*metric_items, *dimension_items, *periods])


def _metric_sort_key(key: str) -> tuple[int, str]:
    lowered = key.lower()
    looks_like_identifier = lowered.endswith("_id") or "_id_" in lowered or lowered in {"sum_id", "avg_id"}
    if looks_like_identifier:
        return (8, key)
    if any(token in lowered for token in ["orders", "rides", "revenue", "amount", "price", "cnt", "count"]):
        if key.startswith("sum_"):
            return (0, key)
        if key.startswith("avg_"):
            return (3, key)
    if key.startswith("sum_"):
        return (1, key)
    if key == "rows_count" or key.startswith("count_"):
        return (2, key)
    if key.startswith("avg_"):
        return (4, key)
    return (3, key)


def _dimension_sort_key(key: str, label: str) -> tuple[int, str]:
    lowered = f"{key} {label}".lower()
    if key == "order_date":
        return (0, key)
    if key in {"order_week", "order_month"}:
        return (1, key)
    if any(token in lowered for token in ["city", "город", "driver", "водител", "user", "пользов"]):
        return (2, key)
    if any(token in lowered for token in ["status", "статус", "source", "источник"]):
        return (3, key)
    if key.startswith("dim_"):
        return (4, key)
    return (5, key)


def _build_quick_questions(metrics: list[str], dimensions: list[str]) -> list[str]:
    primary_metric = _plain_metric_label(metrics[0]) if metrics else "количество строк"
    second_metric = _plain_metric_label(metrics[1]) if len(metrics) > 1 else primary_metric
    time_dimension = next((item for item in dimensions if "день" in item.lower()), dimensions[0] if dimensions else "")
    category_dimension = next((item for item in dimensions if "день" not in item.lower() and "недел" not in item.lower() and "месяц" not in item.lower()), "")
    questions = [
        f"Покажи {primary_metric} по дням за прошлую неделю",
        f"Покажи {second_metric} за текущий месяц",
    ]
    if category_dimension:
        questions.append(f"Покажи {primary_metric} по {_plain_dimension_label(category_dimension)} за прошлую неделю")
    if time_dimension:
        questions.append(f"Покажи динамику {primary_metric} по {_plain_dimension_label(time_dimension)}")
    return _dedupe(questions)


def _build_composing_hints(metrics: list[str], dimensions: list[str], columns: list[str]) -> list[str]:
    metric_hint = ", ".join(_plain_metric_label(item) for item in metrics[:5]) or "количество строк"
    dimension_hint = ", ".join(_plain_dimension_label(item) for item in dimensions[:5]) or "дни, категории"
    column_hint = ", ".join(columns[:6])
    hints = [
        f"Что считать: {metric_hint}.",
        f"Как разбить: {dimension_hint}.",
        "За какой период: за вчера, за прошлую неделю, за текущий месяц, с даты по дату.",
    ]
    if column_hint:
        hints.append(f"Колонки датасета: {column_hint}.")
    return hints


def _plain_metric_label(label: str) -> str:
    cleaned = label.replace("Сумма ", "").replace("Среднее ", "среднее ")
    return cleaned.strip()


def _plain_dimension_label(label: str) -> str:
    return label.replace(" (день)", "").replace(" (неделя)", "").replace(" (месяц)", "").strip()


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


@router.post("/run", response_model=QueryResult)
def run_query(
    payload: QueryRequest,
    response: Response,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueryResult:
    decision = query_rate_limiter.check(f"{user.id}:query_run")
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    if not decision.allowed:
        metrics_service.observe_rate_limit_block()
        response.headers["Retry-After"] = str(decision.retry_after_seconds)
        raise HTTPException(
            status_code=429,
            detail=(
                "Слишком много запросов за короткое время. "
                f"Лимит: {decision.limit} за окно. Повторите через {decision.retry_after_seconds} сек."
            ),
        )
    return QueryService(db).run(payload, user)


@router.post("/interpretation-feedback", response_model=MessageResponse)
def interpretation_feedback(
    payload: InterpretationFeedbackRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MessageResponse:
    try:
        return QueryService(db).record_interpretation_feedback(
            question=payload.question,
            helpful=payload.helpful,
            user=user,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/history", response_model=list[QueryHistoryItem])
def list_history(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[QueryHistoryItem]:
    items = QueryService(db).list_history(user)
    return [QueryHistoryItem.model_validate(item, from_attributes=True) for item in items]


@router.delete("/history/{history_id}", response_model=MessageResponse)
def delete_history_item(
    history_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MessageResponse:
    deleted = QueryService(db).delete_history_item(history_id, user)
    if not deleted:
        raise HTTPException(status_code=404, detail="Запись истории не найдена")
    return MessageResponse(message="Запись из истории удалена")


@router.delete("/history", response_model=MessageResponse)
def clear_history(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> MessageResponse:
    deleted = QueryService(db).clear_history(user)
    return MessageResponse(message=f"История очищена. Удалено записей: {deleted}")


@router.get("/examples", response_model=list[QueryExampleSummary])
def list_examples(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[QueryExampleSummary]:
    items = QueryService(db).list_examples(user)
    return [QueryExampleSummary.model_validate(item, from_attributes=True) for item in items]


@router.get("/templates", response_model=list[QueryTemplateSummary])
def list_templates(user: User = Depends(get_current_user)) -> list[QueryTemplateSummary]:
    _ = user
    templates = semantic_loader.load_templates()
    response: list[QueryTemplateSummary] = []
    for item in templates.templates:
        response.append(
            QueryTemplateSummary(
                name=item.get("name", ""),
                description=item.get("description", ""),
                example_question=item.get("example_question", ""),
                pattern=item.get("pattern", ""),
                guidance=item.get("guidance", ""),
                output_shape_json=item.get("output_shape", {}) or {},
            )
        )
    return response


@router.post("/examples", response_model=QueryExampleSummary)
def create_example(
    payload: QueryExampleCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueryExampleSummary:
    item = QueryService(db).create_example(payload, user)
    return QueryExampleSummary.model_validate(item, from_attributes=True)


@router.delete("/examples/{example_id}", response_model=MessageResponse)
def delete_example(example_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> MessageResponse:
    deleted = QueryService(db).delete_example(example_id, user)
    if not deleted:
        raise HTTPException(status_code=404, detail="Пример не найден")
    return MessageResponse(message="Пример удалён")
