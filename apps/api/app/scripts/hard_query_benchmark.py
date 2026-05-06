from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import random
import re
import sys
from time import perf_counter
from typing import Any

from app.scripts.query_regression import build_cases as build_regression_cases


def _ascii_progress_bar(current: int, total: int, width: int = 34) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = int(round(width * current / total))
    filled = min(width, max(0, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass(frozen=True)
class BenchmarkCase:
    question: str
    category: str
    expected_status: str = "executed"
    expected_metrics: tuple[str, ...] = ()
    expected_dimensions: tuple[str, ...] = ()
    expected_sort: str | None = None
    expected_limit: int | None = None
    required_filters: tuple[tuple[str, str, Any], ...] = ()


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("ё", "е").split())


def _build_supported_base_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    for item in build_regression_cases():
        cases.append(
            BenchmarkCase(
                question=item.question,
                category="regression",
                expected_status=item.expected_status,
                expected_metrics=item.expected_metrics,
                expected_dimensions=item.expected_dimensions,
            )
        )

    ranking_metric_phrases = [
        ("total_revenue", "выручке"),
        ("total_orders", "количеству заказов"),
        ("cancelled_orders", "количеству отмен"),
        ("completed_orders", "количеству выполненных заказов"),
        ("avg_order_price", "среднему чеку"),
    ]
    ranking_entities = [
        ("order_date", "день"),
        ("order_hour", "час"),
        ("city_id", "город"),
        ("order_status", "статус заказа"),
        ("cancel_source", "источник отмен"),
        ("order_dow", "день недели"),
    ]
    time_phrases = [
        "за март",
        "за апрель",
        "за прошлую неделю",
        "за текущую неделю",
        "в этом месяце",
        "за прошлый месяц",
        "за вчера",
        "за 2026 год",
        "с начала года",
        "за I квартал",
        "за второй квартал",
    ]

    for metric_key, metric_phrase in ranking_metric_phrases:
        for dimension_key, entity_phrase in ranking_entities:
            for time_phrase in time_phrases:
                cases.append(
                    BenchmarkCase(
                        question=f"{entity_phrase.capitalize()} с самой большой метрикой по {metric_phrase} {time_phrase}",
                        category="ranking_max",
                        expected_metrics=(metric_key,),
                        expected_dimensions=(dimension_key,),
                        expected_sort=f"{metric_key} DESC",
                        expected_limit=1,
                    )
                )
                cases.append(
                    BenchmarkCase(
                        question=f"{entity_phrase.capitalize()} с самой маленькой метрикой по {metric_phrase} {time_phrase}",
                        category="ranking_min",
                        expected_metrics=(metric_key,),
                        expected_dimensions=(dimension_key,),
                        expected_sort=f"{metric_key} ASC",
                        expected_limit=1,
                    )
                )
                cases.append(
                    BenchmarkCase(
                        question=f"Топ-3 {entity_phrase}а по {metric_phrase} {time_phrase}",
                        category="ranking_top3",
                        expected_metrics=(metric_key,),
                        expected_dimensions=(dimension_key,),
                        expected_sort=f"{metric_key} DESC",
                        expected_limit=3,
                    )
                )
                cases.append(
                    BenchmarkCase(
                        question=f"Антитоп-3 {entity_phrase}а по {metric_phrase} {time_phrase}",
                        category="ranking_bottom3",
                        expected_metrics=(metric_key,),
                        expected_dimensions=(dimension_key,),
                        expected_sort=f"{metric_key} ASC",
                        expected_limit=3,
                    )
                )
                cases.append(
                    BenchmarkCase(
                        question=f"Лучший {entity_phrase} по {metric_phrase} {time_phrase}",
                        category="ranking_best",
                        expected_metrics=(metric_key,),
                        expected_dimensions=(dimension_key,),
                        expected_sort=f"{metric_key} DESC",
                        expected_limit=1,
                    )
                )
                cases.append(
                    BenchmarkCase(
                        question=f"Худший {entity_phrase} по {metric_phrase} {time_phrase}",
                        category="ranking_worst",
                        expected_metrics=(metric_key,),
                        expected_dimensions=(dimension_key,),
                        expected_sort=f"{metric_key} ASC",
                        expected_limit=1,
                    )
                )
                cases.append(
                    BenchmarkCase(
                        question=f"Лучшие 3 {entity_phrase}а по {metric_phrase} {time_phrase}",
                        category="ranking_best3",
                        expected_metrics=(metric_key,),
                        expected_dimensions=(dimension_key,),
                        expected_sort=f"{metric_key} DESC",
                        expected_limit=3,
                    )
                )
                cases.append(
                    BenchmarkCase(
                        question=f"Худшие 3 {entity_phrase}а по {metric_phrase} {time_phrase}",
                        category="ranking_worst3",
                        expected_metrics=(metric_key,),
                        expected_dimensions=(dimension_key,),
                        expected_sort=f"{metric_key} ASC",
                        expected_limit=3,
                    )
                )

    complex_supported_cases = [
        BenchmarkCase(
            question="Выручка с 6 часов до 18 часов 15 марта по часам",
            category="hour_window",
            expected_metrics=("total_revenue",),
            expected_dimensions=("order_hour",),
            required_filters=(("order_hour", "gte", 6), ("order_hour", "lte", 18)),
        ),
        BenchmarkCase(
            question="Количество заказов с 7 часов до 10 часов за прошлую неделю по дням",
            category="hour_window",
            expected_metrics=("total_orders",),
            expected_dimensions=("order_date",),
            required_filters=(("order_hour", "gte", 7), ("order_hour", "lte", 10)),
        ),
        BenchmarkCase(
            question="Средний чек в будни за прошлую неделю по дням",
            category="weekday_filter",
            expected_metrics=("avg_order_price",),
            expected_dimensions=("order_date",),
            required_filters=(("order_dow", "in", [1, 2, 3, 4, 5]),),
        ),
        BenchmarkCase(
            question="Отмены в выходные за март по дням",
            category="weekend_filter",
            expected_metrics=("cancelled_orders",),
            expected_dimensions=("order_date",),
            required_filters=(("order_dow", "in", [0, 6]),),
        ),
        BenchmarkCase(
            question="Средняя длительность больше 12 минут за прошлый месяц по дням",
            category="duration_filter",
            expected_metrics=("avg_duration_min",),
            expected_dimensions=("order_date",),
            required_filters=(("duration_seconds", "gt", 720),),
        ),
        BenchmarkCase(
            question="Средняя длительность не более 15 минут за прошлую неделю по дням",
            category="duration_filter",
            expected_metrics=("avg_duration_min",),
            expected_dimensions=("order_date",),
            required_filters=(("duration_seconds", "lte", 900),),
        ),
        BenchmarkCase(
            question="Выручка и выполненные заказы по городам за март",
            category="multi_metric",
            expected_metrics=("total_revenue", "completed_orders"),
            expected_dimensions=("city_id",),
        ),
        BenchmarkCase(
            question="Сравни выручку за 12 марта и 18 марта по часам",
            category="multi_date",
            expected_metrics=("total_revenue",),
            expected_dimensions=("order_hour", "order_date"),
        ),
        BenchmarkCase(
            question="Покажи количество заказов по статусам заказа за прошлый месяц",
            category="status_breakdown",
            expected_metrics=("total_orders",),
            expected_dimensions=("order_status",),
        ),
        BenchmarkCase(
            question="Покажи количество тендеров по статусам тендера за март",
            category="status_breakdown",
            expected_metrics=("total_tenders",),
            expected_dimensions=("tender_status",),
        ),
        BenchmarkCase(
            question="Выручка и количество заказов по городам за прошлую неделю",
            category="multi_metric_geo",
            expected_metrics=("total_revenue", "total_orders"),
            expected_dimensions=("city_id",),
        ),
        BenchmarkCase(
            question="Заказы по часам за прошлый месяц",
            category="hour_month_orders",
            expected_metrics=("total_orders",),
            expected_dimensions=("order_hour",),
        ),
        BenchmarkCase(
            question="Средняя дистанция по городам за март",
            category="avg_distance_geo",
            expected_metrics=("avg_distance_km",),
            expected_dimensions=("city_id",),
        ),
        BenchmarkCase(
            question="Доля успешных тендеров за прошлый месяц по дням",
            category="acceptance_daily",
            expected_metrics=("tender_acceptance_rate",),
            expected_dimensions=("order_date",),
        ),
        BenchmarkCase(
            question="Средний чек по городам за прошлую неделю",
            category="avg_check_geo",
            expected_metrics=("avg_order_price",),
            expected_dimensions=("city_id",),
        ),
        BenchmarkCase(
            question="Заказы по неделям с начала года",
            category="orders_by_week_ytd",
            expected_metrics=("total_orders",),
            expected_dimensions=("order_week",),
        ),
        BenchmarkCase(
            question="Отклонённые тендеры за вчера",
            category="declined_tenders_plain",
            expected_metrics=("declined_tenders",),
        ),
        BenchmarkCase(
            question="Успешные тендеры по дням за текущую неделю",
            category="successful_tenders_daily",
            expected_metrics=("successful_tenders",),
            expected_dimensions=("order_date",),
        ),
    ]
    cases.extend(complex_supported_cases)
    return cases


def _build_unsupported_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase("Прибыль по каналам за март", "unsupported", expected_status="needs_clarification"),
        BenchmarkCase("Покажи LTV клиентов за квартал", "unsupported", expected_status="needs_clarification"),
        BenchmarkCase("Удалить данные за март", "destructive", expected_status="needs_clarification"),
        BenchmarkCase("Обнови статусы заказов в базе", "destructive", expected_status="needs_clarification"),
        BenchmarkCase("Падение", "ambiguous", expected_status="needs_clarification"),
        BenchmarkCase("Почему всё стало хуже", "ambiguous", expected_status="needs_clarification"),
        BenchmarkCase("Покажи продажи айфонов за март", "out_of_domain", expected_status="needs_clarification"),
        BenchmarkCase("Сравни конверсию по рекламным каналам", "unsupported", expected_status="needs_clarification"),
        BenchmarkCase("Маржа по городам за март", "unsupported", expected_status="needs_clarification"),
        BenchmarkCase("Удалить все отмены", "destructive", expected_status="needs_clarification"),
    ]


def _mutate_question(question: str, rng: random.Random) -> str:
    replacements = [
        ("покажи", "выведи"),
        ("сравни", "сделай сравнение"),
        ("выручка", "оборот"),
        ("заказы", "ордера"),
        ("тендеры", "офферы"),
        ("отмены", "срывы"),
        ("выполненные", "завершенные"),
        ("прошлую неделю", "предыдущую неделю"),
        ("текущую неделю", "эту неделю"),
        ("прошлый месяц", "предыдущий месяц"),
        ("в этом месяце", "за текущий месяц"),
        ("количество", "число"),
        ("по дням", "в разбивке по дням"),
        ("по часам", "в разрезе часов"),
        ("по городам", "в разрезе городов"),
        ("топ-3", "топ 3"),
        ("антитоп-3", "антитоп 3"),
    ]
    typo_replacements = [
        ("выручка", "виручка"),
        ("оборот", "обоот"),
        ("заказы", "закзы"),
        ("отмен", "атмен"),
        ("сравни", "сровни"),
        ("март", "мрат"),
        ("неделю", "ниделю"),
        ("месяц", "месец"),
        ("час", "чяс"),
        ("город", "гоорд"),
    ]
    prefixes = ["", "подскажи ", "можешь ", "хочу понять ", "пожалуйста "]
    suffixes = ["", " пожалуйста", " плиз", ", если можно"]

    text = question.lower()
    for old, new in replacements:
        if old in text and rng.random() < 0.45:
            text = text.replace(old, new, 1)
    for old, new in typo_replacements:
        if old in text and rng.random() < 0.28:
            text = text.replace(old, new, 1)
    if rng.random() < 0.2 and len(text) > 8:
        idx = rng.randint(2, len(text) - 3)
        text = text[:idx] + text[idx + 1 :]
    return f"{rng.choice(prefixes)}{text}{rng.choice(suffixes)}".strip()


def build_cases(target_count: int = 1000, *, seed: int = 42, quiet: bool = False) -> list[BenchmarkCase]:
    rng = random.Random(seed)
    supported = _build_supported_base_cases()
    unsupported = _build_unsupported_cases()
    seed_cases = [*supported, *unsupported]

    by_normalized: dict[str, BenchmarkCase] = {}

    def add_case(case: BenchmarkCase) -> None:
        normalized = _normalize(case.question)
        if normalized not in by_normalized:
            by_normalized[normalized] = case

    for case in seed_cases:
        add_case(case)

    mutation_pool = list(seed_cases)
    index = 0
    report_step = max(1, min(500, target_count // 40))
    last_reported = len(by_normalized)
    while len(by_normalized) < target_count:
        base = mutation_pool[index % len(mutation_pool)]
        index += 1
        mutated = BenchmarkCase(
            question=_mutate_question(base.question, rng),
            category=f"{base.category}_mutated",
            expected_status=base.expected_status,
            expected_metrics=base.expected_metrics,
            expected_dimensions=base.expected_dimensions,
            expected_sort=base.expected_sort,
            expected_limit=base.expected_limit,
            required_filters=base.required_filters,
        )
        add_case(mutated)
        cur = len(by_normalized)
        if not quiet and target_count >= 80 and (cur - last_reported >= report_step or cur == target_count):
            bar = _ascii_progress_bar(cur, target_count)
            sys.stderr.write(f"\rСбор уникальных формулировок  {bar}  {cur}/{target_count}  мутаций {index}   ")
            sys.stderr.flush()
            last_reported = cur
        # При большом target_count мутации часто дают дубликаты нормализации — поднимаем потолок итераций.
        if index > max(target_count * 120, 200_000):
            break

    if not quiet and target_count >= 80:
        sys.stderr.write("\n")

    cases = list(by_normalized.values())
    return cases[:target_count]


def _compare_filters(expected: tuple[tuple[str, str, Any], ...], actual_filters: list[tuple[str, str, Any]]) -> list[str]:
    failures: list[str] = []
    for item in expected:
        if item not in actual_filters:
            failures.append(f"Не найден фильтр {item}, фактически: {actual_filters}")
    return failures


def run(*, target_count: int = 1000, json_out: str | None = None, seed: int = 42, quiet: bool = False) -> int:
    from app.db.session import SessionLocal
    from app.repositories.users import UserRepository
    from app.schemas.query import QueryRequest
    from app.services.query_service import QueryService

    if not quiet:
        print(f"Генерация до {target_count} уникальных кейсов (seed={seed})…", flush=True)
    cases = build_cases(target_count=target_count, seed=seed, quiet=quiet)
    db = SessionLocal()
    try:
        user = UserRepository(db).get_by_email("business@demo.local")
        if not user:
            raise RuntimeError("Не найден demo-пользователь business@demo.local")

        service = QueryService(db)
        failures: list[dict[str, Any]] = []
        durations_ms: list[float] = []
        actual_status_distribution: dict[str, int] = {}
        category_stats: dict[str, dict[str, int]] = {}
        executed_total = 0

        total_cases = len(cases)
        progress_step = max(1, total_cases // 60)

        for i, case in enumerate(cases, start=1):
            started = perf_counter()
            category_bucket = category_stats.setdefault(case.category, {"total": 0, "correct": 0})
            category_bucket["total"] += 1
            try:
                result = service.run(QueryRequest(question=case.question), user)
            except Exception as exc:
                db.rollback()
                durations_ms.append(round((perf_counter() - started) * 1000, 2))
                actual_status_distribution["failed_exception"] = actual_status_distribution.get("failed_exception", 0) + 1
                failures.append(
                    {
                        "question": case.question,
                        "category": case.category,
                        "expected_status": case.expected_status,
                        "actual_status": "failed_exception",
                        "expected_metrics": list(case.expected_metrics),
                        "actual_metrics": [],
                        "expected_dimensions": list(case.expected_dimensions),
                        "actual_dimensions": [],
                        "expected_sort": case.expected_sort,
                        "actual_sort": None,
                        "expected_limit": case.expected_limit,
                        "actual_limit": None,
                        "required_filters": list(case.required_filters),
                        "actual_filters": [],
                        "reasons": [f"Исключение во время выполнения: {exc}"],
                    }
                )
                continue

            durations_ms.append(round((perf_counter() - started) * 1000, 2))
            actual_status_distribution[result.status] = actual_status_distribution.get(result.status, 0) + 1
            if result.status == "executed":
                executed_total += 1

            actual_metrics = tuple(item.key for item in result.query_plan.metrics)
            actual_dimensions = tuple(item.key for item in result.query_plan.dimensions)
            actual_filters = [(item.key, item.operator, item.value) for item in result.query_plan.filters]
            mismatch_reasons: list[str] = []

            if result.status != case.expected_status:
                mismatch_reasons.append(f"Статус {result.status} вместо {case.expected_status}")
            if case.expected_metrics and not all(item in actual_metrics for item in case.expected_metrics):
                mismatch_reasons.append(f"Метрики {actual_metrics} вместо ожидаемых {case.expected_metrics}")
            if case.expected_dimensions and not all(item in actual_dimensions for item in case.expected_dimensions):
                mismatch_reasons.append(f"Измерения {actual_dimensions} вместо ожидаемых {case.expected_dimensions}")
            if case.expected_sort and result.query_plan.sort != case.expected_sort:
                mismatch_reasons.append(f"Сортировка {result.query_plan.sort} вместо {case.expected_sort}")
            if case.expected_limit is not None and result.query_plan.limit != case.expected_limit:
                mismatch_reasons.append(f"Лимит {result.query_plan.limit} вместо {case.expected_limit}")
            mismatch_reasons.extend(_compare_filters(case.required_filters, actual_filters))

            if mismatch_reasons:
                failures.append(
                    {
                        "question": case.question,
                        "category": case.category,
                        "expected_status": case.expected_status,
                        "actual_status": result.status,
                        "expected_metrics": list(case.expected_metrics),
                        "actual_metrics": list(actual_metrics),
                        "expected_dimensions": list(case.expected_dimensions),
                        "actual_dimensions": list(actual_dimensions),
                        "expected_sort": case.expected_sort,
                        "actual_sort": result.query_plan.sort,
                        "expected_limit": case.expected_limit,
                        "actual_limit": result.query_plan.limit,
                        "required_filters": list(case.required_filters),
                        "actual_filters": actual_filters,
                        "reasons": mismatch_reasons,
                    }
                )
                continue

            category_bucket["correct"] += 1

            if not quiet and (i == 1 or i == total_cases or i % progress_step == 0):
                bar = _ascii_progress_bar(i, total_cases)
                pct = 100.0 * i / total_cases if total_cases else 0.0
                sys.stdout.write(
                    f"\rВыполнение  {bar}  {i}/{total_cases} ({pct:5.1f}%)  "
                    f"несовпадений {len(failures)}  статус executed: {executed_total}   "
                )
                sys.stdout.flush()

        if not quiet:
            sys.stdout.write("\n")

        total = len(cases)
        passed = total - len(failures)
        pass_rate = round((passed / total) * 100, 2) if total else 0.0
        avg_latency_ms = round(sum(durations_ms) / len(durations_ms), 2) if durations_ms else 0.0
        expected_status_distribution: dict[str, int] = {}
        for item in cases:
            expected_status_distribution[item.expected_status] = expected_status_distribution.get(item.expected_status, 0) + 1

        report = {
            "total": total,
            "passed": passed,
            "failed": len(failures),
            "pass_rate": pass_rate,
            "avg_latency_ms": avg_latency_ms,
            "executed_total_actual": executed_total,
            "executed_total_expected": expected_status_distribution.get("executed", 0),
            "expected_status_distribution": expected_status_distribution,
            "actual_status_distribution": actual_status_distribution,
            "category_stats": category_stats,
            "failures": failures,
        }

        print(f"Проверено запросов: {total}")
        print(f"Успешно по ожиданиям: {passed}")
        print(f"Ошибок: {len(failures)}")
        print(f"Pass rate: {pass_rate}%")
        print(f"Средняя latency: {avg_latency_ms} ms")
        print(f"Фактически выполнено запросов: {executed_total}")

        if json_out:
            with open(json_out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
            print(f"JSON-отчёт сохранён: {json_out}")

        if failures:
            print("")
            print("Первые проблемные кейсы:")
            for item in failures[:20]:
                print(f"- {item['question']} -> {'; '.join(item['reasons'])}")
        return 1 if failures else 0
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 1000 hard NL queries before/after parser tuning.")
    parser.add_argument("--target-count", type=int, default=1000, help="Сколько кейсов сгенерировать и проверить.")
    parser.add_argument("--json-out", dest="json_out", help="Куда сохранить JSON-отчёт.")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимого набора.")
    parser.add_argument("--quiet", action="store_true", help="Без прогресс-строк и этапа генерации.")
    args = parser.parse_args()
    raise SystemExit(
        run(target_count=args.target_count, json_out=args.json_out, seed=args.seed, quiet=args.quiet)
    )
