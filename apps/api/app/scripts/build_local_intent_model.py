from __future__ import annotations

import ast
import argparse
import json
import random
import re
import sys
from pathlib import Path

from app.ai.percent_change import is_percent_change_request
from app.scripts.hard_query_benchmark import build_cases as build_hard_benchmark_cases

def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("ё", "е").split())


PREFIXES = [
    "",
    "Подскажи, ",
    "Покажи, ",
    "Можешь показать, ",
    "Хочу понять, ",
    "А покажи, ",
    "Нужно: ",
    "Срочно ",
    "Дай ",
    "Выведи ",
]
SUFFIXES = ["", " пожалуйста", " плиз", ", если можно", " в дашборде", " для отчёта"]
SYNONYM_GROUPS = [
    ["покажи", "выведи", "дай", "покажи"],
    ["сколько", "какое количество", "скока"],
    ["выручка", "оборот", "виручка"],
    ["выполненные", "завершенные", "выполненые"],
    ["отмены", "срывы", "атмены"],
    ["по дням", "в разбивке по дням", "по деням"],
    ["по часам", "в разбивке по часам", "по чесам"],
    ["сравни", "сделай сравнение", "сровни"],
    ["прошлую неделю", "предыдущую неделю", "прошлую ниделю"],
    ["текущую неделю", "эту неделю", "текущию неделю"],
    ["в этом месяце", "за текущий месяц", "в етом месяце"],
    ["прошлый месяц", "предыдущий месяц", "прошлый месец"],
    ["тендеры", "офферы", "предложения водителям"],
    ["заказы", "ордера", "заявки"],
    ["отмены", "срывы", "отказы"],
    ["текущий год", "этот год", "нынешний год"],
    ["прошлый год", "минувший год", "год назад"],
    ["за вчера", "вчера", "за прошедшие сутки"],
    ["по дням", "в разбивке по дням", "ежедневно"],
    ["по городам", "в разрезе городов", "по городам доставки"],
    ["доля успешных тендеров", "успешность тендеров", "acceptance"],
]
WORD_NOISE_TARGETS = {
    "выручка": ["виручка", "вырука", "вырчка"],
    "оборот": ["обоот", "абарот"],
    "заказы": ["заказы", "закзы"],
    "отмены": ["атмены", "отмны"],
    "сравни": ["сровни", "срани"],
    "тендеров": ["тендиров", "тендерв"],
    "средняя": ["сридняя", "средня"],
    "вчера": ["вчра", "вчераа"],
    "месяце": ["месеце", "месяцэ"],
    "неделе": ["ниделе", "недли"],
    "тендер": ["тендр", "тендар"],
    "город": ["гоорд", "горот"],
    "марте": ["мрате", "мартее"],
    "январь": ["январ", "янваь"],
    "февраль": ["феврал", "февраь"],
    "дистанция": ["дистанцыя", "дистанция"],
}


def _extract_cases(script_path: Path) -> list[dict]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    cases: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "RegressionCase":
            continue
        if not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        item = {
            "question": node.args[0].value,
            "expected_status": "executed",
            "expected_metrics": [],
            "expected_dimensions": [],
        }
        for kw in node.keywords:
            if kw.arg == "expected_status" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    item["expected_status"] = kw.value.value
            elif kw.arg in {"expected_metrics", "expected_dimensions"} and isinstance(kw.value, ast.Tuple):
                values: list[str] = []
                for el in kw.value.elts:
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        values.append(el.value)
                item[kw.arg] = values
        cases.append(item)
    return cases


def _infer_payload(question: str, status: str, metrics: list[str], dimensions: list[str]) -> dict:
    normalized = _normalize(question)
    time_expression = None
    known_time_phrases = [
        "за вчера",
        "вчера",
        "за текущую неделю",
        "на этой неделе",
        "за прошлую неделю",
        "на прошлой неделе",
        "за текущий месяц",
        "в этом месяце",
        "за прошлый месяц",
        "в прошлом месяце",
        "за текущий год",
        "в этом году",
        "за прошлый год",
        "за всё время",
        "за весь период",
        "с начала года",
        "за квартал",
        "за i квартал",
        "за первый квартал",
        "за второй квартал",
        "за март",
        "за апрель",
        "за май",
        "за осень",
        "за зиму",
        "за лето",
        "за весну",
    ]
    for phrase in known_time_phrases:
        if _normalize(phrase) in normalized:
            time_expression = phrase
            break

    comparison_enabled = any(token in normalized for token in ["сравни", "сравнение", "по сравнению", "относительно"])
    if is_percent_change_request(normalized):
        comparison_enabled = True

    intent_type = "aggregation"
    if comparison_enabled:
        intent_type = "comparison"
    elif any(dim in {"order_date", "order_week", "order_month", "order_hour"} for dim in dimensions):
        intent_type = "trend"

    ambiguity_reasons: list[str] = []
    clarification_questions: list[str] = []
    if status == "needs_clarification":
        ambiguity_reasons = ["Запрос требует уточнения или не относится к поддерживаемому домену."]
        clarification_questions = ["Уточните метрику, разрез и период в рамках датасета поездок."]

    return {
        "intent_type": intent_type,
        "metrics": metrics,
        "dimensions": dimensions,
        "filters": [],
        "time_expression": time_expression,
        "time_range_override": None,
        "multi_date": None,
        "comparison": {
            "enabled": comparison_enabled,
            "mode": "previous_period" if comparison_enabled else "none",
            "baseline_label": "Предыдущий период" if comparison_enabled else None,
            "baseline_start_date": None,
            "baseline_end_date": None,
        },
        "preferred_chart_type": None,
        "sort": None,
        "limit": 50,
        "confidence": 0.7 if status != "needs_clarification" else 0.45,
        "ambiguity_reasons": ambiguity_reasons,
        "clarification_questions": clarification_questions,
        "notes": ["Локальная переносимая модель классификации интента."],
    }


def _generate_variants(question: str, *, rng: random.Random, max_variants: int) -> list[str]:
    variants: set[str] = set()
    base = _cleanup_variant(question)
    variants.add(base)

    clean_candidates = {base}
    for phrase in list(clean_candidates):
        for prefix in PREFIXES:
            for suffix in SUFFIXES:
                clean_candidates.add(_cleanup_variant(f"{prefix}{phrase}{suffix}"))

    expanded_candidates = set(clean_candidates)
    for phrase in list(clean_candidates):
        for group in SYNONYM_GROUPS:
            expanded_candidates.update(_swap_synonyms(phrase, group))

    variants.update(_cleanup_variant(item) for item in expanded_candidates if item)

    noisy_candidates: set[str] = set()
    seed_pool = sorted(variants)
    for phrase in seed_pool[: min(len(seed_pool), 36)]:
        noisy_candidates.update(_generate_noisy_versions(phrase, rng))
    variants.update(_cleanup_variant(item) for item in noisy_candidates if item)

    filtered = [item for item in sorted(variants) if item and len(item) >= 3]
    return filtered[:max_variants]


def _swap_synonyms(text: str, group: list[str]) -> set[str]:
    lowered = text.lower()
    variants = {text}
    for source in group:
        if source not in lowered:
            continue
        for target in group:
            replaced = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
            variants.add(replaced)
    return variants


def _generate_noisy_versions(text: str, rng: random.Random) -> set[str]:
    results: set[str] = set()
    lowered = text.lower()

    for source, targets in WORD_NOISE_TARGETS.items():
        if source in lowered:
            for target in targets:
                results.add(re.sub(re.escape(source), target, text, flags=re.IGNORECASE))

    tokens = text.split()
    for index, token in enumerate(tokens):
        stripped = token.strip(",.?!")
        if len(stripped) < 4:
            continue
        for variant in _mutate_token(stripped):
            mutated_tokens = list(tokens)
            mutated_tokens[index] = token.replace(stripped, variant)
            results.add(" ".join(mutated_tokens))
        if len(results) >= 18:
            break

    if len(tokens) >= 4:
        dropped_index = rng.randrange(len(tokens))
        compact = [token for idx, token in enumerate(tokens) if idx != dropped_index]
        results.add(" ".join(compact))

    return {item for item in results if item != text}


def _mutate_token(token: str) -> set[str]:
    variants: set[str] = set()
    if len(token) >= 4:
        variants.add(token[:2] + token[3:] if len(token) > 3 else token)
    if len(token) >= 5:
        variants.add(token[:-1])
    if len(token) >= 4:
        chars = list(token)
        chars[1], chars[2] = chars[2], chars[1]
        variants.add("".join(chars))
    if len(token) >= 4:
        variants.add(token[0] + token[1] + token[1:] )
    if "е" in token:
        variants.add(token.replace("е", "и", 1))
    if "о" in token:
        variants.add(token.replace("о", "а", 1))
    return {item for item in variants if item and item != token}


def _cleanup_variant(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" ,")
    cleaned = re.sub(r"(?i)\b(пожалуйста)\b(?:\s+\1\b)+", r"\1", cleaned)
    cleaned = re.sub(r"(?i)\b(покажи|подскажи|можешь показать|хочу понять)\b(?:,\s*\1\b)+", r"\1", cleaned)
    return cleaned.strip()


def _bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = int(round(width * current / total))
    filled = min(width, max(0, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def run(
    *,
    max_variants_per_case: int = 48,
    seed: int = 42,
    include_hard_benchmark: bool = True,
    hard_benchmark_variants_per_case: int = 8,
    quiet: bool = False,
) -> int:
    script_path = Path(__file__).resolve().with_name("query_regression.py")
    model_path = Path(__file__).resolve().parents[1] / "ai" / "model" / "local_intent_model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    cases = _extract_cases(script_path)
    entries = []
    seen_questions: set[str] = set()
    n_reg = len(cases)
    if not quiet:
        print(f"Регрессия: {n_reg} шаблонов -> варианты (до {max_variants_per_case} на шаблон)...", flush=True)
    for reg_i, item in enumerate(cases, start=1):
        payload = _infer_payload(
            question=item["question"],
            status=item["expected_status"],
            metrics=item["expected_metrics"],
            dimensions=item["expected_dimensions"],
        )
        for variant in _generate_variants(item["question"], rng=rng, max_variants=max_variants_per_case):
            normalized = _normalize(variant)
            if normalized in seen_questions:
                continue
            seen_questions.add(normalized)
            entries.append({"question": variant, "payload": payload})
        if not quiet and n_reg and (reg_i == 1 or reg_i == n_reg or reg_i % max(1, n_reg // 12) == 0):
            sys.stdout.write(f"\r  {_bar(reg_i, n_reg)}  шаблон {reg_i}/{n_reg}  записей в модели: {len(entries)}   ")
            sys.stdout.flush()
    if not quiet and n_reg:
        sys.stdout.write("\n")

    if include_hard_benchmark:
        hb_cases = build_hard_benchmark_cases(target_count=1000, seed=seed)
        hb_n = len(hb_cases)
        if not quiet:
            print(f"Hard benchmark: {hb_n} кейсов -> варианты (до {hard_benchmark_variants_per_case} на кейс)...", flush=True)
        for hb_i, case in enumerate(hb_cases, start=1):
            payload = _infer_payload(
                question=case.question,
                status=case.expected_status,
                metrics=list(case.expected_metrics),
                dimensions=list(case.expected_dimensions),
            )
            payload["sort"] = case.expected_sort
            payload["limit"] = case.expected_limit or payload["limit"]
            payload["filters"] = [
                {"key": key, "operator": operator, "value": value}
                for key, operator, value in case.required_filters
            ]
            for variant in _generate_variants(
                case.question,
                rng=rng,
                max_variants=max(1, hard_benchmark_variants_per_case),
            ):
                normalized = _normalize(variant)
                if normalized in seen_questions:
                    continue
                seen_questions.add(normalized)
                entries.append({"question": variant, "payload": payload})
            if not quiet and hb_n and (hb_i == 1 or hb_i == hb_n or hb_i % max(1, hb_n // 15) == 0):
                sys.stdout.write(f"\r  {_bar(hb_i, hb_n)}  benchmark {hb_i}/{hb_n}  записей: {len(entries)}   ")
                sys.stdout.flush()
        if not quiet and hb_n:
            sys.stdout.write("\n")

    payload = {
        "version": 1,
        "name": "askora-local-intent-v1",
        "description": "Portable NL intent model built from regression templates with paraphrase augmentation.",
        "entries": entries,
    }
    model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    size_mb = model_path.stat().st_size / (1024 * 1024)
    if not quiet:
        print(f"Сохранено: {model_path}")
        print(f"Записей (entries): {len(entries)}  |  размер файла: {size_mb:.2f} МиБ")
    else:
        print(f"{model_path}: {len(entries)} entries, {size_mb:.2f} MiB")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build local intent model with paraphrases and typo augmentation.")
    parser.add_argument(
        "--max-variants-per-case",
        type=int,
        default=48,
        help="Сколько максимум вариантов строить на один regression-case.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимой генерации шумных вариантов.")
    parser.add_argument(
        "--without-hard-benchmark",
        action="store_true",
        help="Не добавлять кейсы из большого benchmark-набора в локальную модель.",
    )
    parser.add_argument(
        "--hard-benchmark-variants-per-case",
        type=int,
        default=8,
        help="Сколько вариантов строить на один hard-benchmark кейс.",
    )
    parser.add_argument("--quiet", action="store_true", help="Только итоговая строка, без прогресса.")
    args = parser.parse_args()
    raise SystemExit(
        run(
            max_variants_per_case=args.max_variants_per_case,
            seed=args.seed,
            include_hard_benchmark=not args.without_hard_benchmark,
            hard_benchmark_variants_per_case=args.hard_benchmark_variants_per_case,
            quiet=args.quiet,
        )
    )
