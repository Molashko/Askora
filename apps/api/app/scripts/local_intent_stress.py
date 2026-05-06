from __future__ import annotations

import argparse
import json
import random

from app.db.session import SessionLocal
from app.repositories.users import UserRepository
from app.schemas.query import QueryRequest
from app.services.query_service import QueryService


BASE_QUESTIONS = [
    "Покажи выполненные заказы по дням за прошлую неделю",
    "Какой процент выполненных заказов в этом месяце",
    "Выручка и выполненные по дням за прошлую неделю",
    "Сравни долю успешных тендеров за текущую неделю и прошлую",
    "Средняя цена заказа по часам за вчера",
    "Сколько отмен было в прошлом месяце",
    "Ну чё у нас по деньгам за вчера?",
    "Сколько у нас сорвалось по дням на прошлой неделе?",
    "Довезли сколько по дням за эту неделю?",
    "На сколько процентов просела выручка в марте к февралю?",
    "Покажи среднюю скорость по дням за прошлую неделю",
    "Покажи выручку в выходные за прошлую неделю",
    "Покажи выручку в будни за прошлую неделю",
    "Покажи выручку по городам за прошлую неделю",
    "Покажи конверсию в выполненный заказ по дням за текущую неделю",
    "Покажи среднее время до принятия тендера по дням за прошлую неделю",
    "Покажи отмены по источникам за текущий месяц",
    "Сравни выручку за 16 апреля и 19 апреля",
    "Выручка за 16 марта и 19 марта по часам",
    "Подскажи обоот за 16 марта и 19 марта по часам",
    "Почему продажи упали в выходные? Покажи график",
    "Падение",
    "Обнови статусы заказов за вчера",
]

PREFIXES = ["", "пожалуйста ", "можешь ", "хочу понять ", "подскажи "]
SUFFIXES = ["", " пожалуйста", " плиз", ", если можно"]
REPLACEMENTS = [
    ("покажи", "выведи"),
    ("сравни", "сделай сравнение"),
    ("выручка", "оборот"),
    ("выполненные", "завершенные"),
    ("прошлую неделю", "предыдущую неделю"),
    ("в этом месяце", "за текущий месяц"),
]
TYPO_REPLACEMENTS = [
    ("выручка", "виручка"),
    ("оборот", "обоот"),
    ("заказы", "закзы"),
    ("отмен", "атмен"),
    ("сравни", "сровни"),
    ("тендер", "тендир"),
    ("средняя", "сридняя"),
    ("месяц", "месец"),
    ("недел", "нидел"),
    ("вчера", "вчра"),
]


def mutate(question: str) -> str:
    text = question.lower()
    for old, new in REPLACEMENTS:
        if old in text and random.random() < 0.5:
            text = text.replace(old, new, 1)
    if random.random() < 0.3 and len(text) > 8:
        index = random.randint(2, len(text) - 3)
        text = text[:index] + text[index + 1 :]
    return f"{random.choice(PREFIXES)}{text}{random.choice(SUFFIXES)}".strip()


def mutate_typo_heavy(question: str) -> str:
    text = mutate(question)
    lowered = text.lower()
    for old, new in TYPO_REPLACEMENTS:
        if old in lowered and random.random() < 0.7:
            text = text.replace(old, new, 1)
            lowered = text.lower()

    words = text.split()
    if words:
        for _ in range(random.randint(1, 2)):
            idx = random.randrange(len(words))
            token = words[idx]
            if len(token) < 4:
                continue
            mode = random.choice(["drop", "swap", "repeat"])
            if mode == "drop" and len(token) > 4:
                pos = random.randint(1, len(token) - 2)
                token = token[:pos] + token[pos + 1 :]
            elif mode == "swap" and len(token) > 4:
                pos = random.randint(1, len(token) - 3)
                chars = list(token)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                token = "".join(chars)
            else:
                pos = random.randint(1, len(token) - 2)
                token = token[:pos] + token[pos] + token[pos:]
            words[idx] = token
    return " ".join(words).strip()


def _normalize(question: str) -> str:
    return " ".join(question.lower().replace("ё", "е").split())


def _generate_unique_questions(target_count: int, *, typo_heavy: bool = False) -> list[str]:
    unique_questions: list[str] = []
    seen: set[str] = set()

    for base in BASE_QUESTIONS:
        normalized = _normalize(base)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_questions.append(base)
        if len(unique_questions) >= target_count:
            return unique_questions

    attempts = 0
    max_attempts = max(target_count * 50, 3000)
    while len(unique_questions) < target_count and attempts < max_attempts:
        attempts += 1
        question = mutate_typo_heavy(random.choice(BASE_QUESTIONS)) if typo_heavy else mutate(random.choice(BASE_QUESTIONS))
        normalized = _normalize(question)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_questions.append(question)
    return unique_questions


def run(*, target_count: int = 100, json_out: str | None = None, typo_heavy: bool = False) -> int:
    random.seed(42)
    db = SessionLocal()
    try:
        user = UserRepository(db).get_by_email("business@demo.local")
        if not user:
            raise RuntimeError("Не найден demo-пользователь business@demo.local")

        service = QueryService(db)
        questions = _generate_unique_questions(target_count, typo_heavy=typo_heavy)
        total = 0
        failures: list[str] = []
        source_distribution: dict[str, int] = {}
        status_distribution: dict[str, int] = {}

        for question in questions:
            total += 1
            result = service.run(QueryRequest(question=question), user)
            extraction = (result.processing_trace or {}).get("extraction", {})
            source = extraction.get("effective_source", "unknown")
            source_distribution[source] = source_distribution.get(source, 0) + 1
            status_distribution[result.status] = status_distribution.get(result.status, 0) + 1
            if result.status not in {"executed", "needs_clarification"}:
                failures.append(f"{question} -> {result.status}")

        print(f"Проверено уникальных стресс-кейсов: {total}")
        print(f"Распределение по источникам: {source_distribution}")
        print(f"Распределение по статусам: {status_distribution}")
        print(f"Недопустимых статусов: {len(failures)}")

        report = {
            "target_count": target_count,
            "typo_heavy": typo_heavy,
            "total": total,
            "source_distribution": source_distribution,
            "status_distribution": status_distribution,
            "failures": failures,
            "questions": questions,
        }
        if json_out:
            with open(json_out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
            print(f"JSON-отчёт сохранён: {json_out}")

        if failures:
            for item in failures[:10]:
                print(f"- {item}")
            return 1
        if total < target_count:
            print(f"Не удалось собрать целевые {target_count} уникальных вопросов, собрано только {total}.")
            return 1
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress run for unique NL intent queries.")
    parser.add_argument("--target-count", type=int, default=100, help="Сколько уникальных запросов прогнать.")
    parser.add_argument("--json-out", dest="json_out", help="Путь до JSON-отчёта по прогону.")
    parser.add_argument(
        "--typo-heavy",
        action="store_true",
        help="Генерировать больше опечаток и орфографического шума.",
    )
    args = parser.parse_args()
    raise SystemExit(run(target_count=args.target_count, json_out=args.json_out, typo_heavy=args.typo_heavy))
