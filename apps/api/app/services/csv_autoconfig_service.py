from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from io import StringIO
from pathlib import Path
import re
from typing import Any

import psycopg
from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ai.gemini_llm import gemini_llm
from app.core.config import settings
from app.data_sources.registry import RuntimeDataSource, data_source_registry
from app.models.data_source import DataSource
from app.semantic_layer.loader import semantic_loader


UPLOAD_SCHEMA = "analytics_uploads"
NUMERIC_RE_SQL = r"^[[:space:]]*-?[0-9]+(?:[,.][0-9]+)?[[:space:]]*$"
ISO_DATE_RE_SQL = r"^[[:space:]]*[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[ T][0-9]{2}:[0-9]{2}:[0-9]{2})?[[:space:]]*$"
RU_DATE_RE_SQL = r"^[[:space:]]*[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}(?:[[:space:]]+[0-9]{2}:[0-9]{2}:[0-9]{2})?[[:space:]]*$"


@dataclass
class ColumnProfile:
    name: str
    original_name: str
    inferred_type: str
    non_null_ratio: float
    unique_ratio: float
    examples: list[str]
    numeric_column: str | None = None
    datetime_column: str | None = None


class CsvAutoConfigService:
    def analyze_and_build_file(
        self,
        *,
        csv_path: Path,
        source_key: str | None,
        table_name: str | None,
        delimiter: str = "auto",
        apply: bool = False,
        auto_mode: bool = True,
        db: Session,
        filename: str | None = None,
        display_name: str | None = None,
        activate: bool = True,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        delimiter = self._normalize_delimiter(delimiter)
        encoding = self._detect_file_encoding(csv_path)
        sample_text = self._read_text_sample(csv_path, encoding=encoding)
        resolved_delimiter = self._resolve_delimiter(sample_text, delimiter)
        fieldnames, sample_rows_raw = self._read_csv_sample_rows(
            csv_path,
            encoding=encoding,
            delimiter=resolved_delimiter,
            max_rows=5000,
        )
        if not fieldnames:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV не содержит заголовок колонок")

        column_pairs = self._sanitize_columns(fieldnames)
        if not column_pairs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Не удалось получить колонки CSV")

        sample_rows = [
            {sanitized: (row.get(original) or "").strip() for original, sanitized in column_pairs}
            for row in sample_rows_raw
        ]
        profiles = self._profile_columns(column_pairs, sample_rows)
        self._assign_derived_columns(profiles)
        llm_enrichment = self._build_llm_enrichment(profiles, sample_rows, use_llm=use_llm)
        resolved_source_key, resolved_table_name, resolution_meta = self._resolve_target(
            requested_source_key=source_key,
            requested_table_name=table_name,
            auto_mode=auto_mode,
            filename=filename,
            db=db,
        )
        catalog = self._build_catalog(
            source_key=resolved_source_key,
            table_name=resolved_table_name,
            profiles=profiles,
            enrichment=llm_enrichment.get("columns", {}),
        )
        query_guidance = self._build_query_guidance(catalog, profiles, llm_enrichment)
        preview = self._build_preview(profiles, catalog)
        data_source_payload: dict[str, Any] | None = None
        row_count = len(sample_rows)

        if apply:
            row_count = self._import_csv_table_from_file(
                csv_path=csv_path,
                encoding=encoding,
                delimiter=resolved_delimiter,
                column_pairs=column_pairs,
                profiles=profiles,
                qualified_table=resolved_table_name,
                db=db,
            )
            data_source = self._persist_uploaded_source(
                db=db,
                key=resolved_source_key,
                name=(display_name or filename or resolved_source_key).strip(),
                table_name=resolved_table_name,
                catalog=catalog,
                profiles=profiles,
                row_count=row_count,
                filename=filename,
                delimiter=resolved_delimiter,
                activate=activate,
                llm_enrichment=llm_enrichment,
                query_guidance=query_guidance,
            )
            data_source_payload = self._data_source_summary(data_source)
            semantic_loader.invalidate()
            data_source_registry.invalidate()

        return {
            "applied": apply,
            "catalog_preview": preview,
            "catalog": catalog if not apply else None,
            "auto_resolution": {
                **resolution_meta,
                "validated": True if apply else resolution_meta.get("validated", False),
                "validation_message": (
                    f"CSV импортирован в {resolved_table_name}, строк: {row_count}."
                    if apply
                    else resolution_meta.get("validation_message")
                ),
            },
            "used_delimiter": resolved_delimiter,
            "data_source": data_source_payload,
        }

    def analyze_and_build(
        self,
        *,
        csv_bytes: bytes,
        source_key: str | None,
        table_name: str | None,
        delimiter: str = "auto",
        apply: bool = False,
        auto_mode: bool = True,
        db: Session,
        filename: str | None = None,
        display_name: str | None = None,
        activate: bool = True,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        delimiter = self._normalize_delimiter(delimiter)
        content = self._decode_csv(csv_bytes)
        resolved_delimiter = self._resolve_delimiter(content, delimiter)
        reader = csv.DictReader(StringIO(content), delimiter=resolved_delimiter)
        if not reader.fieldnames:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV не содержит заголовок колонок")

        column_pairs = self._sanitize_columns(reader.fieldnames)
        if not column_pairs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Не удалось получить колонки CSV")

        sample_rows: list[dict[str, str]] = []
        for index, row in enumerate(reader):
            if index >= 5000:
                break
            sample_rows.append({sanitized: (row.get(original) or "").strip() for original, sanitized in column_pairs})

        profiles = self._profile_columns(column_pairs, sample_rows)
        self._assign_derived_columns(profiles)
        llm_enrichment = self._build_llm_enrichment(profiles, sample_rows, use_llm=use_llm)
        resolved_source_key, resolved_table_name, resolution_meta = self._resolve_target(
            requested_source_key=source_key,
            requested_table_name=table_name,
            auto_mode=auto_mode,
            filename=filename,
            db=db,
        )
        catalog = self._build_catalog(
            source_key=resolved_source_key,
            table_name=resolved_table_name,
            profiles=profiles,
            enrichment=llm_enrichment.get("columns", {}),
        )
        query_guidance = self._build_query_guidance(catalog, profiles, llm_enrichment)
        preview = self._build_preview(profiles, catalog)
        data_source_payload: dict[str, Any] | None = None
        row_count = len(sample_rows)

        if apply:
            row_count = self._import_csv_table(
                content=content,
                delimiter=resolved_delimiter,
                column_pairs=column_pairs,
                profiles=profiles,
                qualified_table=resolved_table_name,
                db=db,
            )
            data_source = self._persist_uploaded_source(
                db=db,
                key=resolved_source_key,
                name=(display_name or filename or resolved_source_key).strip(),
                table_name=resolved_table_name,
                catalog=catalog,
                profiles=profiles,
                row_count=row_count,
                filename=filename,
                delimiter=resolved_delimiter,
                activate=activate,
                llm_enrichment=llm_enrichment,
                query_guidance=query_guidance,
            )
            data_source_payload = self._data_source_summary(data_source)
            semantic_loader.invalidate()
            data_source_registry.invalidate()

        return {
            "applied": apply,
            "catalog_preview": preview,
            "catalog": catalog if not apply else None,
            "auto_resolution": {
                **resolution_meta,
                "validated": True if apply else resolution_meta.get("validated", False),
                "validation_message": (
                    f"CSV импортирован в {resolved_table_name}, строк: {row_count}."
                    if apply
                    else resolution_meta.get("validation_message")
                ),
            },
            "used_delimiter": resolved_delimiter,
            "data_source": data_source_payload,
        }

    def activate_uploaded_source(self, *, db: Session, source_id: str) -> DataSource:
        source = db.query(DataSource).filter(DataSource.id == source_id).one_or_none()
        if source is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Датасет не найден")
        if not self._source_has_catalog(source) and source.key != settings.default_data_source_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="У выбранного источника нет сохранённого semantic catalog для активации.",
            )
        db.query(DataSource).filter(DataSource.is_default.is_(True)).update({"is_default": False})
        source.is_default = True
        source.is_active = True
        db.add(source)
        db.commit()
        db.refresh(source)
        semantic_loader.invalidate()
        data_source_registry.invalidate()
        return source

    def optimize_uploaded_source(self, *, db: Session, source_id: str) -> DataSource:
        source = db.query(DataSource).filter(DataSource.id == source_id).one_or_none()
        if source is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Датасет не найден")
        capabilities = dict(source.capabilities_json or {})
        if capabilities.get("kind") != "uploaded_csv":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Оптимизация доступна только для CSV-датасетов")

        table_name = str(capabilities.get("table_name") or "").strip()
        if "." not in table_name:
            raw_catalog = capabilities.get("semantic_catalog") if isinstance(capabilities.get("semantic_catalog"), dict) else {}
            dataset = (raw_catalog.get("datasets") or {}).get(source.key) if isinstance(raw_catalog, dict) else None
            table_name = str((dataset or {}).get("table") or "").strip()
        if "." not in table_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="У датасета нет корректного имени таблицы")

        profiles = self._profiles_from_capabilities(capabilities)
        if not profiles:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="У датасета нет профиля колонок")

        schema_name, table_leaf = table_name.split(".", 1)
        self._assign_derived_columns(profiles)
        self._alter_derived_columns(db, schema_name, table_leaf, profiles)
        quoted_table = f"{self._quote_ident(schema_name)}.{self._quote_ident(table_leaf)}"
        self._populate_derived_columns(db, quoted_table, profiles)
        self._create_optimization_indexes(db, schema_name, table_leaf, profiles)
        db.execute(text(f"ANALYZE {quoted_table}"))

        enrichment = self._enrichment_from_catalog(capabilities.get("semantic_catalog"), profiles)
        catalog = self._build_catalog(
            source_key=source.key,
            table_name=table_name,
            profiles=profiles,
            enrichment=enrichment,
        )
        query_guidance = self._build_query_guidance(
            catalog,
            profiles,
            {"used": bool((capabilities.get("llm_enrichment") or {}).get("used")), "query_guidance": capabilities.get("query_guidance")},
        )
        capabilities.update(
            {
                "semantic_catalog": catalog,
                "query_guidance": query_guidance,
                "columns": [self._profile_payload(profile) for profile in profiles],
                "performance": {
                    "typed_columns": True,
                    "indexed": True,
                    "optimized_at": datetime.now(UTC).isoformat(),
                },
            }
        )
        source.capabilities_json = capabilities
        db.add(source)
        db.commit()
        db.refresh(source)
        semantic_loader.invalidate()
        data_source_registry.invalidate()
        return source

    def delete_uploaded_source(self, *, db: Session, source_id: str) -> str:
        source = db.query(DataSource).filter(DataSource.id == source_id).one_or_none()
        if source is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

        capabilities = dict(source.capabilities_json or {})
        if capabilities.get("kind") != "uploaded_csv":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only uploaded CSV datasets can be deleted here.",
            )

        table_name = self._table_name_from_source(source)
        was_default = bool(source.is_default)
        deleted_name = source.name or source.key

        if table_name:
            schema_name, table_leaf = table_name.split(".", 1)
            if schema_name != UPLOAD_SCHEMA:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Dataset table is outside upload schema; automatic delete is blocked.",
                )
            quoted_table = f"{self._quote_ident(schema_name)}.{self._quote_ident(table_leaf)}"
            db.execute(text(f"DROP TABLE IF EXISTS {quoted_table} CASCADE"))

        db.delete(source)
        db.flush()

        if was_default:
            fallback = (
                db.query(DataSource)
                .filter(DataSource.key == settings.default_data_source_key, DataSource.is_active.is_(True))
                .one_or_none()
            )
            if fallback is None:
                fallback = (
                    db.query(DataSource)
                    .filter(DataSource.is_active.is_(True))
                    .order_by(DataSource.created_at.asc())
                    .first()
                )
            if fallback is not None:
                db.query(DataSource).filter(DataSource.is_default.is_(True)).update({"is_default": False})
                fallback.is_default = True
                db.add(fallback)

        db.commit()
        semantic_loader.invalidate()
        data_source_registry.invalidate()
        return deleted_name

    def _normalize_delimiter(self, delimiter: str) -> str:
        normalized = delimiter.strip().lower()
        if normalized in {"tab", "\\t"}:
            return "\t"
        if normalized in {"comma"}:
            return ","
        if normalized in {"semicolon"}:
            return ";"
        if normalized in {"pipe"}:
            return "|"
        if delimiter != "auto" and len(delimiter) != 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Delimiter должен быть одним символом")
        return delimiter

    def _decode_csv(self, payload: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось декодировать CSV. Поддерживаются UTF-8/UTF-8-BOM/CP1251.",
        )

    def _detect_file_encoding(self, csv_path: Path) -> str:
        with csv_path.open("rb") as file:
            sample = file.read(1024 * 1024)
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                sample.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось декодировать CSV. Поддерживаются UTF-8/UTF-8-BOM/CP1251.",
        )

    def _read_text_sample(self, csv_path: Path, *, encoding: str, max_bytes: int = 1024 * 1024) -> str:
        with csv_path.open("rb") as file:
            sample = file.read(max_bytes)
        return sample.decode(encoding, errors="ignore")

    def _read_csv_sample_rows(
        self,
        csv_path: Path,
        *,
        encoding: str,
        delimiter: str,
        max_rows: int,
    ) -> tuple[list[str], list[dict[str, str]]]:
        with csv_path.open("r", encoding=encoding, newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            fieldnames = list(reader.fieldnames or [])
            sample_rows: list[dict[str, str]] = []
            for index, row in enumerate(reader):
                if index >= max_rows:
                    break
                sample_rows.append({str(key): (value or "") for key, value in row.items() if key is not None})
        return fieldnames, sample_rows

    def _profile_columns(self, column_pairs: list[tuple[str, str]], rows: list[dict[str, str]]) -> list[ColumnProfile]:
        total_rows = max(1, len(rows))
        profiles: list[ColumnProfile] = []
        for original_name, column in column_pairs:
            values = [row.get(column, "").strip() for row in rows]
            non_empty = [value for value in values if value]
            unique_values = len(set(non_empty))
            inferred_type = self._infer_type(column, non_empty)
            examples = list(dict.fromkeys(non_empty[:8]))
            profiles.append(
                ColumnProfile(
                    name=column,
                    original_name=original_name,
                    inferred_type=inferred_type,
                    non_null_ratio=round(len(non_empty) / total_rows, 4),
                    unique_ratio=round((unique_values / max(1, len(non_empty))) if non_empty else 0.0, 4),
                    examples=examples,
                )
            )
        return profiles

    def _assign_derived_columns(self, profiles: list[ColumnProfile]) -> None:
        used = {profile.name for profile in profiles}
        used.update(profile.numeric_column for profile in profiles if profile.numeric_column)
        used.update(profile.datetime_column for profile in profiles if profile.datetime_column)
        for profile in profiles:
            if profile.inferred_type in {"int", "float"} and not profile.numeric_column:
                profile.numeric_column = self._unique_derived_column(profile.name, "num", used)
            elif profile.inferred_type == "datetime" and not profile.datetime_column:
                profile.datetime_column = self._unique_derived_column(profile.name, "ts", used)

    def _unique_derived_column(self, base_name: str, suffix: str, used: set[str]) -> str:
        base = f"{base_name}__{suffix}"
        candidate = base
        index = 2
        while candidate in used:
            candidate = f"{base}_{index}"
            index += 1
        used.add(candidate)
        return candidate

    def _infer_type(self, column_name: str, values: list[str]) -> str:
        normalized = column_name.lower()
        if not values:
            if any(token in normalized for token in ("date", "timestamp", "created", "updated", "dt")):
                return "datetime"
            return "text"

        checked = values[:300]
        datetime_hits = 0
        int_hits = 0
        float_hits = 0
        for raw in checked:
            if self._looks_like_datetime(raw):
                datetime_hits += 1
                continue
            if re.fullmatch(r"-?\d+", raw.strip()):
                int_hits += 1
                continue
            if re.fullmatch(r"-?\d+(?:[.,]\d+)?", raw.strip()):
                float_hits += 1
                continue

        threshold = max(1, int(len(checked) * 0.7))
        if datetime_hits >= threshold:
            return "datetime"
        if int_hits >= threshold:
            return "int"
        if float_hits >= threshold:
            return "float"
        if any(token in normalized for token in ("date", "timestamp", "created", "updated", "dt")):
            return "datetime"
        return "text"

    def _looks_like_datetime(self, raw: str) -> bool:
        cleaned = raw.strip().replace("T", " ")
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y", "%d.%m.%Y %H:%M:%S"):
            try:
                datetime.strptime(cleaned, fmt)
                return True
            except ValueError:
                continue
        return False

    def _build_catalog(
        self,
        *,
        source_key: str,
        table_name: str,
        profiles: list[ColumnProfile],
        enrichment: dict[str, Any],
    ) -> dict[str, Any]:
        alias = "ot"
        dataset_key = source_key
        time_column = self._pick_time_column(profiles)
        default_time_field = self._timestamp_expr_for_profile(alias, time_column) if time_column else "CURRENT_DATE::timestamp"

        metrics: dict[str, Any] = {
            "rows_count": {
                "key": "rows_count",
                "label": "Количество строк",
                "description": "Общее количество строк в датасете",
                "sql": "COUNT(*)",
                "synonyms": ["количество", "сколько", "число строк", "count", "rows", "записи"],
                "allowed_roles": ["admin", "analyst", "business_user"],
            }
        }
        dimensions: dict[str, Any] = {}
        filters: dict[str, Any] = {}
        business_terms: dict[str, Any] = {
            "количество": {"entity_type": "metric", "target_key": "rows_count"},
            "сколько строк": {"entity_type": "metric", "target_key": "rows_count"},
            "число строк": {"entity_type": "metric", "target_key": "rows_count"},
        }

        if time_column:
            time_expr = self._timestamp_expr_for_profile(alias, time_column)
            human = self._column_label(time_column, enrichment)
            dimensions["order_date"] = {
                "key": "order_date",
                "label": f"{human} (день)",
                "sql": f"DATE({time_expr})",
                "synonyms": ["по дням", "по датам", f"по {human}", time_column.name, time_column.original_name],
                "kind": "time",
                "grain": "day",
                "allowed_roles": ["admin", "analyst", "business_user"],
            }
            dimensions["order_week"] = {
                "key": "order_week",
                "label": f"{human} (неделя)",
                "sql": f"DATE_TRUNC('week', {time_expr})::date",
                "synonyms": ["по неделям", "понедельно", f"week {time_column.name}"],
                "kind": "time",
                "grain": "week",
                "allowed_roles": ["admin", "analyst", "business_user"],
            }
            dimensions["order_month"] = {
                "key": "order_month",
                "label": f"{human} (месяц)",
                "sql": f"DATE_TRUNC('month', {time_expr})::date",
                "synonyms": ["по месяцам", "помесячно", f"month {time_column.name}"],
                "kind": "time",
                "grain": "month",
                "allowed_roles": ["admin", "analyst", "business_user"],
            }

        for profile in profiles:
            human = self._column_label(profile, enrichment)
            col_expr = self._col(alias, profile.name)
            dim_key = f"dim_{profile.name}"
            base_synonyms = self._column_synonyms(profile, enrichment)

            if profile.inferred_type in {"int", "float"}:
                numeric_expr = self._numeric_expr_for_profile(alias, profile)
                if self._should_create_numeric_metrics(profile):
                    metric_sum_key = f"sum_{profile.name}"
                    metric_avg_key = f"avg_{profile.name}"
                    sum_synonyms, avg_synonyms = self._metric_synonyms(profile, human, enrichment)
                    metrics[metric_sum_key] = {
                        "key": metric_sum_key,
                        "label": f"Сумма {human}",
                        "description": f"Сумма поля {profile.original_name}",
                        "sql": f"COALESCE(ROUND(SUM({numeric_expr})::numeric, 2), 0)",
                        "synonyms": sum_synonyms,
                        "allowed_roles": ["admin", "analyst", "business_user"],
                    }
                    metrics[metric_avg_key] = {
                        "key": metric_avg_key,
                        "label": f"Среднее {human}",
                        "description": f"Среднее значение поля {profile.original_name}",
                        "sql": f"ROUND(AVG({numeric_expr})::numeric, 2)",
                        "synonyms": avg_synonyms,
                        "allowed_roles": ["admin", "analyst", "business_user"],
                    }
                    business_terms[f"сумма {human}"] = {"entity_type": "metric", "target_key": metric_sum_key}
                    business_terms[f"среднее {human}"] = {"entity_type": "metric", "target_key": metric_avg_key}
                    for synonym in sum_synonyms[:8]:
                        business_terms.setdefault(synonym, {"entity_type": "metric", "target_key": metric_sum_key})
                    for synonym in avg_synonyms[:8]:
                        business_terms.setdefault(synonym, {"entity_type": "metric", "target_key": metric_avg_key})

                filters[profile.name] = {
                    "key": profile.name,
                    "label": human,
                    "field": numeric_expr,
                    "operators": ["gt", "gte", "lt", "lte", "eq"],
                    "synonyms": base_synonyms,
                }

            if profile.inferred_type == "datetime":
                filters[profile.name] = {
                    "key": profile.name,
                    "label": human,
                    "field": self._timestamp_expr_for_profile(alias, profile),
                    "operators": ["eq", "in", "gte", "lte"],
                    "synonyms": base_synonyms,
                }
                continue

            if profile.inferred_type == "text" or self._looks_like_dimension_name(profile.name):
                dimensions[dim_key] = {
                    "key": dim_key,
                    "label": human,
                    "sql": f"COALESCE({col_expr}, 'unknown')",
                    "synonyms": [f"по {human}", profile.name, profile.original_name, *base_synonyms],
                    "kind": "category",
                    "allowed_roles": ["admin", "analyst", "business_user"],
                }
                filters[profile.name] = {
                    "key": profile.name,
                    "label": human,
                    "field": col_expr,
                    "operators": ["eq", "in"],
                    "synonyms": base_synonyms,
                }

        return {
            "version": 4,
            "base_dataset": dataset_key,
            "datasets": {
                dataset_key: {
                    "table": table_name,
                    "alias": alias,
                    "default_time_field": default_time_field,
                    "source_key": source_key,
                    "joins": [],
                }
            },
            "metrics": metrics,
            "dimensions": dimensions,
            "filters": filters,
            "joins": {},
            "business_terms": business_terms,
            "time_mappings": self._default_time_mappings(),
        }

    def _build_llm_enrichment(self, profiles: list[ColumnProfile], sample_rows: list[dict[str, str]], *, use_llm: bool) -> dict[str, Any]:
        if not use_llm or not gemini_llm.remote_configured:
            return {"enabled": False, "used": False, "columns": {}, "trace": {"status": "disabled"}}

        compact_columns = [
            {
                "name": profile.name,
                "original_name": profile.original_name,
                "inferred_type": profile.inferred_type,
                "examples": profile.examples[:5],
            }
            for profile in profiles[:80]
        ]
        system_prompt = (
            "Ты помогаешь построить semantic layer для CSV-датасета. "
            "Верни только JSON без markdown. Не придумывай колонок, используй только имена из входа. "
            "Для каждой понятной колонки дай русскую label, 3-8 русских/английских синонимов и короткое описание. "
            "Ответ: {\"columns\": {\"column_name\": {\"label\": \"...\", \"synonyms\": [\"...\"], \"description\": \"...\"}}}."
        )
        system_prompt += (
            " Also return query_guidance with Russian natural-language helpers for this dataset: "
            "{\"quick_questions\": [4-8 example questions], \"quick_fragments\": [8-14 short query chips], "
            "\"composing_hints\": [3-4 short hints]}. Keep examples grounded in the given columns only."
        )
        user_prompt = json.dumps({"columns": compact_columns, "sample_rows": sample_rows[:20]}, ensure_ascii=False, default=str)
        payload, trace = gemini_llm.extract_json_with_trace(system_prompt, user_prompt, max_output_tokens=8192)
        columns = payload.get("columns", {}) if isinstance(payload, dict) else {}
        query_guidance = payload.get("query_guidance", {}) if isinstance(payload, dict) else {}
        return {
            "enabled": True,
            "used": bool(columns) or bool(query_guidance),
            "columns": columns if isinstance(columns, dict) else {},
            "query_guidance": query_guidance if isinstance(query_guidance, dict) else {},
            "trace": trace,
        }

    def _build_query_guidance(
        self,
        catalog: dict[str, Any],
        profiles: list[ColumnProfile],
        llm_enrichment: dict[str, Any],
    ) -> dict[str, Any]:
        llm_guidance = llm_enrichment.get("query_guidance")
        if isinstance(llm_guidance, dict):
            quick_questions = self._clean_guidance_list(llm_guidance.get("quick_questions"))
            quick_fragments = self._clean_guidance_list(llm_guidance.get("quick_fragments"))
            composing_hints = self._clean_guidance_list(llm_guidance.get("composing_hints"))
            if quick_questions or quick_fragments or composing_hints:
                return {
                    "llm_used": bool(llm_enrichment.get("used")),
                    "quick_questions": quick_questions,
                    "quick_fragments": quick_fragments,
                    "composing_hints": composing_hints,
                }

        metrics = [str(item.get("label") or key) for key, item in catalog.get("metrics", {}).items()]
        dimensions = [str(item.get("label") or key) for key, item in catalog.get("dimensions", {}).items()]
        columns = [profile.original_name for profile in profiles[:8]]
        primary_metric = self._plain_metric_label(metrics[0]) if metrics else "количество строк"
        second_metric = self._plain_metric_label(metrics[1]) if len(metrics) > 1 else primary_metric
        category_dimension = next(
            (
                self._plain_dimension_label(item)
                for item in dimensions
                if "день" not in item.lower() and "недел" not in item.lower() and "месяц" not in item.lower()
            ),
            "",
        )
        quick_questions = [
            f"Покажи {primary_metric} по дням за прошлую неделю",
            f"Покажи {second_metric} за текущий месяц",
        ]
        if category_dimension:
            quick_questions.append(f"Покажи {primary_metric} по {category_dimension} за прошлую неделю")

        quick_fragments = [
            *[self._plain_metric_label(item) for item in metrics[:6]],
            *[f"по {self._plain_dimension_label(item)}" for item in dimensions[:5]],
            "за вчера",
            "за прошлую неделю",
            "за текущий месяц",
        ]
        composing_hints = [
            f"Что считать: {', '.join(self._plain_metric_label(item) for item in metrics[:5]) or primary_metric}.",
            f"Как разбить: {', '.join(self._plain_dimension_label(item) for item in dimensions[:5]) or 'по дням и категориям'}.",
            "За какой период: за вчера, за прошлую неделю, за текущий месяц, с даты по дату.",
        ]
        if columns:
            composing_hints.append(f"Колонки датасета: {', '.join(columns)}.")
        return {
            "llm_used": False,
            "quick_questions": self._dedupe(quick_questions),
            "quick_fragments": self._dedupe(quick_fragments),
            "composing_hints": self._dedupe(composing_hints),
        }

    def _clean_guidance_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _plain_metric_label(self, label: str) -> str:
        return (
            label.replace("РЎСѓРјРјР° ", "")
            .replace("РЎСЂРµРґРЅРµРµ ", "среднее ")
            .replace("Сумма ", "")
            .replace("Среднее ", "среднее ")
            .strip()
        )

    def _plain_dimension_label(self, label: str) -> str:
        return (
            label.replace(" (РґРµРЅСЊ)", "")
            .replace(" (РЅРµРґРµР»СЏ)", "")
            .replace(" (РјРµСЃСЏС†)", "")
            .replace(" (день)", "")
            .replace(" (неделя)", "")
            .replace(" (месяц)", "")
            .strip()
        )

    def _dedupe(self, items: list[str]) -> list[str]:
        result: list[str] = []
        for item in items:
            cleaned = item.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result

    def _pick_time_column(self, profiles: list[ColumnProfile]) -> ColumnProfile | None:
        for profile in profiles:
            if profile.inferred_type == "datetime":
                return profile
        return None

    def _build_preview(self, profiles: list[ColumnProfile], catalog: dict[str, Any]) -> dict[str, Any]:
        columns = [
            {
                "name": item.name,
                "inferred_type": item.inferred_type,
                "non_null_ratio": item.non_null_ratio,
                "unique_ratio": item.unique_ratio,
            }
            for item in profiles
        ]
        return {
            "columns": columns,
            "metrics_count": len(catalog["metrics"]),
            "dimensions_count": len(catalog["dimensions"]),
            "filters_count": len(catalog["filters"]),
            "base_dataset": catalog["base_dataset"],
        }

    def _resolve_target(
        self,
        *,
        requested_source_key: str | None,
        requested_table_name: str | None,
        auto_mode: bool,
        filename: str | None,
        db: Session,
    ) -> tuple[str, str, dict[str, Any]]:
        file_stem = Path(filename or "dataset").stem
        base_key = self._sanitize_identifier(requested_source_key or file_stem or "dataset")
        source = self._unique_source_key(db, f"csv_{base_key}")
        requested_table = (requested_table_name or "").strip()
        table_leaf = self._sanitize_identifier(requested_table.split(".")[-1] if requested_table else base_key)
        table = self._unique_table_name(db, UPLOAD_SCHEMA, table_leaf)
        qualified_table = f"{UPLOAD_SCHEMA}.{table}"
        strategy = "safe_auto" if auto_mode else "manual_name"
        notes = [
            "CSV будет импортирован в управляемую таблицу Postgres; исходный файл не исполняется как SQL.",
            "Семантический слой сохранится рядом с датасетом и может быть активирован из админки.",
        ]
        return source, qualified_table, {
            "strategy": strategy,
            "resolved_source_key": source,
            "resolved_table_name": qualified_table,
            "notes": notes,
            "validated": False,
            "validation_message": "Таблица будет создана при применении CSV.",
            "candidates": [
                {
                    "source_key": source,
                    "table_name": qualified_table,
                    "confidence": 0.98,
                    "reason": "Новая управляемая таблица для загруженного CSV.",
                }
            ],
        }

    def _validate_target_table(self, source: RuntimeDataSource | None, table_name: str, db: Session) -> tuple[bool, str | None]:
        if not source:
            return False, "Источник данных не найден в активном реестре."
        if source.dialect not in {"postgres", "postgresql"}:
            return True, "Проверка существования таблицы доступна только для Postgres, пропущено."

        check_sql = "SELECT to_regclass(:table_name)"
        try:
            if data_source_registry.is_primary_source(source):
                value = db.execute(text(check_sql), {"table_name": table_name}).scalar()
            else:
                engine = data_source_registry.get_engine(source)
                with engine.begin() as connection:
                    value = connection.execute(text(check_sql), {"table_name": table_name}).scalar()
            if value:
                return True, "Таблица найдена в выбранном источнике."
            return False, f"Таблица {table_name} не найдена в выбранном источнике."
        except Exception as exc:
            return False, f"Не удалось проверить таблицу автоматически: {exc}"

    def _import_csv_table(
        self,
        *,
        content: str,
        delimiter: str,
        column_pairs: list[tuple[str, str]],
        profiles: list[ColumnProfile],
        qualified_table: str,
        db: Session,
    ) -> int:
        schema_name, table_name = qualified_table.split(".", 1)
        quoted_table = f"{self._quote_ident(schema_name)}.{self._quote_ident(table_name)}"
        raw_defs = [f"{self._quote_ident(sanitized)} TEXT" for _, sanitized in column_pairs]
        derived_defs = self._derived_column_defs(profiles)
        column_defs = ", ".join([*raw_defs, *derived_defs])
        db.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self._quote_ident(schema_name)}"))
        db.execute(text(f"DROP TABLE IF EXISTS {quoted_table}"))
        db.execute(text(f"CREATE TABLE {quoted_table} (row_id BIGSERIAL PRIMARY KEY, {column_defs})"))
        db.commit()

        source_reader = csv.DictReader(StringIO(content), delimiter=delimiter)
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[sanitized for _, sanitized in column_pairs], lineterminator="\n")
        writer.writeheader()
        row_count = 0
        for row in source_reader:
            writer.writerow({sanitized: row.get(original) or "" for original, sanitized in column_pairs})
            row_count += 1
        output.seek(0)

        copy_columns = ", ".join(self._quote_ident(sanitized) for _, sanitized in column_pairs)
        copy_sql = f"COPY {quoted_table} ({copy_columns}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')"
        with psycopg.connect(self._psycopg_conn_string()) as connection:
            with connection.cursor() as cursor:
                with cursor.copy(copy_sql) as copy:
                    while chunk := output.read(1024 * 1024):
                        copy.write(chunk)
            connection.commit()
        self._populate_derived_columns(db, quoted_table, profiles)
        self._create_optimization_indexes(db, schema_name, table_name, profiles)
        db.execute(text(f"ANALYZE {quoted_table}"))
        db.commit()
        return row_count

    def _import_csv_table_from_file(
        self,
        *,
        csv_path: Path,
        encoding: str,
        delimiter: str,
        column_pairs: list[tuple[str, str]],
        profiles: list[ColumnProfile],
        qualified_table: str,
        db: Session,
    ) -> int:
        schema_name, table_name = qualified_table.split(".", 1)
        quoted_table = f"{self._quote_ident(schema_name)}.{self._quote_ident(table_name)}"
        raw_defs = [f"{self._quote_ident(sanitized)} TEXT" for _, sanitized in column_pairs]
        derived_defs = self._derived_column_defs(profiles)
        column_defs = ", ".join([*raw_defs, *derived_defs])
        db.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self._quote_ident(schema_name)}"))
        db.execute(text(f"DROP TABLE IF EXISTS {quoted_table}"))
        db.execute(text(f"CREATE TABLE {quoted_table} (row_id BIGSERIAL PRIMARY KEY, {column_defs})"))
        db.commit()

        copy_columns = ", ".join(self._quote_ident(sanitized) for _, sanitized in column_pairs)
        copy_sql = (
            f"COPY {quoted_table} ({copy_columns}) FROM STDIN "
            f"WITH (FORMAT CSV, HEADER TRUE, NULL '', DELIMITER {self._copy_delimiter_literal(delimiter)})"
        )
        with psycopg.connect(self._psycopg_conn_string()) as connection:
            with connection.cursor() as cursor:
                with cursor.copy(copy_sql) as copy:
                    with csv_path.open("r", encoding=encoding, newline="") as file:
                        while chunk := file.read(1024 * 1024):
                            copy.write(chunk)
            connection.commit()

        row_count = int(db.execute(text(f"SELECT COUNT(*) FROM {quoted_table}")).scalar() or 0)
        self._populate_derived_columns(db, quoted_table, profiles)
        self._create_optimization_indexes(db, schema_name, table_name, profiles)
        db.execute(text(f"ANALYZE {quoted_table}"))
        db.commit()
        return row_count

    def _copy_delimiter_literal(self, delimiter: str) -> str:
        if delimiter == "\t":
            return "E'\\t'"
        return "'" + delimiter.replace("'", "''") + "'"

    def _derived_column_defs(self, profiles: list[ColumnProfile]) -> list[str]:
        defs: list[str] = []
        for profile in profiles:
            if profile.numeric_column:
                defs.append(f"{self._quote_ident(profile.numeric_column)} NUMERIC")
            if profile.datetime_column:
                defs.append(f"{self._quote_ident(profile.datetime_column)} TIMESTAMP")
        return defs

    def _populate_derived_columns(self, db: Session, quoted_table: str, profiles: list[ColumnProfile]) -> None:
        set_parts: list[str] = []
        for profile in profiles:
            if profile.numeric_column:
                set_parts.append(
                    f"{self._quote_ident(profile.numeric_column)} = {self._numeric_cast_expr(self._quote_ident(profile.name))}"
                )
            if profile.datetime_column:
                set_parts.append(
                    f"{self._quote_ident(profile.datetime_column)} = {self._timestamp_cast_expr(self._quote_ident(profile.name))}"
                )
        if not set_parts:
            return
        db.execute(text(f"UPDATE {quoted_table} SET {', '.join(set_parts)}"))

    def _alter_derived_columns(self, db: Session, schema_name: str, table_name: str, profiles: list[ColumnProfile]) -> None:
        quoted_table = f"{self._quote_ident(schema_name)}.{self._quote_ident(table_name)}"
        for profile in profiles:
            if profile.numeric_column:
                db.execute(
                    text(
                        f"ALTER TABLE {quoted_table} "
                        f"ADD COLUMN IF NOT EXISTS {self._quote_ident(profile.numeric_column)} NUMERIC"
                    )
                )
            if profile.datetime_column:
                db.execute(
                    text(
                        f"ALTER TABLE {quoted_table} "
                        f"ADD COLUMN IF NOT EXISTS {self._quote_ident(profile.datetime_column)} TIMESTAMP"
                    )
                )

    def _create_optimization_indexes(
        self,
        db: Session,
        schema_name: str,
        table_name: str,
        profiles: list[ColumnProfile],
    ) -> None:
        quoted_table = f"{self._quote_ident(schema_name)}.{self._quote_ident(table_name)}"
        indexed_columns: list[tuple[str, str]] = []

        for profile in [item for item in profiles if item.datetime_column][:4]:
            indexed_columns.append((profile.datetime_column or "", "ts"))

        dimension_candidates = sorted(
            [
                profile
                for profile in profiles
                if self._looks_like_dimension_name(profile.name)
                or (profile.inferred_type == "text" and profile.unique_ratio <= 0.25)
            ],
            key=self._dimension_index_priority,
        )
        for profile in dimension_candidates[:12]:
            indexed_columns.append((profile.name, "dim"))

        numeric_candidates = [profile for profile in profiles if profile.numeric_column and self._should_create_numeric_metrics(profile)]
        for profile in numeric_candidates[:8]:
            indexed_columns.append((profile.numeric_column or "", "num"))

        seen: set[str] = set()
        for column, suffix in indexed_columns:
            if not column or column in seen:
                continue
            seen.add(column)
            index_name = self._index_name(table_name, column, suffix)
            db.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {self._quote_ident(index_name)} "
                    f"ON {quoted_table} ({self._quote_ident(column)})"
                )
            )

    def _dimension_index_priority(self, profile: ColumnProfile) -> tuple[int, float, str]:
        name = profile.name.lower()
        preferred = any(
            token in name
            for token in [
                "status",
                "city",
                "driver",
                "user",
                "client",
                "customer",
                "category",
                "type",
                "region",
                "country",
                "segment",
            ]
        )
        return (0 if preferred else 1, profile.unique_ratio, profile.name)

    def _index_name(self, table_name: str, column: str, suffix: str) -> str:
        raw = self._sanitize_identifier(f"idx_{table_name}_{column}_{suffix}")
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        base = raw[:54].strip("_") or "idx_upload"
        return f"{base}_{digest}"[:63]

    def _profiles_from_capabilities(self, capabilities: dict[str, Any]) -> list[ColumnProfile]:
        profiles: list[ColumnProfile] = []
        raw_columns = capabilities.get("columns")
        if not isinstance(raw_columns, list):
            return profiles
        for item in raw_columns:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            column_name = str(item.get("name"))
            inferred_type = str(item.get("inferred_type") or "text")
            if self._looks_like_numeric_duration(column_name):
                inferred_type = "int"
            profiles.append(
                ColumnProfile(
                    name=column_name,
                    original_name=str(item.get("original_name") or item.get("name")),
                    inferred_type=inferred_type,
                    non_null_ratio=float(item.get("non_null_ratio") or 0),
                    unique_ratio=float(item.get("unique_ratio") or 0),
                    examples=[str(value) for value in item.get("examples", [])[:8]] if isinstance(item.get("examples"), list) else [],
                    numeric_column=str(item.get("numeric_column")) if item.get("numeric_column") else None,
                    datetime_column=None if inferred_type != "datetime" else (str(item.get("datetime_column")) if item.get("datetime_column") else None),
                )
            )
        return profiles

    def _looks_like_numeric_duration(self, name: str) -> bool:
        normalized = name.lower()
        tokens = {token for token in re.split(r"[_\s]+", normalized) if token}
        return "seconds" in tokens or normalized.endswith("_sec") or "duration" in tokens

    def _profile_payload(self, profile: ColumnProfile) -> dict[str, Any]:
        return {
            "name": profile.name,
            "original_name": profile.original_name,
            "inferred_type": profile.inferred_type,
            "non_null_ratio": profile.non_null_ratio,
            "unique_ratio": profile.unique_ratio,
            "examples": profile.examples[:8],
            "numeric_column": profile.numeric_column,
            "datetime_column": profile.datetime_column,
        }

    def _enrichment_from_catalog(self, raw_catalog: Any, profiles: list[ColumnProfile]) -> dict[str, Any]:
        if not isinstance(raw_catalog, dict):
            return {}
        dimensions = raw_catalog.get("dimensions") if isinstance(raw_catalog.get("dimensions"), dict) else {}
        filters = raw_catalog.get("filters") if isinstance(raw_catalog.get("filters"), dict) else {}
        enrichment: dict[str, Any] = {}
        for profile in profiles:
            synonyms: list[str] = []
            label: str | None = None
            dim = dimensions.get(f"dim_{profile.name}")
            if not isinstance(dim, dict) and profile.inferred_type == "datetime":
                dim = dimensions.get("order_date")
            if isinstance(dim, dict):
                label = str(dim.get("label") or "").strip() or label
                if isinstance(dim.get("synonyms"), list):
                    synonyms.extend(
                        str(item)
                        for item in dim["synonyms"]
                        if not self._is_metricish_synonym(str(item)) and not self._is_grouping_synonym(str(item))
                    )
            filt = filters.get(profile.name)
            if isinstance(filt, dict):
                label = str(filt.get("label") or "").strip() or label
                if isinstance(filt.get("synonyms"), list):
                    synonyms.extend(
                        str(item)
                        for item in filt["synonyms"]
                        if not self._is_metricish_synonym(str(item)) and not self._is_grouping_synonym(str(item))
                    )
            if label or synonyms:
                enrichment[profile.name] = {"label": self._plain_dimension_label(label) if label else "", "synonyms": self._dedupe(synonyms)}
        return enrichment

    def _is_metricish_synonym(self, value: str) -> bool:
        normalized = value.strip().lower()
        return normalized.startswith(("sum ", "avg ", "average ", "сумма ", "среднее ", "средний ", "итого "))

    def _persist_uploaded_source(
        self,
        *,
        db: Session,
        key: str,
        name: str,
        table_name: str,
        catalog: dict[str, Any],
        profiles: list[ColumnProfile],
        row_count: int,
        filename: str | None,
        delimiter: str,
        activate: bool,
        llm_enrichment: dict[str, Any],
        query_guidance: dict[str, Any],
    ) -> DataSource:
        if activate:
            db.query(DataSource).filter(DataSource.is_default.is_(True)).update({"is_default": False})

        now = datetime.now(UTC).isoformat()
        source = DataSource(
            key=key,
            name=name or key,
            description=f"CSV-датасет {filename or key}, импортирован {now}",
            dialect="postgres",
            connection_url=settings.database_url,
            schema_name=UPLOAD_SCHEMA,
            is_active=True,
            is_default=activate,
            allowed_roles_json=["admin", "analyst", "business_user"],
            capabilities_json={
                "kind": "uploaded_csv",
                "table_name": table_name,
                "row_count": row_count,
                "uploaded_filename": filename,
                "uploaded_at": now,
                "used_delimiter": delimiter,
                "semantic_catalog": catalog,
                "columns": [self._profile_payload(profile) for profile in profiles],
                "llm_enrichment": {
                    "enabled": bool(llm_enrichment.get("enabled")),
                    "used": bool(llm_enrichment.get("used")),
                    "trace": llm_enrichment.get("trace", {}),
                },
                "query_guidance": query_guidance,
            },
        )
        db.add(source)
        db.commit()
        db.refresh(source)
        return source

    def _resolve_delimiter(self, content: str, delimiter: str) -> str:
        if delimiter != "auto":
            return delimiter
        sample = "\n".join(content.splitlines()[:8])
        try:
            sniffed = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return sniffed.delimiter
        except csv.Error:
            return ","

    def _sanitize_columns(self, fieldnames: list[str]) -> list[tuple[str, str]]:
        used: dict[str, int] = {}
        result: list[tuple[str, str]] = []
        for raw in fieldnames:
            if raw is None or not str(raw).strip():
                continue
            base = self._sanitize_identifier(str(raw))
            count = used.get(base, 0)
            used[base] = count + 1
            sanitized = base if count == 0 else f"{base}_{count + 1}"
            result.append((str(raw), sanitized))
        return result

    def _sanitize_identifier(self, raw: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", raw.strip())
        value = re.sub(r"_+", "_", value).strip("_").lower()
        if value and value[0].isdigit():
            value = f"c_{value}"
        return value or "field"

    def _humanize(self, value: str) -> str:
        return value.replace("_", " ").strip()

    def _column_label(self, profile: ColumnProfile, enrichment: dict[str, Any]) -> str:
        item = enrichment.get(profile.name)
        translated = self._russian_label_from_name(profile.name, profile.original_name)
        if isinstance(item, dict) and isinstance(item.get("label"), str) and item["label"].strip():
            label = item["label"].strip()
            if translated and not re.search(r"[А-Яа-яЁё]", label):
                return translated
            return label
        if translated:
            return translated
        return self._humanize(profile.original_name or profile.name)

    def _column_synonyms(self, profile: ColumnProfile, enrichment: dict[str, Any]) -> list[str]:
        values = [profile.name, profile.original_name, self._humanize(profile.name), self._humanize(profile.original_name)]
        translated = self._russian_label_from_name(profile.name, profile.original_name)
        if translated:
            values.append(translated)
            values.extend(self._label_synonym_variants(translated))
        item = enrichment.get(profile.name)
        if isinstance(item, dict) and isinstance(item.get("synonyms"), list):
            values.extend(str(s) for s in item["synonyms"] if str(s).strip())
        return self._dedupe(values)

    def _metric_synonyms(self, profile: ColumnProfile, human: str, enrichment: dict[str, Any]) -> tuple[list[str], list[str]]:
        base = [item for item in self._column_synonyms(profile, enrichment) if not self._is_grouping_synonym(item)]
        sum_synonyms = [f"сумма {human}", f"итого {human}", f"sum {profile.name}", *base]
        avg_synonyms = [f"среднее {human}", f"avg {profile.name}", f"average {profile.name}", f"средний {human}"]
        normalized = f"{profile.name} {profile.original_name}".lower()
        if any(token in normalized for token in ["revenue", "sales", "amount", "price", "cost", "total", "выруч", "продаж", "сумм", "цена", "стоим"]):
            sum_synonyms.extend(["выручка", "выручку", "продажи", "оборот", "доход", "сумма"])
        if any(token in normalized for token in ["count", "cnt", "qty", "quantity", "orders", "rides", "колич"]):
            sum_synonyms.extend(["количество", "сколько", "число"])
        if "order" in normalized or "заказ" in normalized:
            sum_synonyms.extend(["заказы", "количество заказов", "выполненные заказы"])
        if "ride" in normalized or "trip" in normalized or "поезд" in normalized:
            sum_synonyms.extend(["поездки", "количество поездок"])
        if "cancel" in normalized or "отмен" in normalized:
            sum_synonyms.extend(["отмены", "отмены клиентом", "количество отмен"])
        if "accept" in normalized or "принят" in normalized:
            sum_synonyms.extend(["принятые", "принятые заказы"])
        if "tender" in normalized or "тендер" in normalized:
            sum_synonyms.extend(["тендеры", "заказы с тендерами"])
        return self._dedupe(sum_synonyms), self._dedupe(avg_synonyms)

    def _label_synonym_variants(self, label: str) -> list[str]:
        normalized = label.strip().lower()
        variants: dict[str, list[str]] = {
            "город": ["города", "городам", "по городам", "по городу"],
            "водитель": ["водители", "водителям", "по водителям", "по водителю"],
            "пользователь": ["пользователи", "пользователям", "по пользователям"],
            "клиент": ["клиенты", "клиентам", "по клиентам"],
            "статус": ["статусы", "статусам", "по статусам"],
        }
        result = list(variants.get(normalized, []))
        if normalized.endswith(" заказы"):
            result.extend([normalized, normalized.replace(" заказы", " заказов")])
        return result

    def _is_grouping_synonym(self, value: str) -> bool:
        normalized = value.strip().lower()
        return normalized.startswith(("по ", "в разрезе", "разбивка", "group by")) or normalized in {
            "по дням",
            "по датам",
            "по неделям",
            "по месяцам",
        }

    def _looks_like_dimension_name(self, name: str) -> bool:
        normalized = name.lower()
        tokens = {token for token in re.split(r"[_\s]+", normalized) if token}
        if normalized.endswith("_id") or normalized == "id":
            return True
        if tokens & {"cancel", "cancelled", "accept", "accepted", "orders", "order", "rides", "ride", "count", "cnt", "amount", "sum", "seconds"}:
            return False
        return bool(
            tokens
            & {
                "status",
                "type",
                "category",
                "name",
                "city",
                "country",
                "region",
                "segment",
                "channel",
                "driver",
                "user",
                "client",
                "customer",
                "store",
                "branch",
                "product",
                "sku",
                "code",
            }
        )

    def _should_create_numeric_metrics(self, profile: ColumnProfile) -> bool:
        normalized = f"{profile.name} {profile.original_name}".lower()
        name = profile.name.lower()
        identifier_patterns = [
            r"(^|_)id($|_)",
            r"(^|_)uuid($|_)",
            r"(^|_)guid($|_)",
            r"(^|_)code($|_)",
            r"(^|_)phone($|_)",
            r"(^|_)zip($|_)",
            r"(^|_)postal($|_)",
            r"(^|_)lat(?:itude)?($|_)",
            r"(^|_)lon(?:gitude)?($|_)",
        ]
        if any(re.search(pattern, name) for pattern in identifier_patterns):
            return False
        if any(token in normalized for token in ["номер", "телефон", "индекс", "широта", "долгота"]):
            return False
        return True

    def _russian_label_from_name(self, name: str, original_name: str) -> str | None:
        normalized = self._sanitize_identifier(original_name or name)
        exact = {
            "city_id": "город",
            "driver_id": "водитель",
            "user_id": "пользователь",
            "client_id": "клиент",
            "customer_id": "клиент",
            "status": "статус",
            "status_order": "статус заказа",
            "orders": "заказы",
            "orders_cnt": "количество заказов",
            "orders_count": "количество заказов",
            "rides": "поездки",
            "rides_count": "количество поездок",
            "orders_cnt_accepted": "принятые заказы",
            "client_cancel_after_accept": "отмены клиентом после принятия",
            "orders_cnt_with_tenders": "заказы с тендерами",
        }
        if normalized in exact:
            return exact[normalized]

        token_map = {
            "city": "город",
            "driver": "водитель",
            "user": "пользователь",
            "client": "клиент",
            "customer": "клиент",
            "order": "заказ",
            "orders": "заказы",
            "cnt": "количество",
            "count": "количество",
            "accepted": "принятые",
            "accept": "принятие",
            "cancel": "отмены",
            "cancelled": "отмены",
            "tender": "тендер",
            "tenders": "тендеры",
            "ride": "поездка",
            "rides": "поездки",
            "price": "цена",
            "amount": "сумма",
            "revenue": "выручка",
            "sales": "продажи",
            "status": "статус",
            "date": "дата",
            "time": "время",
        }
        tokens = [token for token in re.split(r"[_\s]+", normalized) if token and token != "id"]
        translated = [token_map[token] for token in tokens if token in token_map]
        if translated:
            return " ".join(translated)
        return None

    def _numeric_expr(self, alias: str, column: str) -> str:
        return self._numeric_cast_expr(self._col(alias, column))

    def _timestamp_expr(self, alias: str, column: str) -> str:
        return self._timestamp_cast_expr(self._col(alias, column))

    def _numeric_expr_for_profile(self, alias: str, profile: ColumnProfile) -> str:
        if profile.numeric_column:
            return self._col(alias, profile.numeric_column)
        return self._numeric_expr(alias, profile.name)

    def _timestamp_expr_for_profile(self, alias: str, profile: ColumnProfile) -> str:
        if profile.datetime_column:
            return self._col(alias, profile.datetime_column)
        return self._timestamp_expr(alias, profile.name)

    def _numeric_cast_expr(self, field_expr: str) -> str:
        raw = f"TRIM({field_expr})"
        return f"CASE WHEN {raw} ~ '{NUMERIC_RE_SQL}' THEN REPLACE({raw}, ',', '.')::numeric ELSE NULL END"

    def _timestamp_cast_expr(self, field_expr: str) -> str:
        raw = f"TRIM({field_expr})"
        return (
            "CASE "
            f"WHEN {raw} ~ '{ISO_DATE_RE_SQL}' THEN REPLACE({raw}, 'T', ' ')::timestamp "
            f"WHEN {raw} ~ '^[[:space:]]*[0-9]{{1,2}}\\.[0-9]{{1,2}}\\.[0-9]{{4}}[[:space:]]*$' THEN to_date({raw}, 'DD.MM.YYYY')::timestamp "
            f"WHEN {raw} ~ '{RU_DATE_RE_SQL}' THEN to_timestamp({raw}, 'DD.MM.YYYY HH24:MI:SS') "
            "ELSE NULL END"
        )

    def _col(self, alias: str, column: str) -> str:
        return f"{alias}.{self._quote_ident(column)}"

    def _quote_ident(self, value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    def _unique_source_key(self, db: Session, base: str) -> str:
        candidate = base[:70].strip("_") or "csv_dataset"
        existing = {row.key for row in db.query(DataSource.key).all()}
        if candidate not in existing:
            return candidate
        for index in range(2, 1000):
            suffix = f"_{index}"
            next_candidate = f"{candidate[: 80 - len(suffix)]}{suffix}"
            if next_candidate not in existing:
                return next_candidate
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Не удалось подобрать уникальный ключ датасета")

    def _unique_table_name(self, db: Session, schema_name: str, base: str) -> str:
        stem = base[:50].strip("_") or "dataset"
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        for index in range(1, 1000):
            suffix = stamp if index == 1 else f"{stamp}_{index}"
            table_name = f"{stem}_{suffix}"[:63]
            exists = db.execute(text("SELECT to_regclass(:name)"), {"name": f"{schema_name}.{table_name}"}).scalar()
            if not exists:
                return table_name
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Не удалось подобрать уникальное имя таблицы")

    def _source_has_catalog(self, source: DataSource) -> bool:
        capabilities = source.capabilities_json or {}
        return isinstance(capabilities.get("semantic_catalog"), dict)

    def _table_name_from_source(self, source: DataSource) -> str | None:
        capabilities = source.capabilities_json or {}
        table_name = str(capabilities.get("table_name") or "").strip()
        if "." in table_name:
            return table_name
        raw_catalog = capabilities.get("semantic_catalog")
        if isinstance(raw_catalog, dict):
            dataset = (raw_catalog.get("datasets") or {}).get(source.key)
            if isinstance(dataset, dict):
                table_name = str(dataset.get("table") or "").strip()
                if "." in table_name:
                    return table_name
        return None

    def _data_source_summary(self, source: DataSource) -> dict[str, Any]:
        return {
            "id": source.id,
            "key": source.key,
            "name": source.name,
            "description": source.description,
            "dialect": source.dialect,
            "connection_url": source.connection_url,
            "schema_name": source.schema_name,
            "is_active": source.is_active,
            "is_default": source.is_default,
            "allowed_roles_json": source.allowed_roles_json or [],
            "capabilities_json": source.capabilities_json or {},
            "created_at": source.created_at,
            "updated_at": source.updated_at,
        }

    def _psycopg_conn_string(self) -> str:
        return settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)

    def _dedupe(self, items: list[str]) -> list[str]:
        result: list[str] = []
        for item in items:
            cleaned = str(item).strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result

    def _default_time_mappings(self) -> dict[str, dict[str, str]]:
        return {
            "вчера": {"label": "Вчера", "kind": "yesterday", "grain": "day"},
            "за вчера": {"label": "Вчера", "kind": "yesterday", "grain": "day"},
            "сегодня": {"label": "Сегодня", "kind": "today", "grain": "day"},
            "за сегодня": {"label": "Сегодня", "kind": "today", "grain": "day"},
            "за прошлую неделю": {"label": "Прошлая неделя", "kind": "previous_week", "grain": "day"},
            "прошлую неделю": {"label": "Прошлая неделя", "kind": "previous_week", "grain": "day"},
            "за текущую неделю": {"label": "Текущая неделя", "kind": "current_week", "grain": "day"},
            "текущая неделя": {"label": "Текущая неделя", "kind": "current_week", "grain": "day"},
            "за прошлый месяц": {"label": "Прошлый месяц", "kind": "previous_month", "grain": "day"},
            "прошлый месяц": {"label": "Прошлый месяц", "kind": "previous_month", "grain": "day"},
            "за текущий месяц": {"label": "Текущий месяц", "kind": "current_month", "grain": "day"},
            "текущий месяц": {"label": "Текущий месяц", "kind": "current_month", "grain": "day"},
            "за прошлый год": {"label": "Прошлый год", "kind": "previous_year", "grain": "month"},
            "прошлый год": {"label": "Прошлый год", "kind": "previous_year", "grain": "month"},
            "за текущий год": {"label": "Текущий год", "kind": "current_year", "grain": "month"},
            "текущий год": {"label": "Текущий год", "kind": "current_year", "grain": "month"},
        }


csv_autoconfig_service = CsvAutoConfigService()
