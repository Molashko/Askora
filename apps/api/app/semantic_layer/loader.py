from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session
import yaml

from app.core.config import settings
from app.semantic_layer.types import SemanticCatalog, TemplateCatalog


class SemanticLayerLoader:
    def __init__(self) -> None:
        self._catalog: SemanticCatalog | None = None
        self._templates: TemplateCatalog | None = None

    def load_catalog(self) -> SemanticCatalog:
        if self._catalog is None:
            path = Path(settings.semantic_catalog_path)
            with path.open("r", encoding="utf-8") as file:
                raw = yaml.safe_load(file)
            self._catalog = SemanticCatalog.model_validate(raw)
        return self._catalog

    def load_catalog_for_db(self, db: Session) -> SemanticCatalog:
        from app.models.data_source import DataSource

        preferred = (
            db.query(DataSource)
            .filter(DataSource.is_active.is_(True), DataSource.is_default.is_(True))
            .order_by(DataSource.updated_at.desc(), DataSource.created_at.desc())
            .all()
        )
        fallback = (
            db.query(DataSource)
            .filter(DataSource.is_active.is_(True))
            .order_by(DataSource.updated_at.desc(), DataSource.created_at.desc())
            .all()
        )

        search_sources = preferred if preferred else fallback
        for source in search_sources:
            capabilities = source.capabilities_json or {}
            raw_catalog = capabilities.get("semantic_catalog")
            if not isinstance(raw_catalog, dict):
                continue
            try:
                return SemanticCatalog.model_validate(raw_catalog)
            except Exception:
                continue

        return self.load_catalog()

    def load_templates(self) -> TemplateCatalog:
        if self._templates is None:
            path = Path(settings.semantic_templates_path)
            with path.open("r", encoding="utf-8") as file:
                raw = yaml.safe_load(file)
            self._templates = TemplateCatalog.model_validate(raw)
        return self._templates

    def invalidate(self) -> None:
        self._catalog = None
        self._templates = None


semantic_loader = SemanticLayerLoader()
