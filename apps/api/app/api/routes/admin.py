from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.data_sources.registry import data_source_registry
from app.core.dependencies import get_db, require_role
from app.models.user import UserRole
from app.schemas.admin import (
    AuditLogSummary,
    CsvAutoConfigResponse,
    CreateUserRequest,
    DataSourceRequest,
    DataSourceSummary,
    RoleUpdateRequest,
    SemanticEntryRequest,
    SemanticEntrySummary,
    TemplateRequest,
    TemplateSummary,
    UserStatusUpdateRequest,
)
from app.schemas.common import MessageResponse, UserSummary
from app.services.admin_service import AdminService
from app.services.audit_service import AuditService
from app.services.csv_autoconfig_service import csv_autoconfig_service

router = APIRouter()


@router.get("/users", response_model=list[UserSummary])
def list_users(db: Session = Depends(get_db), _=Depends(require_role(UserRole.admin))):
    items = AdminService(db).list_users()
    return [UserSummary.model_validate(item, from_attributes=True) for item in items]


@router.post("/users", response_model=UserSummary)
def create_user(payload: CreateUserRequest, db: Session = Depends(get_db), _=Depends(require_role(UserRole.admin))):
    user = AdminService(db).create_user(payload)
    return UserSummary.model_validate(user, from_attributes=True)


@router.put("/users/{user_id}/role", response_model=UserSummary)
def update_role(user_id: str, payload: RoleUpdateRequest, db: Session = Depends(get_db), _=Depends(require_role(UserRole.admin))):
    service = AdminService(db)
    user = service.users.get_by_id(user_id)
    updated = service.update_user_role(user, payload.role)
    return UserSummary.model_validate(updated, from_attributes=True)


@router.put("/users/{user_id}/status", response_model=UserSummary)
def update_user_status(
    user_id: str,
    payload: UserStatusUpdateRequest,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.admin)),
):
    service = AdminService(db)
    user = service.users.get_by_id(user_id)
    updated = service.update_user_status(user, payload.is_active)
    return UserSummary.model_validate(updated, from_attributes=True)


@router.get("/semantic-entries", response_model=list[SemanticEntrySummary])
def list_semantic_entries(db: Session = Depends(get_db), _=Depends(require_role(UserRole.analyst, UserRole.admin))):
    items = AdminService(db).list_semantic_entries()
    return [SemanticEntrySummary.model_validate(item, from_attributes=True) for item in items]


@router.post("/semantic-entries", response_model=SemanticEntrySummary)
def create_semantic_entry(
    payload: SemanticEntryRequest,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.analyst, UserRole.admin)),
):
    item = AdminService(db).create_semantic_entry(payload)
    return SemanticEntrySummary.model_validate(item, from_attributes=True)


@router.get("/templates", response_model=list[TemplateSummary])
def list_templates(db: Session = Depends(get_db), _=Depends(require_role(UserRole.analyst, UserRole.admin))):
    items = AdminService(db).list_templates()
    return [TemplateSummary.model_validate(item, from_attributes=True) for item in items]


@router.post("/templates", response_model=TemplateSummary)
def create_template(
    payload: TemplateRequest,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.analyst, UserRole.admin)),
):
    item = AdminService(db).create_template(payload)
    return TemplateSummary.model_validate(item, from_attributes=True)


@router.get("/audit-logs", response_model=list[AuditLogSummary])
def list_audit_logs(db: Session = Depends(get_db), _=Depends(require_role(UserRole.admin))):
    items = AuditService(db).list_recent()
    return [AuditLogSummary.model_validate(item, from_attributes=True) for item in items]


@router.get("/data-sources", response_model=list[DataSourceSummary])
def list_data_sources(db: Session = Depends(get_db), _=Depends(require_role(UserRole.admin))):
    items = AdminService(db).list_data_sources()
    return [DataSourceSummary.model_validate(item, from_attributes=True) for item in items]


@router.post("/data-sources", response_model=DataSourceSummary)
def create_data_source(
    payload: DataSourceRequest,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.admin)),
):
    item = AdminService(db).create_data_source(payload)
    data_source_registry.invalidate()
    return DataSourceSummary.model_validate(item, from_attributes=True)


@router.put("/data-sources/{source_id}", response_model=DataSourceSummary)
def update_data_source(
    source_id: str,
    payload: DataSourceRequest,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.admin)),
):
    service = AdminService(db)
    source = service.data_sources.get_by_id(source_id)
    updated = service.update_data_source(source, payload)
    data_source_registry.invalidate()
    return DataSourceSummary.model_validate(updated, from_attributes=True)


@router.post("/data-sources/{source_id}/activate", response_model=DataSourceSummary)
def activate_data_source(
    source_id: str,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.admin)),
):
    updated = csv_autoconfig_service.activate_uploaded_source(db=db, source_id=source_id)
    return DataSourceSummary.model_validate(updated, from_attributes=True)


@router.post("/data-sources/{source_id}/optimize", response_model=DataSourceSummary)
def optimize_data_source(
    source_id: str,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.admin)),
):
    updated = csv_autoconfig_service.optimize_uploaded_source(db=db, source_id=source_id)
    return DataSourceSummary.model_validate(updated, from_attributes=True)


@router.delete("/data-sources/{source_id}", response_model=MessageResponse)
def delete_data_source(
    source_id: str,
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.admin)),
):
    deleted_name = csv_autoconfig_service.delete_uploaded_source(db=db, source_id=source_id)
    data_source_registry.invalidate()
    return MessageResponse(message=f"Dataset deleted: {deleted_name}")


@router.post("/data-sources/auto-config/csv", response_model=CsvAutoConfigResponse)
async def auto_config_from_csv(
    file: UploadFile = File(...),
    source_key: str | None = Form(None),
    table_name: str | None = Form(None),
    display_name: str | None = Form(None),
    delimiter: str = Form("auto"),
    auto_mode: bool = Form(True),
    apply: bool = Form(False),
    activate: bool = Form(True),
    use_llm: bool = Form(True),
    db: Session = Depends(get_db),
    _=Depends(require_role(UserRole.admin)),
):
    suffix = Path(file.filename or "dataset.csv").suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
    try:
        result = csv_autoconfig_service.analyze_and_build_file(
            csv_path=tmp_path,
            source_key=source_key,
            table_name=table_name,
            delimiter=delimiter,
            apply=apply,
            auto_mode=auto_mode,
            db=db,
            filename=file.filename,
            display_name=display_name,
            activate=activate,
            use_llm=use_llm,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    if apply:
        data_source_registry.invalidate()
    return CsvAutoConfigResponse.model_validate(result)


def _legacy_auto_config_from_csv_bytes(
    payload: bytes,
    *,
    source_key: str | None,
    table_name: str | None,
    display_name: str | None,
    delimiter: str,
    auto_mode: bool,
    apply: bool,
    activate: bool,
    use_llm: bool,
    filename: str | None,
    db: Session,
):
    return csv_autoconfig_service.analyze_and_build(
        csv_bytes=payload,
        source_key=source_key,
        table_name=table_name,
        delimiter=delimiter,
        apply=apply,
        auto_mode=auto_mode,
        db=db,
        filename=file.filename,
        display_name=display_name,
        activate=activate,
        use_llm=use_llm,
    )
