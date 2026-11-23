"""Request and response contracts for the inference API."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, field_validator


Priority = Literal["Low", "Medium", "High", "Critical"]
Region = Literal["Europe", "North America", "Asia Pacific"]
Category = Literal[
    "Access Management",
    "Application Support",
    "Data Quality",
    "Infrastructure",
    "Workforce Operations",
]
AssignedTeam = Literal[
    "Global Operations", "Regional Support", "Application Services", "Data Management"
]
Channel = Literal["Portal", "Email", "Phone", "Monitoring"]
RiskClassification = Literal["Low", "Moderate", "High", "Critical"]
MAX_DESCRIPTION_LENGTH = 4096
UnitInterval = Annotated[float, Field(ge=0, le=1)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_DESCRIPTION_LENGTH),
    ]
    region: Region
    category: Category
    priority: Priority
    assigned_team: AssignedTeam
    channel: Channel
    backlog_at_creation: Annotated[int, Field(strict=True, ge=0, le=120)]
    team_experience_years: Annotated[float, Field(strict=True, ge=0.5, le=10)]
    previous_reassignments: Annotated[int, Field(strict=True, ge=0, le=4)]
    sla_target_hours: Annotated[int, Field(strict=True, ge=1, le=168)]
    created_at: AwareDatetime = Field(
        description="ISO 8601 timestamp with a timezone offset and a year from 1900 through 2100."
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def require_timestamp(cls, value: object) -> object:
        if isinstance(value, datetime) or (isinstance(value, str) and "T" in value):
            return value
        raise ValueError("Use an ISO 8601 datetime with a timezone offset.")

    @field_validator("created_at")
    @classmethod
    def limit_creation_year(cls, value: datetime) -> datetime:
        if not 1900 <= value.year <= 2100:
            raise ValueError("Creation year must be between 1900 and 2100.")
        return value


class PredictionResponse(BaseModel):
    model_version: str
    risk_score: UnitInterval
    decision_threshold: UnitInterval
    risk_classification: RiskClassification
    suggested_priority: Priority
    recommendations: list[str]


class HoldoutMetrics(BaseModel):
    average_precision: UnitInterval
    roc_auc: UnitInterval
    precision: UnitInterval
    recall: UnitInterval
    f2: UnitInterval


class ModelInfoResponse(BaseModel):
    model_version: str
    model_name: str
    feature_schema_version: Annotated[int, Field(ge=1)]
    decision_threshold: UnitInterval
    evaluation_data: Literal["synthetic"]
    holdout_metrics: HoldoutMetrics


class LiveResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]


class ErrorResponse(BaseModel):
    detail: str


class ValidationIssue(BaseModel):
    field: str
    code: str
    message: str


class ValidationErrorResponse(BaseModel):
    detail: list[ValidationIssue]
