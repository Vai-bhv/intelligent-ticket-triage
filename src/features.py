from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "support_tickets.csv"
FEATURE_SCHEMA_VERSION = 1

TEXT_COLUMN = "description"
NUMERICAL_FEATURES = (
    "description_length",
    "backlog_at_creation",
    "team_experience_years",
    "previous_reassignments",
    "weekend_created",
    "sla_target_hours",
    "day_of_week",
    "month",
    "quarter",
    "hour",
    "is_month_end",
)
CATEGORICAL_FEATURES = (
    "region",
    "category",
    "priority",
    "assigned_team",
    "channel",
    "backlog_bucket",
)
FEATURE_COLUMNS = (TEXT_COLUMN, *NUMERICAL_FEATURES, *CATEGORICAL_FEATURES)
RAW_FEATURE_COLUMNS = (
    TEXT_COLUMN,
    "created_at",
    "backlog_at_creation",
    "team_experience_years",
    "previous_reassignments",
    "sla_target_hours",
    *CATEGORICAL_FEATURES[:-1],
)


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw tickets for feature engineering."""
    return pd.read_csv(path)


def parse_created_at(values: pd.Series) -> pd.Series:
    """Normalize timestamps to UTC, interpreting naive timestamps as UTC."""
    timestamps = pd.to_datetime(values, format="mixed", utc=True, errors="raise")
    if timestamps.isna().any():
        raise ValueError("Every ticket must have a creation timestamp.")
    return timestamps


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derive model inputs from raw ticket fields without fitting any state."""
    missing = set(RAW_FEATURE_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Missing raw ticket columns: {sorted(missing)}")

    tickets = data.copy()
    tickets["created_at"] = parse_created_at(tickets["created_at"])
    created_at = tickets["created_at"].dt
    tickets[TEXT_COLUMN] = tickets[TEXT_COLUMN].astype("string").fillna("")
    tickets["description_length"] = tickets[TEXT_COLUMN].str.split().str.len().astype(int)
    tickets["day_of_week"] = created_at.dayofweek
    tickets["weekend_created"] = (created_at.dayofweek >= 5).astype(int)
    tickets["month"] = created_at.month
    tickets["quarter"] = created_at.quarter
    tickets["hour"] = created_at.hour
    tickets["is_month_end"] = created_at.is_month_end.astype(int)
    tickets["backlog_bucket"] = pd.cut(
        tickets["backlog_at_creation"],
        bins=[0, 20, 40, 60, float("inf")],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    ).astype(object)
    return tickets


def main() -> None:
    """Optionally export a feature inspection CSV."""
    output = PROJECT_ROOT / "data" / "processed" / "tickets_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    engineer_features(load_data()).to_csv(output, index=False)
    print(f"Processed dataset saved to {output}")


if __name__ == "__main__":
    main()
