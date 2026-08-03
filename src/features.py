from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data() -> pd.DataFrame:
    """Load the synthetic support ticket dataset."""
    return pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "raw"
        / "support_tickets.csv",
        parse_dates=["created_at"],
    )


def engineer_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create machine-learning features."""

    df = data.copy()

    df["day_of_week"] = (
        df["created_at"]
        .dt.dayofweek
    )

    df["month"] = (
        df["created_at"]
        .dt.month
    )

    df["quarter"] = (
        df["created_at"]
        .dt.quarter
    )

    df["hour"] = (
        df["created_at"]
        .dt.hour
    )

    df["is_month_end"] = (
        df["created_at"]
        .dt.is_month_end
        .astype(int)
    )

    df["ticket_age_bucket"] = pd.cut(
        df["backlog_at_creation"],
        bins=[0,20,40,60,120],
        labels=[
            "Low",
            "Medium",
            "High",
            "Critical",
        ],
    )

    return df


def main():

    data = load_data()

    processed = engineer_features(data)

    output = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "tickets_features.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        output,
        index=False,
    )

    print("Processed dataset saved.")

    print(processed.head())


if __name__ == "__main__":
    main()