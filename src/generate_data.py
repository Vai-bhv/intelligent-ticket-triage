from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_TICKETS = 12_000

CATEGORIES = {
    "Access Management": {
        "base_hours": 10,
        "sla_hours": 24,
        "descriptions": [
            "User cannot access the application",
            "Account is locked after multiple login attempts",
            "New employee requires system access",
            "Permissions are missing for shared folder",
            "User access needs to be removed",
        ],
    },
    "Application Support": {
        "base_hours": 22,
        "sla_hours": 48,
        "descriptions": [
            "Application fails when submitting the request",
            "Unexpected error appears during processing",
            "Application performance is very slow",
            "User cannot save changes in the system",
            "Application page remains blank after login",
        ],
    },
    "Data Quality": {
        "base_hours": 30,
        "sla_hours": 72,
        "descriptions": [
            "Dashboard contains incorrect worker information",
            "Duplicate records appear in the report",
            "Data is missing from the latest refresh",
            "Incorrect cost centre assigned to worker",
            "Report totals do not match the source system",
        ],
    },
    "Infrastructure": {
        "base_hours": 18,
        "sla_hours": 36,
        "descriptions": [
            "Server connection is unavailable",
            "Network latency is affecting users",
            "Scheduled process did not complete",
            "Shared service is currently unavailable",
            "System integration has stopped responding",
        ],
    },
    "Workforce Operations": {
        "base_hours": 16,
        "sla_hours": 48,
        "descriptions": [
            "External worker onboarding is delayed",
            "Worker extension has not been processed",
            "Supplier assignment requires correction",
            "Worker record is missing required details",
            "Offboarding request remains incomplete",
        ],
    },
}

PRIORITIES = {
    "Low": 0.70,
    "Medium": 1.00,
    "High": 1.35,
    "Critical": 1.75,
}

TEAMS = [
    "Global Operations",
    "Regional Support",
    "Application Services",
    "Data Management",
]

REGIONS = [
    "Europe",
    "North America",
    "Asia Pacific",
]

CHANNELS = [
    "Portal",
    "Email",
    "Phone",
    "Monitoring",
]

URGENCY_TERMS = [
    "",
    "",
    "",
    " urgent",
    " business critical",
    " blocking multiple users",
    " production issue",
]


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def generate_ticket_data() -> pd.DataFrame:
    """Generate realistic synthetic enterprise support tickets."""
    rng = np.random.default_rng(RANDOM_SEED)

    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2025-12-31")
    total_days = (end_date - start_date).days

    rows = []

    category_names = list(CATEGORIES.keys())
    priority_names = list(PRIORITIES.keys())

    for ticket_number in range(1, NUMBER_OF_TICKETS + 1):
        category = rng.choice(category_names)
        category_config = CATEGORIES[category]

        priority = rng.choice(
            priority_names,
            p=[0.28, 0.43, 0.22, 0.07],
        )

        region = rng.choice(
            REGIONS,
            p=[0.38, 0.35, 0.27],
        )

        assigned_team = rng.choice(TEAMS)
        channel = rng.choice(CHANNELS, p=[0.50, 0.27, 0.13, 0.10])

        created_at = (
            start_date
            + pd.Timedelta(days=int(rng.integers(0, total_days + 1)))
            + pd.Timedelta(hours=int(rng.integers(0, 24)))
        )

        description = rng.choice(category_config["descriptions"])
        description += rng.choice(URGENCY_TERMS)

        description_length = len(description.split())

        backlog_at_creation = int(
            np.clip(
                rng.normal(34, 15),
                0,
                120,
            )
        )

        team_experience_years = float(
            np.clip(
                rng.normal(3.8, 1.7),
                0.5,
                10,
            )
        )

        previous_reassignments = int(
            rng.choice(
                [0, 1, 2, 3, 4],
                p=[0.52, 0.27, 0.13, 0.06, 0.02],
            )
        )

        weekend_created = int(created_at.dayofweek >= 5)

        priority_multiplier = PRIORITIES[priority]

        workload_multiplier = (
            1
            + backlog_at_creation / 180
            + previous_reassignments * 0.14
        )

        experience_multiplier = max(
            0.65,
            1.18 - team_experience_years * 0.045,
        )

        weekend_multiplier = 1.20 if weekend_created else 1.0

        random_noise = rng.lognormal(
            mean=0,
            sigma=0.32,
        )

        resolution_hours = (
            category_config["base_hours"]
            * workload_multiplier
            * experience_multiplier
            * weekend_multiplier
            * random_noise
            / priority_multiplier
        )

        resolution_hours = max(1.0, resolution_hours)

        sla_target_hours = category_config["sla_hours"]

        sla_breached = int(
            resolution_hours > sla_target_hours
        )

        rows.append(
            {
                "ticket_id": f"TKT-{ticket_number:06d}",
                "created_at": created_at,
                "region": region,
                "category": category,
                "priority": priority,
                "assigned_team": assigned_team,
                "channel": channel,
                "description": description,
                "description_length": description_length,
                "backlog_at_creation": backlog_at_creation,
                "team_experience_years": round(
                    team_experience_years,
                    2,
                ),
                "previous_reassignments": previous_reassignments,
                "weekend_created": weekend_created,
                "sla_target_hours": sla_target_hours,
                "resolution_hours": round(
                    resolution_hours,
                    2,
                ),
                "sla_breached": sla_breached,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("created_at")
        .reset_index(drop=True)
    )


def main() -> None:
    dataset = generate_ticket_data()

    output_path = (
        get_project_root()
        / "data"
        / "raw"
        / "support_tickets.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    print("Synthetic ticket dataset created successfully.")
    print(f"Saved to: {output_path}")
    print(f"Rows: {len(dataset):,}")
    print(
        f"SLA breach rate: "
        f"{dataset['sla_breached'].mean() * 100:.2f}%"
    )
    print()
    print(dataset.head())


if __name__ == "__main__":
    main()