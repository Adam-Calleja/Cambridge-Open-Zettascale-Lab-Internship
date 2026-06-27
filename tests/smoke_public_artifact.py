"""Smoke checks for the public portfolio artifact.

These checks deliberately avoid running the private VictoriaMetrics-dependent
carbon calculation. They verify that the published repo has the files, data
schema, and framework entry points a reviewer should expect.
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "carbon-framework.py",
    "docs/portfolio-case-study.md",
    "data/JulyAnonymizedData-sample.csv",
    "data/partition-info-all.txt",
    "Summer Internship Technical Report.pdf",
    "Summer-Internship-Presentation-Slides.pdf",
]

REQUIRED_JOB_COLUMNS = {
    "JobIDRaw",
    "JobName",
    "Partition",
    "ElapsedRaw",
    "Account",
    "State",
    "CPUTimeRAW",
    "NodeList",
    "User",
    "AllocCPUS",
    "AllocNodes",
    "QOS",
    "Start",
    "End",
    "Timelimit",
    "Suspended",
}

REQUIRED_FRAMEWORK_FUNCTIONS = {
    "_require_victoria_metrics",
    "fFindSharedJobs",
    "sharedSameUser",
    "isExclusive",
    "dfToUTC",
    "getJobPower",
    "getJobEnergy",
    "getCarbonIntensities",
    "getCarbonFootprint",
    "findCarbonEnergy",
}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    assert_true(not missing, f"Missing required public artifact files: {missing}")


def check_sample_job_data() -> None:
    sample_path = ROOT / "data/JulyAnonymizedData-sample.csv"

    with sample_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        rows = list(reader)

    missing_columns = sorted(REQUIRED_JOB_COLUMNS - columns)
    assert_true(not missing_columns, f"Sample data is missing columns: {missing_columns}")
    assert_true(len(rows) >= 100, "Sample data should contain a useful public fixture")

    sample_users = {row["User"] for row in rows[:50] if row.get("User")}
    sample_jobs = {row["JobName"] for row in rows[:50] if row.get("JobName")}

    assert_true(
        any(user.startswith("user_") for user in sample_users),
        "Sample users should use anonymised user_* identifiers",
    )
    assert_true(
        any(job.startswith("job_") for job in sample_jobs),
        "Sample job names should use anonymised job_* identifiers",
    )


def check_partition_data() -> None:
    partition_path = ROOT / "data/partition-info-all.txt"

    with partition_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        columns = reader.fieldnames or []
        rows = list(reader)

    assert_true(columns[:3] == ["PARTITION", "NODES", "CPUS"], "Partition file schema changed")
    assert_true(len(rows) >= 5, "Partition file should contain multiple cluster partitions")


def check_framework_structure() -> None:
    framework_path = ROOT / "carbon-framework.py"
    source = framework_path.read_text()
    tree = ast.parse(source, filename=str(framework_path))

    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    missing_functions = sorted(REQUIRED_FRAMEWORK_FUNCTIONS - function_names)

    assert_true(
        not missing_functions,
        f"Framework is missing expected functions: {missing_functions}",
    )
    assert_true(
        "VICTORIA_METRICS_URL" in source,
        "Framework should keep private VictoriaMetrics access behind configuration",
    )


def main() -> None:
    check_required_files()
    check_sample_job_data()
    check_partition_data()
    check_framework_structure()
    print("Public artifact smoke check passed.")


if __name__ == "__main__":
    main()
