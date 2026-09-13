"""LIBERO-plus metadata and reports, adapted from the X-VLA evaluation helpers.

Kept local so VLA-Adapter works without a sibling repository or ML dependencies.
The per-task schema and category aggregation match the openpi/X-VLA reports.
"""
from __future__ import annotations

import collections
import json
import logging
import os
from pathlib import Path
import tempfile


LIBERO_PLUS_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CLASSIFICATION_FILE = Path("libero/libero/benchmark/task_classification.json")
STAT_HEADERS = ["Category", "Logged tasks", "Expected tasks", "Episodes", "Successes", "Success rate"]


def load_metadata(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    metadata = {}
    for suite in LIBERO_PLUS_SUITES:
        if suite not in raw:
            raise ValueError(f"Missing LIBERO-plus suite {suite} in {path}")
        metadata[suite] = {}
        for item in raw[suite]:
            name = item["name"]
            if name in metadata[suite]:
                raise ValueError(f"Duplicate LIBERO-plus task in {suite}: {name}")
            metadata[suite][name] = {"id": item["id"], "category": item["category"]}
        if not metadata[suite]:
            raise ValueError(f"No task classifications for {suite} in {path}")
    return metadata


def ensure_libero_config(root: Path, config_dir: Path) -> Path:
    """Create LIBERO's non-interactive path config without touching ~/.libero."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        package_root = root / "libero/libero"
        config = {
            "benchmark_root": str(package_root),
            "bddl_files": str(package_root / "bddl_files"),
            "init_states": str(package_root / "init_files"),
            "datasets": str(root / "libero/datasets"),
            "assets": str(package_root / "assets"),
        }
        # JSON is valid YAML and avoids importing PyYAML before LIBERO is configured.
        config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        logging.info("Created non-interactive LIBERO-plus config at %s", config_file)
    return config_file


def validate_checkout(root: Path) -> None:
    required_paths = [
        root / CLASSIFICATION_FILE,
        root / "libero/libero/bddl_files",
        root / "libero/libero/init_files",
        root / "libero/libero/assets",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete LIBERO-plus checkout at {root}; missing required paths:\n" + "\n".join(missing)
        )


def task_results_path(output_dir: Path, save_name: str, suite: str) -> Path:
    return output_dir / "results" / f"{save_name}_plus_{suite}_tasks.jsonl"


def read_task_results(path: Path, suite: str, suite_metadata: dict) -> list:
    """Read the latest record per task, including logs produced by openpi."""
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            name = record["task_name"]
            if record["suite"] != suite:
                raise ValueError(f"Expected suite {suite} in {path}, line {line_number}")
            if name not in suite_metadata:
                raise ValueError(f"Unknown LIBERO-plus task in {suite} results: {name}")
            if record["category"] != suite_metadata[name]["category"]:
                raise ValueError(f"Category mismatch for {suite}/{name} in {path}")
            episodes, successes = record["episodes"], record["successes"]
            if (type(episodes) is not int or type(successes) is not int
                    or episodes <= 0 or not 0 <= successes <= episodes):
                raise ValueError(f"Invalid episode/success counts for {suite}/{name} in {path}")
            record["success_rate"] = successes / episodes
            records[name] = record
    return list(records.values())


def write_task_results(path: Path, records: list) -> None:
    """Replace a task snapshot atomically so interruption preserves prior progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def aggregate_records(records: list) -> dict:
    stats = {}
    for record in records:
        categories = stats.setdefault(record["suite"], {})
        category = categories.setdefault(record["category"], {"tasks": 0, "episodes": 0, "successes": 0})
        category["tasks"] += 1
        category["episodes"] += record["episodes"]
        category["successes"] += record["successes"]
    return stats


def total_stats(categories: dict) -> dict:
    return {key: sum(stats[key] for stats in categories.values()) for key in ("tasks", "episodes", "successes")}


def success_rate(stats: dict) -> float:
    return stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0


def format_rate(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def markdown_table(headers: list, rows: list) -> str:
    return "\n".join(
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    ) + "\n"


def stats_row(name: str, stats: dict, expected: int) -> list:
    return [name, stats["tasks"], expected, stats["episodes"], stats["successes"], format_rate(success_rate(stats))]


def suite_summary(suite: str, categories: dict, expected_counts: dict) -> str:
    rows = [
        stats_row(category, categories.get(category, total_stats({})), expected_counts[category])
        for category in sorted(expected_counts)
    ]
    total = total_stats(categories)
    expected = sum(expected_counts.values())
    rows.append(stats_row("**Overall**", total, expected))
    text = f"### {suite}\n\n" + markdown_table(STAT_HEADERS, rows)
    if total["tasks"] != expected:
        text += f"\n> WARNING: logged {total['tasks']} of {expected} expected tasks for `{suite}`.\n"
    return text


def write_suite_summary(output_dir: Path, save_name: str, suite: str, records: list, suite_metadata: dict) -> Path:
    categories = aggregate_records(records).get(suite, {})
    counts = collections.Counter(item["category"] for item in suite_metadata.values())
    path = output_dir / "results" / f"{save_name}_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"# LIBERO-plus Suite Result\n\n**Save name:** `{save_name}`\n\n")
        handle.write(suite_summary(suite, categories, counts) + "\n")
    return path


def write_summary(output_dir: Path, save_name: str, metadata: dict) -> Path:
    paths = {suite: task_results_path(output_dir, save_name, suite) for suite in LIBERO_PLUS_SUITES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing LIBERO-plus per-task result files:\n" + "\n".join(missing))
    records = []
    for suite, path in paths.items():
        records.extend(read_task_results(path, suite, metadata[suite]))
    stats = aggregate_records(records)
    counts = {
        suite: collections.Counter(item["category"] for item in metadata[suite].values())
        for suite in LIBERO_PLUS_SUITES
    }
    lines = [f"# LIBERO-plus Category Summary\n\n**Save name:** `{save_name}`\n\n",
             "## Per-Suite Category Results\n\n"]
    suite_rows, suite_totals = [], {}
    for suite in LIBERO_PLUS_SUITES:
        categories = stats.get(suite, {})
        lines.append(suite_summary(suite, categories, counts[suite]) + "\n")
        suite_totals[suite] = total_stats(categories)
        suite_rows.append(stats_row(suite, suite_totals[suite], sum(counts[suite].values())))
    suite_rows.append(stats_row("**All cases**", total_stats(suite_totals), sum(map(len, metadata.values()))))
    lines.extend(["## Suite Averages\n\n",
                  markdown_table(["Suite", *STAT_HEADERS[1:-1], "Average success rate"], suite_rows),
                  "\n## Four-Suite Category Averages\n\n"])
    category_rows = []
    for category in sorted({category for suite_counts in counts.values() for category in suite_counts}):
        rates = [success_rate(stats.get(suite, {}).get(category, total_stats({}))) for suite in LIBERO_PLUS_SUITES]
        category_rows.append([category, format_rate(sum(rates) / len(rates)), *map(format_rate, rates)])
    lines.append(markdown_table(["Category", "Average success rate", *LIBERO_PLUS_SUITES], category_rows))
    path = output_dir / "results" / f"{save_name}_plus_category_summary.md"
    path.write_text("".join(lines), encoding="utf-8")
    logging.info("Wrote LIBERO-plus category summary to %s", path)
    return path


def validate_task_suite(suite: str, task_suite, suite_metadata: dict) -> None:
    names = {task_suite.get_task(task_id).name for task_id in range(task_suite.n_tasks)}
    missing, extra = set(suite_metadata) - names, names - set(suite_metadata)
    if missing or extra:
        raise ValueError(
            f"Task/classification mismatch for {suite}: {len(missing)} classified tasks missing from benchmark, "
            f"{len(extra)} unclassified tasks. Check --libero_plus_root and --classification_path; "
            "the loaded benchmark must be LIBERO-plus."
        )


