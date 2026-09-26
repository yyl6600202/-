"""Batch-generate Day 1 graph profiles and compact case summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

try:
    from graph_features import profile_graph
except ImportError:  # pragma: no cover - useful when imported from another cwd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from graph_features import profile_graph


CASE_RE = re.compile(r"^case_(\d+)\.json$")


def iter_case_paths(data_dir: Path):
    paths = []
    for path in data_dir.iterdir():
        match = CASE_RE.match(path.name)
        if path.is_file() and match:
            paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(paths)]


def read_capacity_file(path: Path):
    values = {}
    section = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        parts = line.split()
        if section == "capacity" and len(parts) == 2:
            values[parts[0]] = int(parts[1])
    missing = {"L1", "UB"} - set(values)
    if missing:
        raise ValueError(f"{path}: missing capacities {sorted(missing)}")
    return {key: values[key] for key in ("L1", "UB")}


def profile_case(path: Path, capacities):
    graph = json.loads(path.read_text(encoding="utf-8"))
    features = profile_graph(graph, capacities)
    features["case_id"] = path.stem
    # Keep the CSV schema stable even if the feature extractor gains fields.
    return features


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def generate_profiles(data_dir: Path, output: Path, summary_output: Path, config: Path):
    capacities = read_capacity_file(config)
    case_paths = iter_case_paths(data_dir)
    if not case_paths:
        raise ValueError(f"no case_###.json files found in {data_dir}")
    rows = []
    for index, path in enumerate(case_paths, 1):
        row = profile_case(path, capacities)
        row["profile_index"] = index
        rows.append(row)

    base_fields = [
        "profile_index", "case_id", "graph_class", "ops_total", "noncopy_ops",
        "copy_in_ops", "copy_out_ops", "tensors_total", "edges_total",
        "op_dag_edges", "op_dag_depth", "op_dag_max_width", "op_dag_avg_width",
        "critical_path_cycles", "noncopy_cycles", "max_tensor_size",
        "max_tensor_fanout", "ddr_repeat_tensor_count", "ddr_repeat_potential_bytes",
        "l1_peak_bytes", "ub_peak_bytes", "l1_peak_ratio", "ub_peak_ratio",
        "spill_risk_ratio", "l1_tensor_count", "l1_tensor_bytes", "ub_tensor_count",
        "ub_tensor_bytes", "ddr_tensor_count", "ddr_tensor_bytes",
    ]
    pipe_fields = sorted(
        field for field in rows[0]
        if field.startswith("pipe_") and not field.endswith("_noncopy_cycles")
    )
    fields = base_fields + pipe_fields
    write_csv(output, rows, fields)
    write_csv(summary_output, rows, [
        "profile_index", "case_id", "graph_class", "ops_total", "noncopy_ops",
        "op_dag_depth", "op_dag_max_width", "critical_path_cycles",
        "l1_peak_bytes", "ub_peak_bytes", "spill_risk_ratio",
        "ddr_repeat_tensor_count", "ddr_repeat_potential_bytes",
    ])
    return len(rows), fields


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Day 1 case_profile.csv")
    parser.add_argument("data_dir", nargs="?", default="data", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("case_profile.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("case_summary.csv"))
    args = parser.parse_args(argv)
    config = args.config or args.data_dir / "config.txt"
    count, fields = generate_profiles(args.data_dir, args.output, args.summary_output, config)
    print(f"OK: profiled {count} cases; columns={len(fields)}; output={args.output}; summary={args.summary_output}")


if __name__ == "__main__":
    _main()
