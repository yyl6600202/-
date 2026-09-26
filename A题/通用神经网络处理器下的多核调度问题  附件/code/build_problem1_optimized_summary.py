"""Build a reproducible incumbent table from a baseline and official sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from evaluation_validation import read_evaluation_config
from initial_partition import generate_candidates


FIELDS = (
    "case", "num_cores", "graph_class", "selected_rule", "baseline_makespan",
    "problem1_makespan", "speedup", "baseline_added_copy_bytes",
    "problem1_added_copy_bytes", "problem1_spill_added_copy_bytes",
    "num_blocks", "evaluated_candidates", "rejected_candidates",
    "selection", "changed_vs_current", "error",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _number(value, default=0.0):
    return float(value) if value not in (None, "") else default


def _candidate_key(row):
    # Makespan is the primary objective; use traffic and plan size only to break ties.
    return (
        int(row["makespan"]),
        int(row.get("added_copy_bytes") or 0),
        int(row.get("num_blocks") or 0),
        row["rule"] != "single",
    )


def build(baseline_path, sweep_paths, output_path, plans_dir, stats_path,
          data_dir, num_cores):
    baseline_rows = read_csv(baseline_path)
    if len(baseline_rows) != 100 or len({row["case"] for row in baseline_rows}) != 100:
        raise ValueError("baseline must contain exactly one row for each of 100 cases")
    baseline_by_case = {row["case"]: row for row in baseline_rows}
    sweep_by_case = {}
    for path in sweep_paths:
        for row in read_csv(path):
            if int(row["num_cores"]) != num_cores:
                raise ValueError(f"sweep {path} contains a different core count")
            sweep_by_case.setdefault(row["case"], []).append(row)

    output_path = Path(output_path)
    plans_dir = Path(plans_dir)
    stats_path = Path(stats_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    changed = []
    for case_name in sorted(baseline_by_case):
        original = baseline_by_case[case_name]
        candidates = sweep_by_case.get(case_name, [])
        successful = [row for row in candidates if not row.get("error") and row.get("makespan")]
        rejected = len(candidates) - len(successful)
        if successful:
            baseline_candidate = next(
                (row for row in successful if row["rule"] == "single"), None
            )
            if baseline_candidate is None:
                raise ValueError(f"{case_name}: candidate sweep is missing the single baseline")
            if int(baseline_candidate["makespan"]) != int(original["baseline_makespan"]):
                raise ValueError(f"{case_name}: sweep and incumbent disagree on baseline makespan")
            winner = min(successful, key=_candidate_key)
            winner_speedup = _number(winner["speedup"])
            if winner_speedup < 1.0 - 1e-12:
                raise ValueError(f"{case_name}: sweep winner regressed below its single baseline")
            chosen_rule = winner["rule"]
            chosen_makespan = int(winner["makespan"])
            added_copy_bytes = int(winner.get("added_copy_bytes") or 0)
            spill_bytes = int(winner.get("spill_added_copy_bytes") or 0)
            block_count = int(winner.get("num_blocks") or 0)
            selection = "best_official_sweep_candidate"
            graph_class = winner["graph_class"] or original["graph_class"]
            speedup = winner_speedup
            baseline_added = int(original.get("baseline_added_copy_bytes") or 0)
            changed_vs_current = (
                chosen_rule != original["rule"]
                or chosen_makespan != int(original["problem1_makespan"])
            )
            if changed_vs_current:
                changed.append(case_name)
                graph = json.loads(
                    (Path(data_dir) / f"{case_name}.json").read_text(encoding="utf-8")
                )
                candidate = next(
                    item for item in generate_candidates(
                        graph, num_cores, target_blocks=num_cores, rules=(chosen_rule,)
                    )
                    if item["rule"] == chosen_rule
                )
                if not candidate["valid"]:
                    raise ValueError(
                        f"{case_name}/{chosen_rule}: selected plan fails structural validation"
                    )
                (plans_dir / f"{case_name}.json").write_text(
                    json.dumps(candidate["plan"], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        else:
            chosen_rule = original["rule"]
            chosen_makespan = int(original["problem1_makespan"])
            added_copy_bytes = int(original.get("problem1_added_copy_bytes") or 0)
            spill_bytes = int(original.get("problem1_spill_added_copy_bytes") or 0)
            block_count = int(original.get("num_blocks") or 0)
            selection = "existing_incumbent"
            graph_class = original["graph_class"]
            speedup = _number(original["speedup"])
            baseline_added = int(original.get("baseline_added_copy_bytes") or 0)
            changed_vs_current = False
        rows.append({
            "case": case_name,
            "num_cores": num_cores,
            "graph_class": graph_class,
            "selected_rule": chosen_rule,
            "baseline_makespan": int(original["baseline_makespan"]),
            "problem1_makespan": chosen_makespan,
            "speedup": speedup,
            "baseline_added_copy_bytes": baseline_added,
            "problem1_added_copy_bytes": added_copy_bytes,
            "problem1_spill_added_copy_bytes": spill_bytes,
            "num_blocks": block_count,
            "evaluated_candidates": len(successful),
            "rejected_candidates": rejected,
            "selection": selection,
            "changed_vs_current": changed_vs_current,
            "error": "",
        })

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    speeds = [float(row["speedup"]) for row in rows]
    sorted_speeds = sorted(speeds)
    baseline_cycles = sum(int(row["baseline_makespan"]) for row in rows)
    candidate_cycles = sum(int(row["problem1_makespan"]) for row in rows)
    stats = {
        "num_cores": num_cores,
        "num_cases": len(rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "arithmetic_mean_speedup": sum(speeds) / len(speeds),
        "geometric_mean_speedup": math.exp(sum(math.log(value) for value in speeds) / len(speeds)),
        "median_speedup": (sorted_speeds[49] + sorted_speeds[50]) / 2,
        "p10_speedup": sorted_speeds[9],
        "minimum_speedup": sorted_speeds[0],
        "maximum_speedup": sorted_speeds[-1],
        "faster_cases": sum(value > 1.0 for value in speeds),
        "equal_cases": sum(value == 1.0 for value in speeds),
        "slower_cases": sum(value < 1.0 for value in speeds),
        "baseline_total_cycles": baseline_cycles,
        "problem1_total_cycles": candidate_cycles,
        "aggregate_cycle_ratio": baseline_cycles / candidate_cycles,
        "cases_with_sweep": len(sweep_by_case),
        "changed_cases": changed,
        "selected_rule_counts": dict(Counter(row["selected_rule"] for row in rows)),
    }
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plans-dir", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-cores", type=int, default=4)
    args = parser.parse_args(argv)
    result = build(
        args.baseline, args.sweep, args.output, args.plans_dir, args.stats,
        args.data_dir, args.num_cores,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
