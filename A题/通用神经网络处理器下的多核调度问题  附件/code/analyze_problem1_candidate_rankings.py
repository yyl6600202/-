"""Reproduce cheap-score shortlist checks against official sweep results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from initial_partition import generate_candidates


FIELDS = (
    "case", "num_cores", "graph_class", "rule", "num_blocks",
    "estimated_score", "estimated_partition_added_copy_bytes",
    "official_makespan", "speedup", "official_error", "estimate_rank",
    "in_estimate_top2", "full_sweep_winner", "top2_winner",
    "top2_matches_full_sweep", "top2_plus_single_winner",
    "top2_plus_single_matches_full_sweep",
)


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _best_official(rows):
    return min(rows, key=lambda row: (
        int(row["makespan"]),
        int(row.get("added_copy_bytes") or 0),
        int(row.get("num_blocks") or 0),
        row["rule"] != "single",
    ))


def analyze(sweep_paths, data_dir, output_path, stats_path, num_cores):
    measured = {}
    for path in sweep_paths:
        for row in _read_rows(path):
            if int(row["num_cores"]) != num_cores:
                raise ValueError(f"{path} contains a different core count")
            measured[(row["case"], row["rule"])] = row
    by_case = defaultdict(list)
    for (case, _), row in measured.items():
        by_case[case].append(row)

    output_rows = []
    case_stats = []
    for case in sorted(by_case):
        graph = json.loads(
            (Path(data_dir) / f"{case}.json").read_text(encoding="utf-8")
        )
        candidate_by_rule = {
            candidate["rule"]: candidate
            for candidate in generate_candidates(graph, num_cores, target_blocks=num_cores)
        }
        rows = by_case[case]
        successful = [row for row in rows if not row.get("error") and row.get("makespan")]
        if not successful:
            continue
        full_winner = _best_official(successful)
        ranked = sorted(
            successful,
            key=lambda row: (
                candidate_by_rule[row["rule"]]["estimated_score"], row["rule"]
            ),
        )
        top2 = ranked[:2]
        top2_winner = _best_official(top2)
        baseline = next((row for row in successful if row["rule"] == "single"), None)
        top2_plus_single = list(top2)
        if baseline and all(row["rule"] != "single" for row in top2_plus_single):
            top2_plus_single.append(baseline)
        top2_plus_single_winner = _best_official(top2_plus_single)
        top2_match = int(top2_winner["makespan"]) == int(full_winner["makespan"])
        top2_single_match = (
            int(top2_plus_single_winner["makespan"]) == int(full_winner["makespan"])
        )
        rank_by_rule = {row["rule"]: index + 1 for index, row in enumerate(ranked)}
        for row in rows:
            candidate = candidate_by_rule[row["rule"]]
            output_rows.append({
                "case": case,
                "num_cores": num_cores,
                "graph_class": row["graph_class"],
                "rule": row["rule"],
                "num_blocks": row["num_blocks"],
                "estimated_score": candidate["estimated_score"],
                "estimated_partition_added_copy_bytes": candidate[
                    "estimated_partition_added_copy_bytes"
                ],
                "official_makespan": row["makespan"],
                "speedup": row["speedup"],
                "official_error": row["error"],
                "estimate_rank": rank_by_rule.get(row["rule"], ""),
                "in_estimate_top2": row["rule"] in {item["rule"] for item in top2},
                "full_sweep_winner": row["rule"] == full_winner["rule"],
                "top2_winner": row["rule"] == top2_winner["rule"],
                "top2_matches_full_sweep": top2_match,
                "top2_plus_single_winner": row["rule"] == top2_plus_single_winner["rule"],
                "top2_plus_single_matches_full_sweep": top2_single_match,
            })
        case_stats.append({
            "case": case,
            "graph_class": rows[0]["graph_class"],
            "full_sweep_best_rule": full_winner["rule"],
            "top2_best_rule": top2_winner["rule"],
            "top2_plus_single_best_rule": top2_plus_single_winner["rule"],
            "full_sweep_speedup": float(full_winner["speedup"]),
            "top2_speedup": float(top2_winner["speedup"]),
            "top2_plus_single_speedup": float(top2_plus_single_winner["speedup"]),
            "top2_matches_full_sweep": top2_match,
            "top2_plus_single_matches_full_sweep": top2_single_match,
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda row: (row["case"], row["rule"])))

    groups = defaultdict(list)
    for row in case_stats:
        groups[row["graph_class"]].append(row)
    def summarize(rows):
        n = len(rows)
        return {
            "cases": n,
            "top2_matches": sum(row["top2_matches_full_sweep"] for row in rows),
            "top2_plus_single_matches": sum(
                row["top2_plus_single_matches_full_sweep"] for row in rows
            ),
            "full_sweep_mean_speedup": sum(row["full_sweep_speedup"] for row in rows) / n,
            "top2_mean_speedup": sum(row["top2_speedup"] for row in rows) / n,
            "top2_plus_single_mean_speedup": sum(
                row["top2_plus_single_speedup"] for row in rows
            ) / n,
        }
    stats = {
        "num_cores": num_cores,
        "total_cases": len(case_stats),
        "overall": summarize(case_stats),
        "by_graph_class": {
            graph_class: summarize(rows) for graph_class, rows in sorted(groups.items())
        },
    }
    Path(stats_path).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--num-cores", type=int, default=4)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(
        args.sweep, args.data_dir, args.output, args.stats, args.num_cores,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
