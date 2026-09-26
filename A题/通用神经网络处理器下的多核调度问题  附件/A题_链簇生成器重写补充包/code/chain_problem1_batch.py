"""Compare the chain generator with the saved 5-core incumbent plans."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from chain_cluster_partition import DEFAULT_MODES, generate_chain_candidates
from evaluation_validation import read_evaluation_config
from initial_partition import generate_candidates
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config


def _case_names(data_dir):
    return sorted(path.stem for path in Path(data_dir).glob("case_*.json")
                  if re.fullmatch(r"case_\d{3}", path.stem))


def _official_result(graph, plan, config, waits):
    result = evaluate_scene_a(
        graph, plan, config["bandwidth"], config["capacity"],
        waits["task_cross_core_wait_cycles"],
        waits["task_same_core_wait_cycles"],
    )
    movement = result.get("data_movement_bytes", {})
    return {
        "makespan": result["makespan"],
        "added_copy_bytes": movement.get("added_copy_bytes", 0),
        "spill_added_copy_bytes": movement.get("spill_added_copy_bytes", 0),
        "plan": plan,
    }


def _load_incumbent(case_name, incumbent_dirs):
    for directory in incumbent_dirs:
        path = Path(directory) / f"{case_name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), str(path)
    return None, ""


def evaluate_case(case_name, data_dir, incumbent_dirs, num_cores=5,
                  modes=DEFAULT_MODES, dominants=("bytes", "cycles", "id"),
                  plans_dir=None, large_graph_ops=10000):
    data_dir = Path(data_dir)
    graph = json.loads((data_dir / f"{case_name}.json").read_text(encoding="utf-8"))
    config_path = data_dir / "config.txt"
    config = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)

    candidates = {}
    # Keep single-core as an authoritative reference and safe fallback.
    single = generate_candidates(graph, num_cores, target_blocks=1, rules=("single",))[0]
    candidates["single"] = single["plan"]

    incumbent, incumbent_path = _load_incumbent(case_name, incumbent_dirs)
    if incumbent is not None:
        candidates["incumbent"] = incumbent

    # Large graphs make official event simulation expensive.  Keep the two
    # contiguous root packings, which were the productive rules in the pilot.
    case_modes = modes
    case_dominants = dominants
    if len(graph.get("ops", ())) > large_graph_ops:
        case_modes = ("root_contiguous", "root_contiguous_weighted")
        case_dominants = ("bytes",)
    for candidate in generate_chain_candidates(
            graph, num_cores, config["bandwidth"], case_modes, case_dominants):
        if candidate["valid"]:
            candidates[candidate["rule"]] = candidate["plan"]

    evaluated = {}
    fingerprints = {}
    for name, plan in candidates.items():
        fingerprint = json.dumps(plan, sort_keys=True, separators=(",", ":"))
        if fingerprint in fingerprints:
            evaluated[name] = evaluated[fingerprints[fingerprint]]
            continue
        try:
            evaluated[name] = _official_result(graph, plan, config, waits)
            fingerprints[fingerprint] = name
        except Exception as exc:
            evaluated[name] = {"error": f"{type(exc).__name__}: {exc}"}

    successful = {name: result for name, result in evaluated.items()
                  if "makespan" in result}
    if "single" not in successful:
        raise RuntimeError(f"{case_name}: official evaluator rejected single baseline")
    baseline = successful["single"]["makespan"]
    winner_name, winner = min(
        successful.items(),
        key=lambda item: (item[1]["makespan"], item[1]["added_copy_bytes"],
                          item[0] != "incumbent", item[0]),
    )
    if plans_dir is not None:
        target = Path(plans_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{case_name}.json").write_text(
            json.dumps(winner["plan"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    chain_success = [(name, result) for name, result in successful.items()
                     if name.startswith("chain_")]
    best_chain_name, best_chain = min(
        chain_success,
        key=lambda item: (item[1]["makespan"], item[1]["added_copy_bytes"], item[0]),
        default=("", {}),
    )
    blocks = len(set(winner["plan"]["node_to_subgraph"].values()))
    return {
        "case": case_name,
        "num_cores": num_cores,
        "winner": winner_name,
        "baseline_makespan": baseline,
        "incumbent_makespan": successful.get("incumbent", {}).get("makespan", ""),
        "best_chain_rule": best_chain_name,
        "best_chain_makespan": best_chain.get("makespan", ""),
        "best_chain_added_copy_bytes": best_chain.get("added_copy_bytes", ""),
        "problem1_makespan": winner["makespan"],
        "speedup": baseline / winner["makespan"] if winner["makespan"] else None,
        "added_copy_bytes": winner["added_copy_bytes"],
        "num_blocks": blocks,
        "evaluated_candidates": len(successful),
        "rejected_candidates": len(evaluated) - len(successful),
        "incumbent_plan": incumbent_path,
        "error": "",
    }


def run(data_dir, output, plans_dir, cases=None, resume=False, num_cores=5,
        modes=DEFAULT_MODES, dominants=("bytes", "cycles", "id"), workers=1,
        incumbent_dirs=(), screen_summary=None, skip_speedup_at_least=None,
        large_graph_ops=10000):
    del workers  # evaluator remains sequential to cap memory and log noise
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = list(cases) if cases else _case_names(data_dir)
    screen_rows = {}
    if screen_summary is not None and Path(screen_summary).exists():
        with Path(screen_summary).open(newline="", encoding="utf-8-sig") as handle:
            screen_rows = {row["case"]: row for row in csv.DictReader(handle)}
    rows = []
    if resume and output.exists():
        with output.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        completed = {row["case"] for row in rows if not row.get("error")}
        selected = [case for case in selected if case not in completed]

    fields = [
        "case", "num_cores", "winner", "baseline_makespan",
        "incumbent_makespan", "best_chain_rule", "best_chain_makespan",
        "best_chain_added_copy_bytes", "problem1_makespan", "speedup",
        "added_copy_bytes", "num_blocks", "evaluated_candidates",
        "rejected_candidates", "incumbent_plan", "error",
    ]
    for case_name in selected:
        prior = screen_rows.get(case_name)
        prior_speedup = None
        if prior:
            try:
                prior_speedup = float(prior["speedup"])
            except (KeyError, TypeError, ValueError):
                pass
        if (skip_speedup_at_least is not None and prior is not None
                and prior_speedup is not None
                and prior_speedup >= skip_speedup_at_least):
            incumbent, incumbent_path = _load_incumbent(case_name, incumbent_dirs)
            if incumbent is None:
                row = {field: "" for field in fields}
                row.update(case=case_name, error="screened incumbent plan not found")
            else:
                row = {
                    "case": case_name,
                    "num_cores": num_cores,
                    "winner": "incumbent_screened_high_speedup",
                    "baseline_makespan": prior.get("baseline_makespan", ""),
                    "incumbent_makespan": prior.get("problem1_makespan", ""),
                    "best_chain_rule": "not_evaluated_existing_speedup_above_threshold",
                    "best_chain_makespan": "",
                    "best_chain_added_copy_bytes": "",
                    "problem1_makespan": prior.get("problem1_makespan", ""),
                    "speedup": prior.get("speedup", ""),
                    "added_copy_bytes": prior.get("problem1_added_copy_bytes", ""),
                    "num_blocks": prior.get("num_blocks", ""),
                    "evaluated_candidates": 0,
                    "rejected_candidates": 0,
                    "incumbent_plan": incumbent_path,
                    "error": "",
                }
                if plans_dir is not None:
                    target = Path(plans_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"{case_name}.json").write_text(
                        json.dumps(incumbent, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            rows = [existing for existing in rows if existing["case"] != case_name]
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with output.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda item: item["case"]))
            continue
        try:
            row = evaluate_case(case_name, data_dir, incumbent_dirs, num_cores,
                                modes, dominants, plans_dir, large_graph_ops)
        except Exception as exc:
            row = {field: "" for field in fields}
            row.update(case=case_name, error=f"{type(exc).__name__}: {exc}")
        rows = [existing for existing in rows if existing["case"] != case_name]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda item: item["case"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("day5_results/chain_generator_summary.csv"))
    parser.add_argument("--plans-dir", type=Path, default=Path("day5_results/chain_generator_plans"))
    parser.add_argument("--incumbent-dir", action="append", dest="incumbent_dirs",
                        default=["day5_results/problem1_5core_optimized_plans",
                                 "day5_results/problem1_5core_plans"])
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-cores", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES,
                        default=list(DEFAULT_MODES))
    parser.add_argument("--dominants", nargs="+", choices=("bytes", "cycles", "id"),
                        default=["bytes", "cycles", "id"])
    parser.add_argument("--screen-summary", type=Path,
                        help="existing official summary used for safe high-speedup screening")
    parser.add_argument("--skip-speedup-at-least", type=float,
                        help="retain incumbent without chain search above this prior speedup")
    parser.add_argument("--large-graph-ops", type=int, default=10000,
                        help="use only byte-dominant contiguous variants above this op count")
    args = parser.parse_args(argv)
    run(args.data_dir, args.output, args.plans_dir, args.cases, args.resume,
        args.num_cores, args.modes, args.dominants,
        incumbent_dirs=args.incumbent_dirs, screen_summary=args.screen_summary,
        skip_speedup_at_least=args.skip_speedup_at_least,
        large_graph_ops=args.large_graph_ops)


if __name__ == "__main__":
    main()
