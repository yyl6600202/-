"""Hybrid genetic and simulated-annealing search for 5-core chain plans.

The chromosome never assigns individual operations.  It assigns contiguous
root groups to cores, so every decoded plan remains a valid acyclic Task
partition.  A cheap traffic/load score ranks the population; only the best
distinct chromosomes are sent to the official event simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from chain_cluster_partition import (
    _execution_order,
    _make_plan,
    _roots_and_lineage,
    _split_contiguous,
)
from core_assignment import estimate_partition_copy_bytes
from evaluation_validation import read_evaluation_config
from graph_features import build_op_dag
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from stub_multicore_cut_and_schedule import validate_multicore_plan


NUM_CORES = 5
ORDER_MODES = ("orig", "id", "load_desc")
DOMINANTS = ("bytes", "id")
ASSIGNMENT_MODES = ("cyclic", "snake", "reverse", "random")


def _root_data(graph_data, order, dominant, order_mode):
    lineage, roots, branch_cycles, _ = _roots_and_lineage(graph_data, order, dominant)
    if order_mode == "id":
        roots = sorted(roots)
    elif order_mode == "load_desc":
        roots = sorted(roots, key=lambda root: (-branch_cycles[root], root))
    return lineage, roots, branch_cycles


def _initial_pattern(group_count, mode, rng):
    if mode == "cyclic":
        return [index % NUM_CORES for index in range(group_count)]
    if mode == "reverse":
        return [(NUM_CORES - 1 - index % NUM_CORES) for index in range(group_count)]
    if mode == "snake":
        path = list(range(NUM_CORES)) + list(range(NUM_CORES - 2, 0, -1))
        return [path[index % len(path)] for index in range(group_count)]
    pattern = [rng.randrange(NUM_CORES) for _ in range(group_count)]
    if group_count >= NUM_CORES:
        pattern[:NUM_CORES] = list(range(NUM_CORES))
    return pattern


def _chromosome(rng, root_count, seed=None):
    if seed is None:
        group_count = rng.randint(2, min(32, root_count))
        assignment = rng.choice(ASSIGNMENT_MODES)
        pattern = _initial_pattern(group_count, assignment, rng)
    else:
        group_count = seed["group_count"]
        pattern = list(seed["pattern"])
    return {
        "dominant": rng.choice(DOMINANTS),
        "order_mode": rng.choice(ORDER_MODES),
        "group_count": group_count,
        "weighted": bool(rng.randrange(2)),
        "pattern": pattern,
    }


def _normalize(chromosome, root_count, rng):
    result = dict(chromosome)
    max_groups = max(1, min(32, root_count))
    result["group_count"] = max(1, min(int(result["group_count"]), max_groups))
    result["pattern"] = [int(core) % NUM_CORES for core in result["pattern"][:result["group_count"]]]
    while len(result["pattern"]) < result["group_count"]:
        result["pattern"].append(rng.randrange(NUM_CORES))
    result["dominant"] = result["dominant"] if result["dominant"] in DOMINANTS else "bytes"
    result["order_mode"] = result["order_mode"] if result["order_mode"] in ORDER_MODES else "orig"
    result["weighted"] = bool(result["weighted"])
    return result


def _decode(chromosome, graph_data, order):
    root_count = sum(1 for node_id in graph_data["noncopy"]["nodes"]
                     if not graph_data["noncopy"]["predecessors"][node_id])
    rng = random.Random(root_count + chromosome["group_count"])
    chromosome = _normalize(chromosome, root_count, rng)
    lineage, roots, branch_cycles = _root_data(
        graph_data, order, chromosome["dominant"], chromosome["order_mode"])
    groups = _split_contiguous(
        roots,
        [branch_cycles[root] for root in roots],
        chromosome["group_count"],
        weighted=chromosome["weighted"],
    )
    root_to_core = {
        root: chromosome["pattern"][group_index % len(chromosome["pattern"])]
        for group_index, group in enumerate(groups)
        for root in group
    }
    plan, blocks = _make_plan(order, lineage, root_to_core, NUM_CORES)
    return chromosome, plan, blocks


def _surrogate(graph_data, plan, blocks, bandwidth):
    op_by_id = graph_data["op_by_id"]
    mapping = {int(node_id): subgraph_id
               for node_id, subgraph_id in plan["node_to_subgraph"].items()}
    core_by_task = {
        subgraph_id: core_id
        for core_id, schedule in enumerate(plan["core_schedules"])
        for subgraph_id in schedule
    }
    pipe_loads = [{"PIPE_MTE2": 0, "PIPE_MTE3": 0, "PIPE_M": 0, "PIPE_V": 0}
                  for _ in range(NUM_CORES)]
    for node_id, task_id in mapping.items():
        pipe = op_by_id[node_id]["pipe"]
        pipe_loads[core_by_task[task_id]][pipe] += op_by_id[node_id]["cycles"]
    max_pipe = max((max(load.values(), default=0) for load in pipe_loads), default=0)
    traffic = estimate_partition_copy_bytes(graph_data, mapping)
    cross_edges = 0
    for source, target in graph_data["noncopy"]["edges"]:
        source_task = mapping[source]
        target_task = mapping[target]
        if source_task != target_task and core_by_task[source_task] != core_by_task[target_task]:
            cross_edges += 1
    # The first two terms match the model note; the last terms discourage
    # excessive Task boundaries before the official simulator is called.
    score = (max_pipe + traffic["partition_added_copy_bytes"] / max(1, bandwidth)
             + 1000 * cross_edges + 100 * len(blocks))
    return score


def _fingerprint(plan):
    return json.dumps(plan, sort_keys=True, separators=(",", ":"))


def _mutate(parent, root_count, rng):
    child = dict(parent)
    child["pattern"] = list(parent["pattern"])
    choice = rng.randrange(6)
    if choice == 0:
        child["group_count"] += rng.choice((-4, -2, -1, 1, 2, 4))
        child["pattern"] = child["pattern"][:child["group_count"]]
    elif choice == 1 and child["pattern"]:
        index = rng.randrange(len(child["pattern"]))
        child["pattern"][index] = rng.randrange(NUM_CORES)
    elif choice == 2 and len(child["pattern"]) > 1:
        left = rng.randrange(len(child["pattern"]) - 1)
        child["pattern"][left], child["pattern"][left + 1] = (
            child["pattern"][left + 1], child["pattern"][left])
    elif choice == 3:
        child["weighted"] = not child["weighted"]
    elif choice == 4:
        child["dominant"] = "id" if child["dominant"] == "bytes" else "bytes"
    else:
        child["order_mode"] = rng.choice(ORDER_MODES)
    return _normalize(child, root_count, rng)


def _crossover(left, right, root_count, rng):
    child = dict(left)
    child["pattern"] = list(left["pattern"])
    if left["group_count"] == right["group_count"] and len(child["pattern"]) > 1:
        cut = rng.randrange(1, len(child["pattern"]))
        child["pattern"] = child["pattern"][:cut] + right["pattern"][cut:]
    else:
        child["group_count"] = rng.choice((left["group_count"], right["group_count"]))
    if rng.random() < 0.5:
        child["dominant"] = right["dominant"]
    if rng.random() < 0.5:
        child["order_mode"] = right["order_mode"]
    if rng.random() < 0.5:
        child["weighted"] = right["weighted"]
    return _normalize(child, root_count, rng)


def _candidate_pool(graph_data, order, seed, population=24, generations=10,
                    anneal_steps=80, rng_seed=20260925):
    rng = random.Random(rng_seed)
    root_count = sum(1 for node_id in graph_data["noncopy"]["nodes"]
                     if not graph_data["noncopy"]["predecessors"][node_id])
    population_items = []
    seeds = [
        {"dominant": "bytes", "order_mode": "orig", "group_count": 5,
         "weighted": False, "pattern": _initial_pattern(5, "cyclic", rng)},
        {"dominant": "bytes", "order_mode": "orig", "group_count": 5,
         "weighted": True, "pattern": _initial_pattern(5, "cyclic", rng)},
        {"dominant": "id", "order_mode": "orig", "group_count": 20,
         "weighted": False, "pattern": _initial_pattern(20, "cyclic", rng)},
        {"dominant": "id", "order_mode": "orig", "group_count": 20,
         "weighted": True, "pattern": _initial_pattern(20, "cyclic", rng)},
        {"dominant": "id", "order_mode": "orig", "group_count": 32,
         "weighted": True, "pattern": _initial_pattern(32, "cyclic", rng)},
    ]
    for seed_item in seeds:
        population_items.append(_normalize(seed_item, root_count, rng))
    while len(population_items) < population:
        population_items.append(_chromosome(rng, root_count))

    scored = {}
    def score(chromosome):
        chromosome, plan, blocks = _decode(chromosome, graph_data, order)
        key = _fingerprint(plan)
        if key not in scored:
            scored[key] = (_surrogate(graph_data, plan, blocks, 60), chromosome, plan, blocks)
        return scored[key]

    for _ in range(generations):
        ranked = sorted((score(item) for item in population_items), key=lambda item: item[0])
        elites = [item[1] for item in ranked[:max(4, population // 5)]]
        next_population = list(elites)
        while len(next_population) < population:
            left = rng.choice(elites)
            right = rng.choice(elites)
            child = _crossover(left, right, root_count, rng)
            if rng.random() < 0.75:
                child = _mutate(child, root_count, rng)
            next_population.append(child)
        population_items = next_population

    # Simulated annealing starts from the strongest genetic chromosomes.
    ranked = sorted((score(item) for item in population_items), key=lambda item: item[0])
    for item in ranked[:max(4, population // 5)]:
        current = item[1]
        current_score = item[0]
        temperature = max(1.0, current_score * 0.08)
        for step in range(anneal_steps):
            proposal = _mutate(current, root_count, rng)
            proposal_score = score(proposal)[0]
            delta = proposal_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9)):
                current, current_score = proposal, proposal_score
            temperature *= 0.96

    return sorted(scored.values(), key=lambda item: item[0])


def search_case(case, data_dir, current_row, plans_dir, official_budget=20,
                population=24, generations=10, anneal_steps=80):
    data_dir = Path(data_dir)
    graph = json.loads((data_dir / f"{case}.json").read_text(encoding="utf-8"))
    config_path = data_dir / "config.txt"
    config = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)
    graph_data = build_op_dag(graph)
    order = _execution_order(graph_data)
    candidates = _candidate_pool(
        graph_data, order, current_row, population, generations, anneal_steps,
        rng_seed=sum(ord(char) for char in case),
    )
    evaluated = []
    seen = set()
    for _, chromosome, plan, blocks in candidates:
        key = _fingerprint(plan)
        if key in seen:
            continue
        seen.add(key)
        try:
            validate_multicore_plan(graph, plan)
            result = evaluate_scene_a(
                graph, plan, config["bandwidth"], config["capacity"],
                waits["task_cross_core_wait_cycles"], waits["task_same_core_wait_cycles"],
            )
        except Exception:
            continue
        movement = result.get("data_movement_bytes", {})
        evaluated.append({
            "rule": "hybrid_" + chromosome["dominant"] + "_" + chromosome["order_mode"]
                    + "_" + str(chromosome["group_count"]),
            "makespan": result["makespan"],
            "added_copy_bytes": movement.get("added_copy_bytes", 0),
            "num_blocks": len(blocks),
            "plan": plan,
            "chromosome": chromosome,
        })
        if len(evaluated) >= official_budget:
            break
    if not evaluated:
        return {"case": case, "improved": False, "error": "no official candidate"}
    best = min(evaluated, key=lambda item: (
        item["makespan"], item["added_copy_bytes"], item["num_blocks"], item["rule"],
    ))
    current_makespan = int(float(current_row["problem1_makespan"]))
    improved = best["makespan"] < current_makespan
    if improved:
        output = Path(plans_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{case}.json").write_text(
            json.dumps(best["plan"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "case": case,
        "improved": improved,
        "current_makespan": current_makespan,
        "new_makespan": best["makespan"],
        "new_speedup": float(current_row["baseline_makespan"]) / best["makespan"],
        "rule": best["rule"],
        "added_copy_bytes": best["added_copy_bytes"],
        "num_blocks": best["num_blocks"],
        "evaluated_candidates": len(evaluated),
        "error": "",
        "plan": best["plan"] if improved else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--current-summary", type=Path,
                        default=Path("day5_results/chain_generator_5core_v4_summary.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("day5_results/hybrid_search_delta.csv"))
    parser.add_argument("--plans-dir", type=Path,
                        default=Path("day5_results/hybrid_search_plans"))
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--official-budget", type=int, default=20)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--anneal-steps", type=int, default=80)
    args = parser.parse_args(argv)
    with args.current_summary.open(newline="", encoding="utf-8-sig") as handle:
        current = {row["case"]: row for row in csv.DictReader(handle)}
    cases = args.cases or [
        row["case"] for row in sorted(
            current.values(), key=lambda row: float(row["speedup"]))[:args.worst]
    ]
    fields = ["case", "improved", "current_makespan", "new_makespan", "new_speedup",
              "rule", "added_copy_bytes", "num_blocks", "evaluated_candidates", "error"]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, 1):
        try:
            result = search_case(
                case, args.data_dir, current[case], args.plans_dir,
                args.official_budget, args.population, args.generations,
                args.anneal_steps,
            )
        except Exception as exc:
            result = {"case": case, "improved": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        row = {field: result.get(field, "") for field in fields}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"progress {index}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()
