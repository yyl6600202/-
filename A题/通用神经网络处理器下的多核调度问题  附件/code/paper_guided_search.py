"""Paper-guided 5-core candidate search for problem 1.

The candidate family combines critical-path-aware topological ordering from
HEFT/CPOP and communication-aware root clustering inspired by CNGA.  Every
candidate is still scored by the official problem-1 event simulator.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import defaultdict
from pathlib import Path

from chain_cluster_partition import _assign_roots, _roots_and_lineage
from evaluation_validation import read_evaluation_config
from graph_features import build_op_dag
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from stub_multicore_cut_and_schedule import validate_multicore_plan


ORDER_RULES = ("heft_up", "cpop", "cnga")
PACK_RULES = ("root_contiguous_weighted", "root_roundrobin_weighted", "root_load")


def _edge_bytes(graph_data):
    noncopy = set(graph_data["noncopy"]["nodes"])
    result = defaultdict(int)
    for tensor_id, source in graph_data["producer"].items():
        if source not in noncopy:
            continue
        size = graph_data["tensor_by_id"][tensor_id]["size"]
        for target in graph_data["consumers"].get(tensor_id, ()):
            if target in noncopy:
                result[source, target] += size
    return dict(result)


def _ranks(graph_data, num_cores, bandwidth, cross_wait):
    dag = graph_data["noncopy"]
    op_by_id = graph_data["op_by_id"]
    traffic = _edge_bytes(graph_data)
    remote_probability = (num_cores - 1) / max(1, num_cores)
    avg_comm = {
        edge: remote_probability * (cross_wait + size / max(1, bandwidth))
        for edge, size in traffic.items()
    }
    avg_comm.setdefault((0, 0), 0.0)

    up, down, comm_tail, best_tail = {}, {}, {}, {}
    for node in reversed(dag["topological_order"]):
        duration = max(1, op_by_id[node]["cycles"])
        succs = dag["successors"][node]
        if succs:
            successor = max(
                succs,
                key=lambda child: (avg_comm.get((node, child), 0.0) + up[child], -child),
            )
            edge_comm = avg_comm.get((node, successor), 0.0)
            up[node] = duration + edge_comm + up[successor]
            comm_tail[node] = edge_comm + comm_tail[successor]
            best_tail[node] = successor
        else:
            up[node] = float(duration)
            comm_tail[node] = 0.0
            best_tail[node] = None

    for node in dag["topological_order"]:
        duration = max(1, op_by_id[node]["cycles"])
        preds = dag["predecessors"][node]
        down[node] = duration + max(
            (avg_comm.get((pred, node), 0.0) + down[pred] for pred in preds),
            default=0.0,
        )
    return up, down, comm_tail, avg_comm, traffic


def _priority_order(graph_data, rule, num_cores, bandwidth, cross_wait):
    dag = graph_data["noncopy"]
    op_by_id = graph_data["op_by_id"]
    up, down, comm_tail, _, _ = _ranks(graph_data, num_cores, bandwidth, cross_wait)
    priorities = {}
    for node in dag["nodes"]:
        if rule == "heft_up":
            priorities[node] = up[node]
        elif rule == "cpop":
            priorities[node] = up[node] + down[node]
        elif rule == "cnga":
            # CNGA ranks the local critical path highly while discounting the
            # communication share, so compute-dense critical work becomes ready first.
            priorities[node] = up[node] - comm_tail[node] / max(1.0, up[node])
        else:
            raise ValueError(f"unknown priority rule: {rule}")

    indegree = {node: len(dag["predecessors"][node]) for node in dag["nodes"]}
    ready = [(-priorities[node], -op_by_id[node]["cycles"], node)
             for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, _, node = heapq.heappop(ready)
        order.append(node)
        for child in sorted(dag["successors"][node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (-priorities[child], -op_by_id[child]["cycles"], child))
    if len(order) != len(dag["nodes"]):
        raise ValueError("priority topological sort did not visit every operation")
    return order


def _smooth_blocks(order, lineage, root_to_core, op_by_id, num_cores, max_blocks=32):
    """Merge short adjacent runs to cap Task-boundary overhead.

    Every run stays an interval of a topological order, so contraction cannot
    introduce a dependency cycle.  When a run is absorbed, choose the adjacent
    core that leaves the per-Pipe work distribution most balanced.
    """
    blocks = []
    for node in order:
        core = root_to_core[lineage[node]]
        if blocks and blocks[-1]["core"] == core:
            blocks[-1]["nodes"].append(node)
        else:
            blocks.append({"core": core, "nodes": [node]})
    if len(blocks) <= max_blocks:
        return blocks

    pipes = sorted({op_by_id[node]["pipe"] for node in order})
    loads = [{pipe: 0 for pipe in pipes} for _ in range(num_cores)]
    for block in blocks:
        block_load = defaultdict(int)
        for node in block["nodes"]:
            block_load[op_by_id[node]["pipe"]] += op_by_id[node]["cycles"]
        block["pipe_load"] = dict(block_load)
        for pipe, value in block["pipe_load"].items():
            loads[block["core"]][pipe] += value

    def absorb(index, target):
        block = blocks[index]
        source = block["core"]
        if source != target:
            for pipe, value in block["pipe_load"].items():
                loads[source][pipe] -= value
                loads[target][pipe] += value
            block["core"] = target
        if index > 0 and blocks[index - 1]["core"] == block["core"]:
            left = blocks[index - 1]
            left["nodes"].extend(block["nodes"])
            for pipe, value in block["pipe_load"].items():
                left["pipe_load"][pipe] = left["pipe_load"].get(pipe, 0) + value
            blocks.pop(index)
            index -= 1
        if index + 1 < len(blocks) and blocks[index + 1]["core"] == blocks[index]["core"]:
            right = blocks.pop(index + 1)
            blocks[index]["nodes"].extend(right["nodes"])
            for pipe, value in right["pipe_load"].items():
                blocks[index]["pipe_load"][pipe] = blocks[index]["pipe_load"].get(pipe, 0) + value

    while len(blocks) > max_blocks:
        index = min(
            range(len(blocks)),
            key=lambda i: (
                sum(blocks[i]["pipe_load"].values()),
                len(blocks[i]["nodes"]),
                i,
            ),
        )
        targets = []
        if index > 0:
            targets.append(blocks[index - 1]["core"])
        if index + 1 < len(blocks):
            targets.append(blocks[index + 1]["core"])
        targets = list(dict.fromkeys(targets))
        if not targets:
            break

        total_by_pipe = {
            pipe: sum(load[pipe] for load in loads)
            for pipe in pipes
        }

        def balance_score(target):
            maximum = 0.0
            for core in range(num_cores):
                for pipe in pipes:
                    load = loads[core][pipe]
                    if core == target:
                        load += blocks[index]["pipe_load"].get(pipe, 0)
                    scale = max(1.0, total_by_pipe[pipe] / num_cores)
                    maximum = max(maximum, load / scale)
            return (maximum, target)

        absorb(index, min(targets, key=balance_score))
    return blocks


def _plan_from_blocks(blocks, num_cores):
    mapping = {}
    core_schedules = [[] for _ in range(num_cores)]
    for subgraph_id, block in enumerate(blocks):
        for node in block["nodes"]:
            mapping[str(node)] = subgraph_id
        core_schedules[block["core"]].append(subgraph_id)
    return {"node_to_subgraph": mapping, "core_schedules": core_schedules}


def generate_candidates(graph, num_cores, bandwidth, cross_wait,
                        order_rules=ORDER_RULES, pack_rules=PACK_RULES):
    """Generate validated critical-path/communication-aware partitions."""
    graph_data = build_op_dag(graph)
    candidates = []
    fingerprints = set()
    for order_rule in order_rules:
        order = _priority_order(graph_data, order_rule, num_cores, bandwidth, cross_wait)
        lineage, roots, branch_cycles, branch_nodes = _roots_and_lineage(
            graph_data, order, dominant="bytes")
        for pack_rule in pack_rules:
            root_to_core = _assign_roots(roots, branch_cycles, num_cores, pack_rule)
            blocks = _smooth_blocks(
                order, lineage, root_to_core, graph_data["op_by_id"], num_cores)
            plan = _plan_from_blocks(blocks, num_cores)
            fingerprint = json.dumps(plan, sort_keys=True, separators=(",", ":"))
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            valid, error = True, ""
            try:
                validate_multicore_plan(graph, plan)
            except Exception as exc:  # candidate rejection is recorded for review
                valid, error = False, f"{type(exc).__name__}: {exc}"
            candidates.append({
                "rule": f"paper_{order_rule}_{pack_rule}",
                "plan": plan,
                "blocks": blocks,
                "num_blocks": len(blocks),
                "num_roots": len(roots),
                "branch_cycles": branch_cycles,
                "branch_sizes": {str(root): len(branch_nodes[root]) for root in roots},
                "valid": valid,
                "validation_error": error,
            })
    return candidates


def _evaluate(graph, plan, config, waits):
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


def run_sweep(data_dir, summary_path, output_path, plans_dir, max_speedup=3.8,
              cases=None, resume=False, num_cores=5):
    data_dir, summary_path = Path(data_dir), Path(summary_path)
    output_path, plans_dir = Path(output_path), Path(plans_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open(newline="", encoding="utf-8-sig") as stream:
        base_rows = {row["case"]: row for row in csv.DictReader(stream)}
    if cases:
        selected = list(cases)
    else:
        eligible = [name for name, row in base_rows.items()
                    if float(row["speedup"]) < max_speedup]
        selected = sorted(
            eligible,
            key=lambda name: (
                len(json.loads((data_dir / f"{name}.json").read_text(encoding="utf-8"))["ops"]),
                name,
            ),
        )

    fields = ["case", "base_makespan", "candidate_makespan", "candidate_speedup",
              "improved", "rule", "added_copy_bytes", "num_blocks",
              "candidate_count", "rejected_count", "error"]
    prior_rows = []
    completed = set()
    if resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8-sig") as stream:
            prior_rows = list(csv.DictReader(stream))
        completed = {row["case"] for row in prior_rows if not row.get("error")}
    rows = [row for row in prior_rows if row["case"] not in selected]
    if resume:
        rows = prior_rows[:]

    config = read_evaluation_config(data_dir / "config.txt")
    waits = read_scene_a_config(data_dir / "config.txt")
    for case_name in selected:
        if case_name in completed:
            continue
        base = base_rows[case_name]
        incumbent = float(base["problem1_makespan"])
        try:
            graph = json.loads((data_dir / f"{case_name}.json").read_text(encoding="utf-8"))
            candidates = generate_candidates(
                graph, num_cores, config["bandwidth"],
                waits["task_cross_core_wait_cycles"],
            )
            evaluated, seen = [], set()
            errors = []
            for candidate in candidates:
                if not candidate["valid"]:
                    errors.append(candidate["validation_error"])
                    continue
                fingerprint = json.dumps(candidate["plan"], sort_keys=True, separators=(",", ":"))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                try:
                    result = _evaluate(graph, candidate["plan"], config, waits)
                    evaluated.append((candidate["rule"], result, candidate["num_blocks"]))
                except Exception as exc:
                    errors.append(f"{candidate['rule']}: {type(exc).__name__}: {exc}")
            if evaluated:
                rule, best, num_blocks = min(
                    evaluated,
                    key=lambda item: (item[1]["makespan"], item[1]["added_copy_bytes"], item[0]),
                )
                candidate_makespan = best["makespan"]
                improved = candidate_makespan < incumbent
                if improved:
                    (plans_dir / f"{case_name}.json").write_text(
                        json.dumps(best["plan"], ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                row = {
                    "case": case_name,
                    "base_makespan": int(incumbent),
                    "candidate_makespan": candidate_makespan,
                    "candidate_speedup": float(base["baseline_makespan"]) / candidate_makespan,
                    "improved": improved,
                    "rule": rule,
                    "added_copy_bytes": best["added_copy_bytes"],
                    "num_blocks": num_blocks,
                    "candidate_count": len(evaluated),
                    "rejected_count": len(candidates) - len(evaluated),
                    "error": "",
                }
            else:
                row = {field: "" for field in fields}
                row.update(case=case_name, base_makespan=int(incumbent),
                           candidate_count=0, rejected_count=len(candidates),
                           error="; ".join(errors[:3]) or "all candidates were duplicates/rejected")
        except Exception as exc:
            row = {field: "" for field in fields}
            row.update(case=case_name, base_makespan=int(incumbent),
                       error=f"{type(exc).__name__}: {exc}")
        rows = [old for old in rows if old["case"] != case_name]
        rows.append(row)
        rows.sort(key=lambda item: item["case"])
        with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--summary", type=Path,
                        default=Path("day5_results/chain_generator_5core_v5_summary.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("day5_results/paper_guided_delta.csv"))
    parser.add_argument("--plans-dir", type=Path,
                        default=Path("day5_results/paper_guided_plans"))
    parser.add_argument("--max-speedup", type=float, default=3.8)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-cores", type=int, default=5)
    args = parser.parse_args(argv)
    run_sweep(args.data_dir, args.summary, args.output, args.plans_dir,
              args.max_speedup, args.cases, args.resume, args.num_cores)


if __name__ == "__main__":
    main()
