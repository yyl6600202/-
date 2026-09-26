"""Bounded local search for problem-1 plans.

The search deliberately treats the official evaluator as the objective.  Cheap
candidate generation is used only to propose mutations; every accepted result
is backed by an evaluator result and a structural validation pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict

from evaluation_validation import read_evaluation_config
from graph_features import build_op_dag
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from stub_multicore_cut_and_schedule import validate_multicore_plan


def _canonical_plan(plan):
    return {
        "node_to_subgraph": {
            str(node_id): int(subgraph_id)
            for node_id, subgraph_id in sorted(
                ((int(node_id), subgraph_id) for node_id, subgraph_id in plan["node_to_subgraph"].items()),
                key=lambda item: item[0],
            )
        },
        "core_schedules": [list(map(int, order)) for order in plan["core_schedules"]],
    }


def plan_key(plan):
    payload = json.dumps(_canonical_plan(plan), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def objective(result):
    movement = result.get("data_movement_bytes", {})
    return (
        result.get("makespan", float("inf")),
        movement.get("added_copy_bytes", float("inf")),
        movement.get("spill_added_copy_bytes", float("inf")),
    )


class OfficialProblem1Cache:
    """Validate and evaluate plans once, retaining compact and full results."""

    def __init__(self, graph, config_path):
        self.graph = graph
        config = read_evaluation_config(config_path)
        waits = read_scene_a_config(config_path)
        self.bandwidth = config["bandwidth"]
        self.capacity = config["capacity"]
        self.cross_core_wait = waits["task_cross_core_wait_cycles"]
        self.same_core_wait = waits["task_same_core_wait_cycles"]
        self.cache = {}

    def evaluate(self, plan):
        canonical = _canonical_plan(plan)
        key = plan_key(canonical)
        if key in self.cache:
            return self.cache[key]
        try:
            validate_multicore_plan(self.graph, canonical)
            result = evaluate_scene_a(
                self.graph,
                canonical,
                self.bandwidth,
                self.capacity,
                self.cross_core_wait,
                self.same_core_wait,
            )
        except Exception as error:  # invalid candidates remain inspectable
            result = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            result["ok"] = True
            result["objective"] = list(objective(result))
        self.cache[key] = result
        return result


def _subgraph_nodes(graph, plan):
    graph_data = build_op_dag(graph)
    position = {
        node_id: index
        for index, node_id in enumerate(graph_data["noncopy"]["topological_order"])
    }
    groups = defaultdict(list)
    for node_id, subgraph_id in plan["node_to_subgraph"].items():
        groups[int(subgraph_id)].append(int(node_id))
    return [
        {
            "subgraph_id": subgraph_id,
            "node_ids": sorted(node_ids, key=position.__getitem__),
            "position": min(position[node_id] for node_id in node_ids),
        }
        for subgraph_id, node_ids in sorted(
            groups.items(), key=lambda item: min(position[node_id] for node_id in item[1])
        )
    ]


def _copy_plan(plan):
    return {
        "node_to_subgraph": dict(plan["node_to_subgraph"]),
        "core_schedules": [list(order) for order in plan["core_schedules"]],
    }


def _remove_empty_subgraphs(plan):
    occupied = set(plan["node_to_subgraph"].values())
    plan["core_schedules"] = [
        [subgraph_id for subgraph_id in order if subgraph_id in occupied]
        for order in plan["core_schedules"]
    ]
    return plan


def propose_moves(graph, plan, limit=48):
    """Propose boundary moves, core swaps, merges, and splits deterministically."""
    base = _canonical_plan(plan)
    groups = _subgraph_nodes(graph, base)
    proposals = []

    def add(kind, candidate, detail):
        if len(proposals) >= limit:
            return
        proposals.append({"operator": kind, "detail": detail, "plan": candidate})

    # Move one boundary operation across a neighboring block.
    for index in range(len(groups) - 1):
        left, right = groups[index], groups[index + 1]
        for source, target, node in (
            (left, right, left["node_ids"][-1]),
            (right, left, right["node_ids"][0]),
        ):
            if len(source["node_ids"]) <= 1:
                continue
            candidate = _copy_plan(base)
            candidate["node_to_subgraph"][str(node)] = target["subgraph_id"]
            add("move", candidate, {"node_id": node, "source": source["subgraph_id"], "target": target["subgraph_id"]})

    # Swap neighboring block ownership between cores while preserving order.
    core_by_subgraph = {
        subgraph_id: core_id
        for core_id, order in enumerate(base["core_schedules"])
        for subgraph_id in order
    }
    for left, right in zip(groups, groups[1:]):
        left_core = core_by_subgraph[left["subgraph_id"]]
        right_core = core_by_subgraph[right["subgraph_id"]]
        if left_core == right_core:
            continue
        candidate = _copy_plan(base)
        candidate["core_schedules"][left_core].remove(left["subgraph_id"])
        candidate["core_schedules"][right_core].remove(right["subgraph_id"])
        candidate["core_schedules"][left_core].append(right["subgraph_id"])
        candidate["core_schedules"][right_core].append(left["subgraph_id"])
        add("swap", candidate, {"left": left["subgraph_id"], "right": right["subgraph_id"]})

    # Merge adjacent blocks.  The lower id remains the task id.
    for left, right in zip(groups, groups[1:]):
        candidate = _copy_plan(base)
        left_id, right_id = left["subgraph_id"], right["subgraph_id"]
        for node_id, subgraph_id in list(candidate["node_to_subgraph"].items()):
            if subgraph_id == right_id:
                candidate["node_to_subgraph"][node_id] = left_id
        candidate["core_schedules"] = [
            [subgraph_id for subgraph_id in order if subgraph_id != right_id]
            for order in candidate["core_schedules"]
        ]
        candidate = _remove_empty_subgraphs(candidate)
        # A merged task must occur once even if the two source tasks were on
        # different cores; place it on the left task's core.
        left_core = core_by_subgraph[left_id]
        for core_id, order in enumerate(candidate["core_schedules"]):
            if core_id != left_core:
                candidate["core_schedules"][core_id] = [
                    subgraph_id for subgraph_id in order if subgraph_id != left_id
                ]
        if left_id not in candidate["core_schedules"][left_core]:
            candidate["core_schedules"][left_core].append(left_id)
        add("merge", candidate, {"left": left_id, "right": right_id})

    # Split the largest blocks by topological position.
    for group in sorted(groups, key=lambda item: (-len(item["node_ids"]), item["position"]))[:8]:
        if len(group["node_ids"]) < 2:
            continue
        split_at = len(group["node_ids"]) // 2
        new_id = max(base["node_to_subgraph"].values(), default=-1) + 1
        candidate = _copy_plan(base)
        for node_id in group["node_ids"][split_at:]:
            candidate["node_to_subgraph"][str(node_id)] = new_id
        core_id = core_by_subgraph[group["subgraph_id"]]
        order = candidate["core_schedules"][core_id]
        order.insert(order.index(group["subgraph_id"]) + 1, new_id)
        add("split", candidate, {"source": group["subgraph_id"], "new": new_id})

    unique = []
    seen = {plan_key(base)}
    for proposal in proposals:
        key = plan_key(proposal["plan"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(proposal)
    return unique


def search(graph, start_plan, config_path, max_rounds=2, max_evaluations=40, proposal_limit=48):
    """Run bounded best-improvement search and return best plan plus history."""
    evaluator = OfficialProblem1Cache(graph, config_path)
    current = _canonical_plan(start_plan)
    start_result = evaluator.evaluate(current)
    if not start_result.get("ok"):
        raise ValueError(f"start plan is not evaluable: {start_result.get('error')}")
    best_plan, best_result = current, start_result
    history = [{
        "round": 0,
        "operator": "start",
        "objective": list(objective(start_result)),
        "evaluations": 1,
    }]
    evaluations = 1
    for round_id in range(1, max_rounds + 1):
        proposals = propose_moves(graph, current, limit=proposal_limit)
        round_best = None
        round_count = 0
        for proposal in proposals:
            if evaluations >= max_evaluations:
                break
            result = evaluator.evaluate(proposal["plan"])
            evaluations += 1
            round_count += 1
            if not result.get("ok"):
                continue
            if round_best is None or objective(result) < objective(round_best["result"]):
                round_best = {"proposal": proposal, "result": result}
        if round_best is None or objective(round_best["result"]) >= objective(best_result):
            history.append({
                "round": round_id,
                "operator": "stop",
                "objective": list(objective(best_result)),
                "evaluations": round_count,
            })
            break
        current = _canonical_plan(round_best["proposal"]["plan"])
        best_plan, best_result = current, round_best["result"]
        history.append({
            "round": round_id,
            "operator": round_best["proposal"]["operator"],
            "detail": round_best["proposal"]["detail"],
            "objective": list(objective(best_result)),
            "evaluations": round_count,
        })
    return {
        "plan": best_plan,
        "result": best_result,
        "history": history,
        "evaluations": evaluations,
        "cache_entries": len(evaluator.cache),
    }
