"""Structured initial partitions for the A题 problem-1 search.

All rules preserve a global topological block order.  This makes the common
continuous/wave/aligned candidates acyclic by construction; every candidate is
still passed through the official plan validator before it is returned.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

from core_assignment import (
    assign_lpt,
    block_statistics,
    estimate_partition_copy_bytes,
    estimate_problem1_score,
)
from graph_features import build_op_dag
from stub_multicore_cut_and_schedule import validate_multicore_plan


def _ordered_blocks(raw_blocks, topo_order, graph_data):
    position = {node_id: index for index, node_id in enumerate(topo_order)}
    raw_blocks = [
        sorted(set(node_ids), key=position.__getitem__)
        for node_ids in raw_blocks
        if node_ids
    ]
    raw_blocks.sort(key=lambda nodes: position[nodes[0]])
    blocks = []
    for subgraph_id, node_ids in enumerate(raw_blocks):
        stats = block_statistics(graph_data, node_ids)
        stats["subgraph_id"] = subgraph_id
        stats["topo_start"] = position[node_ids[0]]
        stats["topo_end"] = position[node_ids[-1]]
        blocks.append(stats)
    return blocks


def _split_sequence(nodes, weights, target_blocks):
    """Split a topological sequence into weighted contiguous blocks."""
    if not nodes:
        return []
    target_blocks = max(1, min(int(target_blocks), len(nodes)))
    if target_blocks == 1:
        return [list(nodes)]
    total = sum(weights)
    result = []
    cursor = 0
    remaining_weight = total
    for block_index in range(target_blocks - 1):
        remaining_blocks = target_blocks - block_index
        desired = remaining_weight / remaining_blocks if remaining_blocks else 0
        start = cursor
        current = 0
        max_cursor = len(nodes) - (remaining_blocks - 1)
        while cursor < max_cursor:
            current += weights[cursor]
            cursor += 1
            if current >= desired and cursor > start:
                break
        result.append(list(nodes[start:cursor]))
        remaining_weight -= current
    result.append(list(nodes[cursor:]))
    return [block for block in result if block]


def _level_groups(dag):
    levels = defaultdict(list)
    level_by_node = {}
    for node_id in dag["topological_order"]:
        predecessor_levels = [level_by_node[pred] for pred in dag["predecessors"][node_id]]
        level = 0 if not predecessor_levels else 1 + max(predecessor_levels)
        level_by_node[node_id] = level
        levels[level].append(node_id)
    return level_by_node, [levels[level] for level in sorted(levels)]


def _continuous_blocks(dag, graph_data, target_blocks):
    order = dag["topological_order"]
    weights = [graph_data["op_by_id"][node_id]["cycles"] for node_id in order]
    return _ordered_blocks(_split_sequence(order, weights, target_blocks), order, graph_data)


def _wave_blocks(dag, graph_data, target_blocks):
    _, levels = _level_groups(dag)
    if not levels:
        return []
    level_weights = [
        sum(graph_data["op_by_id"][node_id]["cycles"] for node_id in level)
        for level in levels
    ]
    wave_indices = _split_sequence(list(range(len(levels))), level_weights, target_blocks)
    raw_blocks = [[node_id for level_index in wave for node_id in levels[level_index]]
                  for wave in wave_indices]
    return _ordered_blocks(raw_blocks, dag["topological_order"], graph_data)


def _aligned_blocks(dag, graph_data, target_blocks):
    """Split broad topological layers while capping deep graphs at 4K blocks."""
    _, levels = _level_groups(dag)
    if not levels:
        return []
    max_total = max(1, int(target_blocks) * 4)
    chunks_per_level = max(1, min(int(target_blocks), max_total // len(levels)))
    raw_blocks = []
    for level in levels:
        weights = [graph_data["op_by_id"][node_id]["cycles"] for node_id in level]
        raw_blocks.extend(_split_sequence(level, weights, chunks_per_level))
    return _ordered_blocks(raw_blocks, dag["topological_order"], graph_data)


def _component_blocks(dag, graph_data):
    nodes = set(dag["nodes"])
    undirected = {node_id: set() for node_id in nodes}
    for source, targets in dag["successors"].items():
        for target in targets:
            undirected[source].add(target)
            undirected[target].add(source)
    components = []
    unseen = set(nodes)
    for start in sorted(nodes):
        if start not in unseen:
            continue
        queue = deque([start])
        unseen.remove(start)
        component = []
        while queue:
            node_id = queue.popleft()
            component.append(node_id)
            for neighbour in sorted(undirected[node_id]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return _ordered_blocks(components, dag["topological_order"], graph_data)



def _make_plan(blocks, num_cores, graph_data):
    mapping = {
        str(node_id): block["subgraph_id"]
        for block in blocks
        for node_id in block["node_ids"]
    }
    core_schedules = assign_lpt(blocks, num_cores)
    return {"node_to_subgraph": mapping, "core_schedules": core_schedules}


def _dependency_pairs(blocks, dag):
    mapping = {
        node_id: block["subgraph_id"]
        for block in blocks
        for node_id in block["node_ids"]
    }
    pairs = set()
    for source, targets in dag["successors"].items():
        for target in targets:
            if mapping[source] != mapping[target]:
                pairs.add((mapping[source], mapping[target]))
    return sorted(pairs)


def _candidate(name, blocks, num_cores, graph_data, dag):
    plan = _make_plan(blocks, num_cores, graph_data)
    traffic = estimate_partition_copy_bytes(graph_data, {
        int(node_id): subgraph_id for node_id, subgraph_id in plan["node_to_subgraph"].items()
    })
    score = estimate_problem1_score(
        blocks,
        plan["core_schedules"],
        dependency_pairs=_dependency_pairs(blocks, dag),
    )
    return {
        "rule": name,
        "plan": plan,
        "blocks": blocks,
        "estimated_score": score,
        "estimated_partition_added_copy_bytes": traffic["partition_added_copy_bytes"],
        "estimated_scheduled_boundary_bytes": traffic["scheduled_boundary_bytes"],
        "valid": True,
        "validation_error": "",
    }


def generate_candidates(graph, num_cores, target_blocks=None, rules=None):
    """Generate and validate structured candidate plans for one graph."""
    graph_data = build_op_dag(graph)
    dag = graph_data["noncopy"]
    target_blocks = target_blocks or max(1, num_cores)
    selected = list(rules or ("single", "continuous", "wave", "aligned", "components"))
    factories = {
        "single": lambda: _ordered_blocks([dag["topological_order"]], dag["topological_order"], graph_data),
        "continuous": lambda: _continuous_blocks(dag, graph_data, target_blocks),
        "wave": lambda: _wave_blocks(dag, graph_data, target_blocks),
        "aligned": lambda: _aligned_blocks(dag, graph_data, target_blocks),
        "components": lambda: _component_blocks(dag, graph_data),
    }
    candidates = []
    for name in selected:
        if name not in factories:
            raise ValueError(f"unknown partition rule {name!r}")
        blocks = factories[name]()
        candidate = _candidate(name, blocks, num_cores, graph_data, dag)
        # Rebuild the exact original graph for the authoritative validator.
        plan = candidate["plan"]
        try:
            validate_multicore_plan(graph, plan)
        except Exception as error:
            candidate["valid"] = False
            candidate["validation_error"] = str(error)
        else:
            candidate["valid"] = True
            candidate["validation_error"] = ""
        candidates.append(candidate)
    return candidates


def _json_safe_candidate(candidate):
    return {
        key: value for key, value in candidate.items()
        if key != "plan"
    }


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Generate structured problem-1 candidate plans")
    parser.add_argument("graph", type=Path)
    parser.add_argument("-n", "--num-cores", type=int, default=4)
    parser.add_argument("--target-blocks", type=int)
    parser.add_argument("--rules", nargs="+", choices=("single", "continuous", "wave", "aligned", "components"))
    parser.add_argument("--output-dir", type=Path, default=Path("day2_results"))
    args = parser.parse_args(argv)
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    candidates = generate_candidates(graph, args.num_cores, args.target_blocks, args.rules)
    case_dir = args.output_dir / args.graph.stem / f"cores_{args.num_cores}"
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for candidate in candidates:
        rule = candidate["rule"]
        plan_path = case_dir / f"{rule}.json"
        meta_path = case_dir / f"{rule}.meta.json"
        plan_path.write_text(json.dumps(candidate["plan"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        meta_path.write_text(json.dumps(_json_safe_candidate(candidate), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary.append({
            "rule": rule,
            "valid": candidate["valid"],
            "num_blocks": len(candidate["blocks"]),
            "estimated_score": candidate["estimated_score"],
            "estimated_partition_added_copy_bytes": candidate["estimated_partition_added_copy_bytes"],
            "plan": str(plan_path),
            "validation_error": candidate["validation_error"],
        })
    (case_dir / "candidates.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    _main()
