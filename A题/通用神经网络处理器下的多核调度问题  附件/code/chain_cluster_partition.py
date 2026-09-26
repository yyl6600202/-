"""Generate acyclic chain based partitions for problem 1.

The generator is a deterministic implementation of the supplied model note.
It assigns each operation to the root of a dominant predecessor chain, packs
whole root branches onto cores, and cuts the physical execution order only
when the core label changes.  Each Task is consequently a contiguous interval
of one topological order, so the contracted Task graph stays acyclic while
the large tensor edges remain local whenever possible.

This module is independent from ``initial_partition`` and can be imported by
a batch search or run directly to emit JSON plans.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from graph_features import build_op_dag
from schedule_step1 import step1_from_adj
from stub_multicore_cut_and_schedule import validate_multicore_plan


DEFAULT_MODES = (
    "root_contiguous",
    "root_contiguous_weighted",
    "root_roundrobin",
    "root_roundrobin_weighted",
    "root_load",
    "root_id",
)


def _execution_order(graph_data):
    """Return the repository's deterministic physical topological order."""
    dag = graph_data["noncopy"]
    ops = [graph_data["op_by_id"][node_id] for node_id in dag["nodes"]]
    order = step1_from_adj(
        dag["predecessors"], dag["successors"], dag["nodes"], ops)
    position = {node_id: index for index, node_id in enumerate(order)}
    if set(order) != set(dag["nodes"]):
        raise ValueError("step1 did not return every non-COPY operation")
    if any(position[source] >= position[target]
           for source, target in dag["edges"]):
        return list(dag["topological_order"])
    return order


def _edge_bytes(graph_data):
    """Aggregate producer-to-consumer tensor bytes for non-COPY operations."""
    producer = graph_data["producer"]
    consumers = graph_data["consumers"]
    tensors = graph_data["tensor_by_id"]
    noncopy = set(graph_data["noncopy"]["nodes"])
    incoming = defaultdict(lambda: defaultdict(int))
    for tensor_id, source in producer.items():
        if source not in noncopy:
            continue
        size = tensors[tensor_id]["size"]
        for target in consumers.get(tensor_id, ()):
            if target in noncopy:
                incoming[target][source] += size
    return {target: dict(sources) for target, sources in incoming.items()}


def _roots_and_lineage(graph_data, order, dominant="bytes"):
    """Assign every operation to a root through a dominant predecessor chain."""
    dag = graph_data["noncopy"]
    op_by_id = graph_data["op_by_id"]
    incoming = _edge_bytes(graph_data)
    lineage = {}
    roots = []
    position = {node_id: index for index, node_id in enumerate(order)}
    for node_id in order:
        predecessors = [pred for pred in dag["predecessors"][node_id]
                        if pred in lineage]
        if not predecessors:
            lineage[node_id] = node_id
            roots.append(node_id)
            continue

        def key(pred):
            traffic = incoming.get(node_id, {}).get(pred, 0)
            pred_cycles = op_by_id[pred]["cycles"]
            if dominant == "cycles":
                return (pred_cycles, traffic, -position[pred], -pred)
            if dominant == "id":
                return (pred, traffic, pred_cycles, -position[pred])
            return (traffic, pred_cycles, -position[pred], -pred)

        parent = max(predecessors, key=key)
        lineage[node_id] = lineage[parent]
    root_order = list(dict.fromkeys(roots))
    branch_cycles = {root: 0 for root in root_order}
    branch_nodes = {root: [] for root in root_order}
    for node_id in order:
        root = lineage[node_id]
        branch_cycles[root] += op_by_id[node_id]["cycles"]
        branch_nodes[root].append(node_id)
    return lineage, root_order, branch_cycles, branch_nodes


def _split_contiguous(items, weights, count, weighted=False):
    """Split ordered roots into ``count`` non-empty contiguous groups."""
    if not items:
        return []
    count = max(1, min(int(count), len(items)))
    if not weighted:
        base, extra = divmod(len(items), count)
        groups, cursor = [], 0
        for index in range(count):
            size = base + (1 if index < extra else 0)
            groups.append(items[cursor:cursor + size])
            cursor += size
        return groups

    groups = []
    cursor = 0
    remaining_weight = sum(weights)
    for group_index in range(count - 1):
        remaining_groups = count - group_index
        target = remaining_weight / remaining_groups
        start = cursor
        current = 0
        max_cursor = len(items) - (remaining_groups - 1)
        while cursor < max_cursor:
            current += weights[cursor]
            cursor += 1
            if current >= target and cursor > start:
                break
        groups.append(items[start:cursor])
        remaining_weight -= current
    groups.append(items[cursor:])
    return [group for group in groups if group]


def _assign_roots(root_order, branch_cycles, num_cores, mode):
    """Map root branches to cores using one of the chain packing rules."""
    if not root_order:
        return {}
    if mode in {"root_contiguous", "root_contiguous_weighted"}:
        groups = _split_contiguous(
            root_order,
            [branch_cycles[root] for root in root_order],
            num_cores,
            weighted=mode.endswith("weighted"),
        )
        return {root: core for core, group in enumerate(groups) for root in group}
    if mode in {"root_roundrobin", "root_roundrobin_weighted"}:
        # More groups than cores let a long branch finish before the next
        # branch starts on the same core.  The resulting short Task runs are
        # useful for deep graphs whose five large contiguous groups serialize
        # too much work.  Use at most 20 groups to cap boundary traffic.
        group_count = min(len(root_order), max(num_cores * 4, num_cores + 1))
        groups = _split_contiguous(
            root_order,
            [branch_cycles[root] for root in root_order],
            group_count,
            weighted=mode.endswith("weighted"),
        )
        return {
            root: (group_index % num_cores)
            for group_index, group in enumerate(groups)
            for root in group
        }
    if mode == "root_load":
        loads = [0] * num_cores
        result = {}
        for root in sorted(root_order, key=lambda item: (-branch_cycles[item], item)):
            core = min(range(num_cores), key=lambda item: (loads[item], item))
            result[root] = core
            loads[core] += branch_cycles[root]
        return result
    if mode == "root_id":
        return {root: index % num_cores for index, root in enumerate(root_order)}
    raise ValueError(f"unknown chain assignment mode {mode!r}")


def _make_plan(order, lineage, root_to_core, num_cores):
    """Cut consecutive equal core labels into Tasks and build the plan."""
    blocks = []
    current_core = None
    current_nodes = []
    for node_id in order:
        core = root_to_core[lineage[node_id]]
        if current_nodes and core != current_core:
            blocks.append((current_core, current_nodes))
            current_nodes = []
        current_core = core
        current_nodes.append(node_id)
    if current_nodes:
        blocks.append((current_core, current_nodes))

    mapping = {}
    core_schedules = [[] for _ in range(num_cores)]
    for subgraph_id, (core, node_ids) in enumerate(blocks):
        for node_id in node_ids:
            mapping[str(node_id)] = subgraph_id
        core_schedules[core].append(subgraph_id)
    return {
        "node_to_subgraph": mapping,
        "core_schedules": core_schedules,
    }, blocks


def generate_chain_candidates(graph, num_cores, bandwidth=60, modes=None,
                              dominants=("bytes", "cycles", "id")):
    """Generate validated chain candidates for one graph.

    ``bandwidth`` is accepted so callers can pass the evaluator config; the
    official evaluator itself prices the resulting boundary traffic.
    """
    del bandwidth
    if type(num_cores) is not int or num_cores <= 0:
        raise ValueError("num_cores must be a positive integer")
    graph_data = build_op_dag(graph)
    order = _execution_order(graph_data)
    selected_modes = tuple(modes or DEFAULT_MODES)
    candidates = []
    for dominant in dominants:
        lineage, roots, branch_cycles, branch_nodes = _roots_and_lineage(
            graph_data, order, dominant)
        for mode in selected_modes:
            root_to_core = _assign_roots(roots, branch_cycles, num_cores, mode)
            plan, blocks = _make_plan(order, lineage, root_to_core, num_cores)
            valid = True
            error = ""
            try:
                validate_multicore_plan(graph, plan)
            except Exception as exc:
                valid = False
                error = f"{type(exc).__name__}: {exc}"
            candidates.append({
                "rule": f"chain_{dominant}_{mode}",
                "mode": mode,
                "dominant": dominant,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--num-cores", type=int, default=5)
    parser.add_argument("--bandwidth", type=int, default=60)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES,
                        default=list(DEFAULT_MODES))
    parser.add_argument("--dominants", nargs="+", choices=("bytes", "cycles", "id"),
                        default=["bytes"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("day5_results/chain_plans"))
    args = parser.parse_args(argv)
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    candidates = generate_chain_candidates(
        graph, args.num_cores, args.bandwidth, args.modes, args.dominants)
    case_dir = args.output_dir / args.graph.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        plan_path = case_dir / f"{candidate['rule']}.json"
        plan_path.write_text(
            json.dumps(candidate["plan"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            key: value for key, value in candidate.items()
            if key not in {"plan", "blocks", "branch_cycles", "branch_sizes"}
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
