"""Build the operation DAG and extract deterministic graph features.

The contest input uses one global id space, but keeps operations and tensors in
separate lists.  This module therefore always resolves edge endpoints through
those two indexes instead of inferring a node type from its numeric id.

The public helpers are intentionally small so later partition/search modules
can reuse the same representation:

``build_op_dag(graph)``
    Validate the Tensor-Op graph and return both the full operation DAG and a
    COPY-filtered DAG.
``extract_features(graph, dag=None, capacities=None)``
    Return JSON/CSV-friendly scalar features and per-operation/tensor details.
"""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


COPY_OPS = frozenset({"COPY_IN", "COPY_OUT"})


def _node_id(node: Any, context: str) -> int:
    if type(node) is not int or node < 0:
        raise ValueError(f"{context} must be a non-negative integer; got {node!r}")
    return node


def _validate_and_index(graph: dict[str, Any]):
    if not isinstance(graph, dict):
        raise ValueError("graph must be an object")
    for key in ("ops", "tensors", "edges"):
        if not isinstance(graph.get(key), list):
            raise ValueError(f"graph.{key} must be a list")

    op_by_id: dict[int, dict[str, Any]] = {}
    tensor_by_id: dict[int, dict[str, Any]] = {}
    for index, op in enumerate(graph["ops"]):
        if not isinstance(op, dict):
            raise ValueError(f"ops[{index}] must be an object")
        op_id = _node_id(op.get("id"), f"ops[{index}].id")
        if op_id in op_by_id:
            raise ValueError(f"duplicate operation id {op_id}")
        if "pipe" not in op or not isinstance(op["pipe"], str):
            raise ValueError(f"operation {op_id} must have a pipe")
        if type(op.get("cycles")) is not int or op["cycles"] < 0:
            raise ValueError(f"operation {op_id}.cycles must be a non-negative integer")
        op_by_id[op_id] = op

    for index, tensor in enumerate(graph["tensors"]):
        if not isinstance(tensor, dict):
            raise ValueError(f"tensors[{index}] must be an object")
        tensor_id = _node_id(tensor.get("id"), f"tensors[{index}].id")
        if tensor_id in tensor_by_id:
            raise ValueError(f"duplicate tensor id {tensor_id}")
        if tensor_id in op_by_id:
            raise ValueError(f"operation/tensor id collision at {tensor_id}")
        if tensor.get("pos") not in {"DDR", "L1", "UB"}:
            raise ValueError(f"tensor {tensor_id}.pos must be DDR, L1, or UB")
        if type(tensor.get("size")) is not int or tensor["size"] < 0:
            raise ValueError(f"tensor {tensor_id}.size must be a non-negative integer")
        tensor_by_id[tensor_id] = tensor

    return op_by_id, tensor_by_id


def _topological_order(nodes: Iterable[int], successors: dict[int, set[int]], context: str):
    """Return a deterministic Kahn order, or raise with a useful cycle error."""
    node_set = set(nodes)
    indegree = {node: 0 for node in node_set}
    for source in node_set:
        for target in successors.get(source, ()):
            if target not in node_set:
                continue
            indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        source = heapq.heappop(ready)
        order.append(source)
        for target in sorted(successors.get(source, ())):
            if target not in indegree:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(order) != len(node_set):
        remaining = sorted(node for node, degree in indegree.items() if degree > 0)
        raise ValueError(f"{context} contains a cycle; remaining nodes include {remaining[:10]}")
    return order


def _dag_for_nodes(all_nodes: set[int], successors: dict[int, set[int]]):
    selected = {node: set(targets) & all_nodes for node, targets in successors.items() if node in all_nodes}
    for node in all_nodes:
        selected.setdefault(node, set())
    predecessors = {node: set() for node in all_nodes}
    for source, targets in selected.items():
        for target in targets:
            predecessors[target].add(source)
    order = _topological_order(all_nodes, selected, "operation DAG")
    return {
        "nodes": sorted(all_nodes),
        "successors": selected,
        "predecessors": predecessors,
        "edges": [(source, target) for source in sorted(selected) for target in sorted(selected[source])],
        "topological_order": order,
    }


def build_op_dag(graph: dict[str, Any]):
    """Validate ``graph`` and return full and COPY-filtered operation DAGs.

    Tensor nodes are contracted as ``producer -> tensor -> consumer``.  The
    returned mapping is made of ordinary Python values so it can be serialized
    or passed directly to later search modules.
    """
    op_by_id, tensor_by_id = _validate_and_index(graph)
    producer: dict[int, int] = {}
    consumers: dict[int, set[int]] = defaultdict(set)
    seen_edges: set[tuple[int, int]] = set()
    for index, edge in enumerate(graph["edges"]):
        if not isinstance(edge, dict):
            raise ValueError(f"edges[{index}] must be an object")
        source = _node_id(edge.get("source"), f"edges[{index}].source")
        target = _node_id(edge.get("target"), f"edges[{index}].target")
        pair = (source, target)
        if pair in seen_edges:
            raise ValueError(f"duplicate edge {source} -> {target}")
        seen_edges.add(pair)
        if source in op_by_id and target in tensor_by_id:
            if target in producer:
                raise ValueError(f"tensor {target} has multiple producers")
            producer[target] = source
        elif source in tensor_by_id and target in op_by_id:
            consumers[source].add(target)
        else:
            raise ValueError(f"edge {source} -> {target} is not Tensor-Op bipartite")

    full_successors: dict[int, set[int]] = {op_id: set() for op_id in op_by_id}
    for tensor_id, source in producer.items():
        for target in consumers.get(tensor_id, ()):
            if source != target:
                full_successors[source].add(target)
    full = _dag_for_nodes(set(op_by_id), full_successors)
    noncopy_nodes = {op_id for op_id, op in op_by_id.items() if op.get("op") not in COPY_OPS}
    noncopy = _dag_for_nodes(noncopy_nodes, full_successors)

    return {
        "op_by_id": op_by_id,
        "tensor_by_id": tensor_by_id,
        "producer": producer,
        "consumers": {tensor_id: set(values) for tensor_id, values in consumers.items()},
        "full": full,
        "noncopy": noncopy,
    }


def _capacity_values(capacities: dict[str, int] | None):
    if capacities is None:
        raise ValueError("capacities are required; read L1 and UB from config.txt")
    values = dict(capacities)
    for position in ("L1", "UB"):
        if type(values.get(position)) is not int or values[position] <= 0:
            raise ValueError(f"capacity {position} must be a positive integer")
    return values


def _peak_lifetimes(graph_data, dag, capacities):
    op_by_id = graph_data["op_by_id"]
    tensor_by_id = graph_data["tensor_by_id"]
    producer = graph_data["producer"]
    consumers = graph_data["consumers"]
    order = dag["topological_order"]
    position = {op_id: index for index, op_id in enumerate(order)}
    peaks = {"L1": 0, "UB": 0}
    interval_count = {"L1": 0, "UB": 0}
    intervals = []
    events: dict[str, dict[int, int]] = {"L1": defaultdict(int), "UB": defaultdict(int)}
    for tensor_id, tensor in tensor_by_id.items():
        storage = tensor["pos"]
        if storage not in events:
            continue
        uses = [position[op_id] for op_id in consumers.get(tensor_id, ()) if op_id in position]
        producer_id = producer.get(tensor_id)
        start = position.get(producer_id, 0)
        if not uses:
            continue
        end = max(uses)
        if end < start:
            start = end
        size = tensor["size"]
        events[storage][start] += size
        events[storage][end + 1] -= size
        interval_count[storage] += 1
        intervals.append({"tensor_id": tensor_id, "pos": storage, "start": start, "end": end, "size": size})
    for storage, storage_events in events.items():
        live = 0
        for index in sorted(storage_events):
            live += storage_events[index]
            peaks[storage] = max(peaks[storage], live)
    return peaks, interval_count, intervals


def extract_features(graph: dict[str, Any], dag=None, capacities=None):
    """Extract scalar and detailed features for one input graph."""
    data = dag if dag is not None and "op_by_id" in dag else build_op_dag(graph)
    capacities = _capacity_values(capacities)
    op_by_id = data["op_by_id"]
    tensor_by_id = data["tensor_by_id"]
    noncopy = data["noncopy"]
    noncopy_ids = set(noncopy["nodes"])
    order = noncopy["topological_order"]
    predecessors = noncopy["predecessors"]

    levels: dict[int, int] = {}
    critical_path: dict[int, int] = {}
    for op_id in order:
        pred = predecessors[op_id]
        levels[op_id] = 0 if not pred else 1 + max(levels[node] for node in pred)
        cycles = op_by_id[op_id]["cycles"]
        critical_path[op_id] = cycles + max((critical_path[node] for node in pred), default=0)
    layer_counts = Counter(levels.values())
    peaks, interval_count, intervals = _peak_lifetimes(data, noncopy, capacities)

    pipe_cycles = Counter()
    pipe_counts = Counter()
    for op in op_by_id.values():
        pipe = op.get("pipe", "UNKNOWN")
        pipe_cycles[pipe] += op["cycles"]
        pipe_counts[pipe] += 1
    noncopy_pipe_cycles = Counter(op_by_id[op_id]["pipe"] for op_id in noncopy_ids)
    noncopy_pipe_cycles = Counter({
        pipe: sum(op_by_id[op_id]["cycles"] for op_id in noncopy_ids if op_by_id[op_id]["pipe"] == pipe)
        for pipe in sorted({op_by_id[op_id]["pipe"] for op_id in noncopy_ids})
    })

    tensor_fanout = {
        tensor_id: len([op_id for op_id in data["consumers"].get(tensor_id, ()) if op_id in noncopy_ids])
        for tensor_id in tensor_by_id
    }
    ddr_repeat_ids = [
        tensor_id for tensor_id, fanout in tensor_fanout.items()
        if tensor_by_id[tensor_id]["pos"] == "DDR" and fanout > 1
    ]
    max_fanout_tensor = max(tensor_fanout, key=lambda item: (tensor_fanout[item], -item), default=None)
    max_tensor_id = max(tensor_by_id, key=lambda item: (tensor_by_id[item]["size"], -item), default=None)
    noncopy_cycles = sum(op_by_id[op_id]["cycles"] for op_id in noncopy_ids)
    max_level = max(levels.values(), default=-1)
    depth = max_level + 1 if order else 0
    max_width = max(layer_counts.values(), default=0)
    peak_ratio = max(
        peaks[storage] / capacities[storage] for storage in ("L1", "UB")
    ) if capacities else 0.0
    # A small capacity overshoot is common in the conservative lifetime
    # estimate. Reserve this class for cases whose estimated peak is at least
    # twice the configured capacity; otherwise topology remains informative.
    if peak_ratio >= 2.0:
        graph_class = "high_spill_risk"
    elif max_width >= max(8, depth * 2):
        graph_class = "wide_shallow"
    elif depth >= max(20, max_width * 3):
        graph_class = "deep_narrow"
    else:
        graph_class = "mixed"

    features = {
        "ops_total": len(op_by_id),
        "noncopy_ops": len(noncopy_ids),
        "copy_in_ops": sum(op.get("op") == "COPY_IN" for op in op_by_id.values()),
        "copy_out_ops": sum(op.get("op") == "COPY_OUT" for op in op_by_id.values()),
        "tensors_total": len(tensor_by_id),
        "edges_total": len(graph["edges"]),
        "op_dag_edges": len(noncopy["edges"]),
        "op_dag_depth": depth,
        "op_dag_max_width": max_width,
        "op_dag_avg_width": (len(order) / depth) if depth else 0.0,
        "critical_path_cycles": max(critical_path.values(), default=0),
        "noncopy_cycles": noncopy_cycles,
        "max_tensor_size": tensor_by_id[max_tensor_id]["size"] if max_tensor_id is not None else 0,
        "max_tensor_id": max_tensor_id,
        "max_tensor_fanout": tensor_fanout.get(max_fanout_tensor, 0) if max_fanout_tensor is not None else 0,
        "max_fanout_tensor_id": max_fanout_tensor,
        "ddr_repeat_tensor_count": len(ddr_repeat_ids),
        "ddr_repeat_potential_bytes": sum(
            tensor_by_id[tensor_id]["size"] * (tensor_fanout[tensor_id] - 1)
            for tensor_id in ddr_repeat_ids
        ),
        "l1_peak_bytes": peaks["L1"],
        "ub_peak_bytes": peaks["UB"],
        "l1_peak_ratio": peaks["L1"] / capacities["L1"],
        "ub_peak_ratio": peaks["UB"] / capacities["UB"],
        "spill_risk_ratio": peak_ratio,
        "graph_class": graph_class,
    }
    for pipe in sorted(pipe_cycles):
        safe_pipe = pipe.lower().replace("pipe_", "pipe_")
        features[f"{safe_pipe}_ops"] = pipe_counts[pipe]
        features[f"{safe_pipe}_cycles"] = pipe_cycles[pipe]
        features[f"{safe_pipe}_noncopy_cycles"] = noncopy_pipe_cycles.get(pipe, 0)
    for storage in ("L1", "UB", "DDR"):
        features[f"{storage.lower()}_tensor_count"] = sum(
            tensor.get("pos") == storage for tensor in tensor_by_id.values())
        features[f"{storage.lower()}_tensor_bytes"] = sum(
            tensor["size"] for tensor in tensor_by_id.values() if tensor.get("pos") == storage)

    return {
        **features,
        "capacities": capacities,
        "layer_counts": {str(level): count for level, count in sorted(layer_counts.items())},
        "op_features": {
            str(op_id): {
                "op": op_by_id[op_id].get("op"),
                "pipe": op_by_id[op_id].get("pipe"),
                "cycles": op_by_id[op_id]["cycles"],
                "level": levels[op_id],
                "critical_path_cycles": critical_path[op_id],
                "in_degree": len(predecessors[op_id]),
                "out_degree": len(noncopy["successors"].get(op_id, ())),
            }
            for op_id in order
        },
        "tensor_intervals": intervals,
    }


def profile_graph(graph: dict[str, Any], capacities=None):
    """Convenience wrapper used by the batch profile script."""
    dag = build_op_dag(graph)
    return extract_features(graph, dag=dag, capacities=capacities)


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Extract A题 operation-DAG features")
    parser.add_argument("graph", help="input graph JSON")
    parser.add_argument("--config", help="config.txt used for L1/UB capacities")
    parser.add_argument("-o", "--output", help="write full feature JSON")
    args = parser.parse_args(argv)
    graph_path = Path(args.graph)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    config_path = Path(args.config) if args.config else graph_path.parent / "config.txt"
    capacities = _read_capacity_file(config_path)
    result = profile_graph(graph, capacities)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _read_capacity_file(path):
    values = {}
    section = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        parts = line.split()
        if section == "capacity" and len(parts) == 2:
            values[parts[0]] = int(parts[1])
    return _capacity_values(values)


if __name__ == "__main__":
    _main()
