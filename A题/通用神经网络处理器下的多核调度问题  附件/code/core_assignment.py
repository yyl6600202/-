"""Core assignment and cheap diagnostics for Day 2 candidate plans."""

from __future__ import annotations

from collections import defaultdict


PIPES = ("PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V")
COPY_OPS = frozenset({"COPY_IN", "COPY_OUT"})


def block_statistics(graph_data, node_ids):
    """Return JSON-friendly load statistics for one subgraph block."""
    op_by_id = graph_data["op_by_id"]
    pipe_cycles = {pipe: 0 for pipe in PIPES}
    cycles = 0
    for op_id in node_ids:
        op = op_by_id[op_id]
        value = op["cycles"]
        cycles += value
        pipe_cycles.setdefault(op["pipe"], 0)
        pipe_cycles[op["pipe"]] += value
    return {
        "node_ids": list(node_ids),
        "cycles": cycles,
        "pipe_cycles": pipe_cycles,
        "load": max(pipe_cycles.values(), default=0),
    }


def assign_lpt(blocks, num_cores):
    """Assign blocks with an LPT-style projected Pipe load objective.

    Blocks are considered from heaviest to lightest.  The returned schedules
    are sorted by subgraph id, which is valid for the topological block rules
    in ``initial_partition.py`` and gives deterministic evaluator input.
    """
    if type(num_cores) is not int or num_cores <= 0:
        raise ValueError("num_cores must be a positive integer")
    if not blocks:
        return [[] for _ in range(num_cores)]
    core_pipe_loads = [{pipe: 0 for pipe in PIPES} for _ in range(num_cores)]
    core_total_loads = [0] * num_cores
    assigned = [[] for _ in range(num_cores)]
    for block in sorted(
        blocks,
        key=lambda item: (-item.get("cycles", 0), item.get("subgraph_id", 0)),
    ):
        pipe_cycles = block.get("pipe_cycles", {})
        scored = []
        for core_id in range(num_cores):
            projected = {
                pipe: core_pipe_loads[core_id].get(pipe, 0) + pipe_cycles.get(pipe, 0)
                for pipe in set(PIPES) | set(pipe_cycles)
            }
            scored.append((
                max(projected.values(), default=0),
                core_total_loads[core_id] + block.get("cycles", 0),
                len(assigned[core_id]),
                core_id,
            ))
        core_id = min(scored)[-1]
        assigned[core_id].append(block["subgraph_id"])
        core_total_loads[core_id] += block.get("cycles", 0)
        for pipe, value in pipe_cycles.items():
            core_pipe_loads[core_id][pipe] = core_pipe_loads[core_id].get(pipe, 0) + value
    for order in assigned:
        order.sort()
    return assigned


def estimate_partition_copy_bytes(graph_data, mapping):
    """Estimate problem-1 boundary traffic for an Op-to-subgraph mapping.

    COPY_IN/COPY_OUT are excluded from ``mapping``.  A tensor is copied once
    per destination Task and once out of its producer Task, matching the
    evaluator's Task/tensor de-duplication rule.
    """
    op_by_id = graph_data["op_by_id"]
    tensor_by_id = graph_data["tensor_by_id"]
    producer = graph_data["producer"]
    consumers = graph_data["consumers"]
    eligible = set(mapping)
    original = 0
    original_tensor_ids = set()
    for tensor_id, producer_id in producer.items():
        if op_by_id[producer_id].get("op") == "COPY_IN":
            original_tensor_ids.add(tensor_id)
    for tensor_id, consumer_ids in consumers.items():
        if any(op_by_id[op_id].get("op") == "COPY_OUT" for op_id in consumer_ids):
            original_tensor_ids.add(tensor_id)
    original = sum(tensor_by_id[tensor_id]["size"] for tensor_id in original_tensor_ids)

    scheduled = 0
    tensor_diagnostics = []
    for tensor_id, tensor in tensor_by_id.items():
        producer_id = producer.get(tensor_id)
        producer_subgraph = mapping.get(producer_id) if producer_id in eligible else None
        consumer_ids = consumers.get(tensor_id, set())
        consumer_subgraphs = {
            mapping[op_id]
            for op_id in consumer_ids
            if op_id in eligible
        }
        has_copy_out = any(
            op_by_id[op_id].get("op") == "COPY_OUT" for op_id in consumer_ids
        )
        input_copies = 0
        output_copies = 0
        if producer_subgraph is None:
            input_copies = len(consumer_subgraphs)
        else:
            outside = consumer_subgraphs - {producer_subgraph}
            input_copies = len(outside)
            output_copies = 1 if outside else 0
        if has_copy_out:
            output_copies = max(output_copies, 1)
        copies = input_copies + output_copies
        if copies:
            scheduled += copies * tensor["size"]
            tensor_diagnostics.append({
                "tensor_id": tensor_id,
                "size": tensor["size"],
                "input_copies": input_copies,
                "output_copies": output_copies,
                "consumer_subgraphs": sorted(consumer_subgraphs),
            })
    return {
        "original_graph_copy_bytes": original,
        "scheduled_boundary_bytes": scheduled,
        "partition_added_copy_bytes": scheduled - original,
        "tensor_diagnostics": tensor_diagnostics,
    }


def estimate_problem1_score(blocks, core_schedules, dependency_pairs=None,
                            same_core_wait=100, cross_core_wait=1000):
    """Cheap cycle-only score used to rank core assignments before simulation."""
    by_id = {block["subgraph_id"]: block for block in blocks}
    core_by_subgraph = {
        subgraph_id: core_id
        for core_id, order in enumerate(core_schedules)
        for subgraph_id in order
    }
    core_pipe_loads = [defaultdict(int) for _ in core_schedules]
    core_end = {}
    for core_id, order in enumerate(core_schedules):
        previous_end = 0
        for index, subgraph_id in enumerate(order):
            block = by_id[subgraph_id]
            predecessor_ends = [
                core_end[source] + (
                    same_core_wait if core_by_subgraph[source] == core_id else cross_core_wait
                )
                for source, target in (dependency_pairs or [])
                if target == subgraph_id and source in core_end
            ]
            start = max([previous_end + (same_core_wait if index else 0)] + predecessor_ends)
            duration = max(block.get("pipe_cycles", {}).values(), default=block.get("cycles", 0))
            core_end[subgraph_id] = start + duration
            previous_end = core_end[subgraph_id]
            for pipe, value in block.get("pipe_cycles", {}).items():
                core_pipe_loads[core_id][pipe] += value
    max_pipe_load = max(
        (max(loads.values(), default=0) for loads in core_pipe_loads),
        default=0,
    )
    return max(max_pipe_load, max(core_end.values(), default=0))
