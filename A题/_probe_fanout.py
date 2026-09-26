import json
import sys
from pathlib import Path

ROOT = Path(r"A题/通用神经网络处理器下的多核调度问题  附件/code")
sys.path.insert(0, str(ROOT))

from multicore_cut_evaluate_problem_2 import _build_scene_b_tasks, evaluate_scene_b
from multicore_cut_evaluate_problem_3 import evaluate_problem_3

GRAPH = {
    "tensors": [
        {"id": 1, "pos": "DDR", "size": 8},
        {"id": 2, "pos": "L1", "size": 8},
        {"id": 3, "pos": "UB", "size": 16},
        {"id": 4, "pos": "L1", "size": 4},
        {"id": 5, "pos": "DDR", "size": 4},
        {"id": 6, "pos": "L1", "size": 4},
        {"id": 7, "pos": "DDR", "size": 4},
    ],
    "ops": [
        {"id": 10, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 1},
        {"id": 11, "op": "P", "pipe": "PIPE_M", "cycles": 2},
        {"id": 12, "op": "C1", "pipe": "PIPE_V", "cycles": 2},
        {"id": 13, "op": "C2", "pipe": "PIPE_V", "cycles": 2},
        {"id": 14, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 1},
        {"id": 15, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 1},
    ],
    "edges": [
        {"source": 1, "target": 10}, {"source": 10, "target": 2},
        {"source": 2, "target": 11}, {"source": 11, "target": 3},
        {"source": 3, "target": 12}, {"source": 3, "target": 13},
        {"source": 12, "target": 4}, {"source": 4, "target": 14},
        {"source": 14, "target": 5}, {"source": 13, "target": 6},
        {"source": 6, "target": 15}, {"source": 15, "target": 7},
    ],
}
PLAN = {"node_to_subgraph": {"11": 0, "12": 1, "13": 2},
        "core_schedules": [[0], [1], [2]]}
CAPACITY = {"L1": 524288, "UB": 131072}

tasks, links, traffic, movement, _ = _build_scene_b_tasks(
    GRAPH, PLAN, 60, CAPACITY)
print("links", json.dumps(links, sort_keys=True))
for core, task in tasks.items():
    copies = [(op_id, op["op"]) for op_id, op in task["op_by_id"].items()
              if op["op"] in ("COPY_IN", "COPY_OUT")]
    print("core", core, copies)
print("q2", evaluate_scene_b(GRAPH, PLAN, 60, CAPACITY, 500)["makespan"])
print("q3", evaluate_problem_3(
    GRAPH, PLAN, 60, CAPACITY, 500, 1048576, 250)["makespan"])
