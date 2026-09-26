"""Batch evaluate frozen 5-core plans for problems 2 and 3."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_2 import evaluate_scene_b, read_scene_b_config
from multicore_cut_evaluate_problem_3 import evaluate_problem_3, read_cache_config

def run(data_dir, plans_dir, output_dir, cases=None):
    data_dir, plans_dir, output_dir = map(Path, (data_dir, plans_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    # data/ also contains evaluator sidecar files such as case_001_problem_2_res.json;
    # only the canonical 3-digit graph inputs are valid here.
    names = cases or sorted(p.stem for p in data_dir.glob('case_*.json')
                            if re.fullmatch(r'case_\d{3}', p.stem))
    config_path = data_dir/'config.txt'
    ev = read_evaluation_config(config_path)
    scene_b = read_scene_b_config(config_path)
    cache = read_cache_config(config_path)
    rows=[]
    fields=['case','num_cores','q2_makespan','q3_makespan','q2_scheduled_copy_bytes','q3_physical_ddr_bytes','cache_hits','cache_accesses','cache_hit_rate','error']
    out=output_dir/'problem23_5core_summary.csv'
    if out.exists():
        with out.open(newline='', encoding='utf-8-sig') as h:
            rows = list(csv.DictReader(h))
        # A long 100-case run may be interrupted by a process timeout.  Keep
        # completed rows and evaluate only cases that are still absent.  An
        # explicit --case list is treated the same way, so rerunning a range
        # is idempotent.
        completed = {r.get('case') for r in rows if r.get('case')}
        names = [name for name in names if name not in completed]
    for name in names:
        row={f:'' for f in fields}; row.update(case=name,num_cores=5)
        try:
            graph=json.loads((data_dir/f'{name}.json').read_text(encoding='utf-8'))
            plan=json.loads((plans_dir/f'{name}.json').read_text(encoding='utf-8'))
            q2=evaluate_scene_b(graph,plan,ev['bandwidth'],ev['capacity'],scene_b['cross_core_copy_delay_cycles'])
            q3=evaluate_problem_3(graph,plan,ev['bandwidth'],ev['capacity'],scene_b['cross_core_copy_delay_cycles'],cache['cache_capacity_bytes'],cache['cache_bandwidth_bytes_per_cycle'])
            m2=q2.get('data_movement_bytes',{}); m3=q3.get('data_movement_bytes',{}); cs=q3.get('cache_stats',{})
            row.update(q2_makespan=q2['makespan'],q3_makespan=q3['makespan'],q2_scheduled_copy_bytes=m2.get('scheduled_copy_bytes',0),q3_physical_ddr_bytes=m3.get('physical_ddr_bytes',0),cache_hits=cs.get('hits',0),cache_accesses=cs.get('accesses',0),cache_hit_rate=cs.get('hit_rate',0.0))
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
        rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
        with out.open('w',newline='',encoding='utf-8-sig') as h:
            w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,default=Path('data')); ap.add_argument('--plans-dir',type=Path,default=Path('day5_results/paper_guided_v6_plans')); ap.add_argument('--output-dir',type=Path,default=Path('day5_results/problem23_5core')); ap.add_argument('--case',action='append',dest='cases'); a=ap.parse_args(); run(a.data_dir,a.plans_dir,a.output_dir,a.cases)
