# A题问题一：5核链簇生成器 v5（混合搜索）

v5 在 v4 的 100 个合法方案上，对最难的 20 个 case 使用遗传算法 + 模拟退火生成候选，并只接受官方事件模拟器验证后 Makespan 严格下降的方案。

## 结果

- 算术平均加速比：**3.351113×**
- 几何平均加速比：2.982523×
- 中位数加速比：3.716115×
- 总周期比：3.633128×
- 100 个 case，官方方案校验错误：0，结构校验错误：0
- 混合搜索接受改进：6 个 case

## 接受的改进

| case | v4 Makespan | v5 Makespan | 规则 |
|---|---:|---:|---|
| case_009 | 92225 | 89919 | `hybrid_id_orig_5` |
| case_044 | 132892 | 131105 | `hybrid_bytes_load_desc_2` |
| case_047 | 326508 | 320115 | `hybrid_id_orig_2` |
| case_066 | 249724 | 242906 | `hybrid_id_id_7` |
| case_069 | 25008 | 21040 | `hybrid_id_load_desc_5` |
| case_071 | 17366 | 16183 | `hybrid_id_orig_4` |

所有候选仍使用连续 Task 切分，并通过 `validate_multicore_plan`；未严格下降的候选保留 v4 方案。

复现：

```powershell
python code/hybrid_chain_search.py --worst 20 --official-budget 12 --population 16 --generations 6 --anneal-steps 40 --output day5_results/hybrid_search_delta.csv --plans-dir day5_results/hybrid_search_plans
python code/merge_hybrid_v5.py
```
