# 2026 华为杯研究生数学建模 A 题详细完成计划

## 0. 先用一分钟看懂题目

题目给你一张“工作先后关系图”：Op 是要做的工作，Tensor 是工作之间传递的数据，Core 是处理工作的处理器核心。你要决定三件事：哪些 Op 放在同一个子图、每个子图交给哪个 Core、同一个 Core 上的子图按什么顺序执行。

例如下面四项工作中，D 必须等 B 和 C 都完成后才能开始：

```text
A -> B -> D
 \-> C ->/
```

箭头表示“前面的工作产出数据，后面的工作要用它”。题目不是让你手算每项工作的具体开始时刻，而是让你输出一份合法的分组和排班方案；官方评估器会模拟这份方案，算出最后一项工作完成的时间，也就是 `Makespan`。分区、负载估计和搜索算法，都是为了找到一份让评估器测得更快的方案。

如果你刚开始做，只要先记住：**先保证每项工作都安排了、依赖顺序没错，再比较哪份排班让官方评估器算出的总时间更短。** 后面的术语和公式都是在帮助程序自动完成这两件事。

## 1. 题目目标与总体路线

题目研究的是通用神经网络处理器上的多核调度。输入是一张 Tensor-Op 二部有向无环图，节点分为 Tensor 和 Op，边表示数据依赖；每个 Op 具有计算周期和所属执行流水线，每个 Tensor 具有存储位置和大小。

需要提交的核心结果不是一段抽象算法，而是对每个测试图生成一个合法的调度方案 JSON，并通过官方离散事件评估器得到：

- 总执行时间 `Makespan`
- 原图搬运量、调度后搬运量和额外搬运量
- 跨核通信量与等待时间
- L1、UB 的峰值占用
- 问题 3 的 Cache 命中次数、命中字节数和命中率

建议采用“规则生成初始方案 + 快速近似评分 + 官方评估器精确回放 + 局部搜索改进”的路线：

1. 先读懂输入格式、配置文件、官方评估器和核内调度流程。
2. 将 Tensor-Op 图收缩成 Op-DAG，提取计算、通信、拓扑和复用特征。
3. 生成多个不同粒度的初始分块方案。
4. 针对三个问题分别进行核心分配和局部优化。
5. 每个候选方案都调用官方评估器做最终判定。
6. 批量运行 100 个 case、2～5 个核心，整理表格、图和论文结论。

三个问题应当递进处理：

- 问题 1：每个子图是独立 Task，所有跨子图边都经过 DDR。
- 问题 2：每个核心的子图合并成一个 Task，同核数据可以驻留片上，跨核边增加 `COPY_OUT/COPY_IN` 和同步延迟。
- 问题 3：问题 2 加共享只读 FIFO Cache，命中时使用 Cache 带宽，不占用 DDR。

## 2. 已知评估规则与约束

### 2.1 输入图

每个 case 包含：

- Tensor：
  - `id`
  - `pos`：`DDR`、`L1` 或 `UB`
  - `size`
- Op：
  - `id`
  - `op`
  - `pipe`
  - `cycles`
- Tensor-Op 二部边

输入校验的硬性前提（README 与 `contest_io` 强制，特征提取代码必须遵守）：

- Op 与 Tensor 共享同一 ID 空间且全局唯一；不能按 ID 大小判断节点类型（实测既有 `id≥10^9` 的普通 Op，也有小 id 的原始 COPY Op），必须按 `ops`/`tensors` 两个列表分别建索引。
- 只允许 Tensor↔Op 二部边；禁止重复边和 Tensor→Tensor 直接边（实测 100 个 case 中也不存在 Op→Op 直接边，评估器中的直接 Op→Op 边全部由 Step3 的内存复用补边产生）。
- 每个 Op 必带 `pipe` 字段且取值属于四条 Pipe；调度与仿真直接读该字段，不按 Op 名称推断。

流水线包括：

```text
PIPE_MTE2
PIPE_MTE3
PIPE_M
PIPE_V
```

### 2.2 方案 JSON

方案必须包含：

```json
{
  "node_to_subgraph": {
    "11": 0,
    "12": 0,
    "21": 1
  },
  "core_schedules": [[0], [1]]
}
```

这里的格式必须严格遵守官方 B.5：`node_to_subgraph` 的键是 Op 编号的十进制字符串，值是非负整数子图编号；`core_schedules` 是二维整数数组，外层下标就是核心编号，内层按执行顺序填写子图编号。不要写成字符串子图编号，也不要写成带 `core_id` 和 `subgraphs` 字段的对象数组。子图编号建议连续使用 `0,1,...,S-1`，每个核心即使没有任务也要保留一个空数组。

必须满足：

- 所有非 `COPY_IN/COPY_OUT` 操作恰好覆盖一次。
- 原始 `COPY_IN/COPY_OUT` 不应放入 `node_to_subgraph`，边界 COPY 由评估器自动补充。
- 每个子图在 `core_schedules` 中恰好出现一次。
- 子图依赖不能形成环。
- 每个核心上的子图顺序要满足依赖关系。
- `core_id` 必须落在指定核心数量范围内。
- 方案生成后必须在收缩子图图上重新检查无环；交错分区可能在原 Op-DAG 无环的情况下形成子图环。
- 问题 2/3 还必须检查按 `core_schedules` 分桶后的核内 Op 顺序是否满足依赖；非法顺序由评估器直接拒绝。

### 2.3 固定配置

当前配置中需要读取的典型参数为：

```text
L1 = 524288 bytes
UB = 131072 bytes
DDR bandwidth = 60 bytes/cycle
Problem 1 cross-core wait = 1000 cycles
Problem 1 same-core task wait = 100 cycles
Problem 2/3 cross-core copy latency = 500 cycles
Problem 3 cache capacity = 1048576 bytes
Problem 3 cache bandwidth = 250 bytes/cycle
```

代码必须从 `data/config.txt` 读取参数，不要在算法中重复硬编码。

## 3. 数据规模与图结构判断

100 个 case 的规模大致为：

- Op 数：约 `766～38666`
- Tensor 数：约 `816～40287`
- Tensor-Op 边数：约 `1841～108272`
- 最大图约为 `case_014`

将 Tensor-Op 图收缩成 Op-DAG 后，图结构通常不是简单链，而是“多源输入、宽中层、多个合流点”的重复块结构：

- 宽浅图：拓扑深度小、宽度大，适合独立块并行和 LPT 负载均衡。
- 深窄图：关键路径长、并行度低，适合少切分并尽量将长链放在同一核心。
- 混合图：需要同时考虑通信边、关键路径、片上工作集和高复用 Tensor。

重点代表 case：

```text
宽浅：case_001、case_025、case_058、case_076、case_079
深窄：case_016、case_024、case_051
混合高复用：case_028、case_067、case_072、case_092
```

### 3.1 老师审阅报告中的待复现实验发现

老师对附件的实测审阅提示了几条重要线索：case_001 上块对齐交错方案报告为问题 2 `makespan=58,984`、加速比约 `3.95`、新增搬运 `3,456 bytes`；同一报告中连续等分约 `193,060`。case_074 报告的最佳多核结果仍慢于单核，case_083 报告问题 3 的高 Cache 命中主要来自 spill 换入重复读取。报告也指出有的大图单次评估超过百秒。

同一份审阅统计还报告：100 个 case 中 DDR Tensor 的消费者数为 1；片上 Tensor 最大扇出达 2666；最大 Tensor 为 294,912 bytes，单个 Tensor 没有超过 1 MiB Cache 容量；case_067 的 spill 搬运约 254 MB，约为原图 COPY 的 42 倍。由此要把小图“切分搬运主导”和大图“spill 成本主导”分开诊断，并且问题 3 应优先探查 spill 产生的重复 `COPY_IN`。

这些数字用于确定本地复现优先级和候选策略，不直接作为最终论文结果。正式写入论文前，须在当前工作区的原始 case、当前官方评估器和最终方案文件上重跑，并保存命令、版本、结果 JSON、Trace 及运行时间；若不能复现，应解释差异并以当前复现为准。规模档可先按非 COPY Op 数 `≤1,000`、`1,000～10,000`、`>10,000` 分层，但 spill 档最终还要结合实际 L1/UB 峰值与 `spill_added_copy_bytes`，不能仅凭 Op 数推断。

原始 DDR 输入的消费者数由题面约束，通常是 1；因此原图的边界搬运量基本是常数，真正可以优化的是切图和 spill 产生的 `added_copy_bytes`。片上 Tensor 可以被多个 Op 消费，但这种片上复用不经过问题 3 的 Cache。

问题 3 的 Cache 只服务 `COPY_IN`，收益必须来自同一个逻辑 Tensor 的重复 `COPY_IN`：

- 切图后同一 DDR Tensor 被多个子图或核心分别搬入；
- 大图的 Step2 spill 造成同一 backing 被多次换入；
- 这些重复访问在 FIFO 淘汰前再次发生，从而命中。

因此，问题 3 开始前要先做“命中潜力探针”，统计每个逻辑 Tensor 的 `COPY_IN` 次数和间隔；没有重复访问的 case 不应强行优化 Cache。片上多消费者只能作为问题 2 的同核复用信号，不能直接写成问题 3 的 Cache 收益来源。

新增画像指标：层数、平均宽度、最大宽度、关键路径周期、各 Pipe 总周期、L1/UB 峰值估计、最大 Tensor 大小、spill 风险和重复 `COPY_IN` 数。根据这些指标将 case 分为宽浅、中等、深窄和高 spill 四档，决定候选数量与评估预算。

## 4. 第一阶段：读题、跑通和建立基线

### 4.1 阅读文件

优先阅读：

```text
通用神经网络处理器下的多核调度问题.docx
通用神经网络处理器下的多核调度问题  附件/README.md
通用神经网络处理器下的多核调度问题  附件/docs/核内调度算法.md
通用神经网络处理器下的多核调度问题  附件/docs/多核并行模拟执行算法.md
通用神经网络处理器下的多核调度问题  附件/code/contest_io.py
```

### 4.2 跑通现有评估器

在附件目录执行：

```powershell
cd "D:\Downloads\dive-into-llms-main\A题\通用神经网络处理器下的多核调度问题  附件"

python code/stub_multicore_cut_and_schedule.py data/case_001.json -n 4 --seed 0 --min-subgraph-size 50 --max-subgraph-size 100
python code/multicore_cut_evaluate_problem_1.py data/case_001.json --config data/config.txt
python code/multicore_cut_evaluate_problem_2.py data/case_001.json --config data/config.txt
python code/multicore_cut_evaluate_problem_3.py data/case_001.json --config data/config.txt
python code/singlecore_evaluate.py data/case_001.json --config data/config.txt
```

当前已有的随机方案回归结果约为：

```text
singlecore       233110
problem 1        256220
problem 2        193254
problem 3        189891
problem 3 hits   121 / 624
hit bytes        495616
```

这些数字用于确认评估链路正常，不作为算法性能基线。

### 4.3 建立最小图测试

先复现题面附录 B.7 官方最小输入：

- 输入图：`D:\Downloads\dive-into-llms-main\A题\_probe_b7\minimal_graph.json`
- 调度方案：`D:\Downloads\dive-into-llms-main\A题\_probe_b7\minimal_graph_multicore_res.json`
- 当前工作区实跑结果：问题 1/2/3 均为 `makespan=6`，搬运 `32 bytes`，新增搬运 `0`；问题 3 为 `0/1` 次命中、`0` 命中字节。

这个回归已在本机当前附件上通过。之后还可以准备以下自定义边界测试：

1. 单个 Op。
2. 两个串联 Op。
3. 两个独立 Op。
4. 一个分支和一个合流。
5. 一个内部 Tensor 被多个消费者读取。
6. 一个跨核通信边。
7. 一个会触发 L1 或 UB 换入换出的图。

最小图测试要检查：

- 输出 JSON 格式。
- 所有 Op 是否覆盖。
- 子图依赖是否无环。
- Makespan 是否可以手工计算。
- 评估结束后内存是否归零。
- 是否出现 `deadlock` 或 `time does not advance`。

注意：README 中的官方微型示例数值应以文档为准。若自行构造测试图并给出预期值，要明确标注为“自定义测试图”。

## 5. 图模型与特征提取

### 5.1 收缩为 Op-DAG

对每个 Tensor 建立：

```text
producer[t]  = 生产 Tensor t 的 Op
consumers[t] = 消费 Tensor t 的 Op 集合
```

只要存在 `u -> tensor -> v`，就在 Op-DAG 中加入边 `u -> v`。

记：

\[
G_{\mathrm{op}}=(V,E_{\mathrm{op}})
\]

其中：

\[
(u,v)\in E_{\mathrm{op}}
\iff
\exists t,\ u\rightarrow t\rightarrow v
\]

使用 Kahn 算法或现有 Step1 逻辑得到拓扑序。

### 5.2 基础特征

对每个 Op 计算：

- `cycles`
- 所属 `pipe`
- 入度和出度
- 前驱、后继
- 所在拓扑层
- 关键路径长度
- 输入 Tensor 总字节数
- 输出 Tensor 总字节数
- 与其他 Op 之间的通信权重

对每个 Tensor 计算：

- `size`
- `pos`
- producer
- consumers
- fan-in
- fan-out
- 在给定拓扑序下的首次使用位置
- 最后使用位置
- 生命周期长度
- `size × fan-out`
- 跨子图、跨核心后的潜在搬运量

### 5.3 拓扑深度和宽度

拓扑层可定义为：

\[
level(v)=
1+\max_{u\in Pred(v)}level(u)
\]

无前驱节点的层为 0。

图的深度为最大层数，宽度可用单层最大节点数近似：

\[
W=\max_l |\{v:level(v)=l\}|
\]

深度和宽度用于判断分区策略：

- `W` 大、深度小：优先并行分块。
- 深度大、宽度小：优先保留长链。

### 5.4 关键路径估计

先用计算周期近似（从源向汇递推，`cp(v)` 是 v 的最早完成时刻估计）：

\[
cp(v)=cycles(v)+\max_{u\in Pred(v)}cp(u)
\]

更完整地，可以把 Tensor 搬运估计加入边权。注意下式方向相反，是从汇向源递推的“到终点的剩余最长路”，\(r_i\) 在汇点取值 \(c_i\)；两式一个向前一个向后，排序时任选其一，不要在同一递推中混用：

\[
r_i=c_i+\max_{j\in Succ(i)}
\left(\frac{s_{ij}}{B_{\mathrm{DDR}}}+r_j\right)
\]

该公式只用于排序和预筛，最终时间必须以官方共享带宽仿真为准。

### 5.5 L1/UB 生命周期压力

沿一个候选拓扑序扫描 Tensor 生命周期：

1. Tensor 首次需要时增加占用。
2. Tensor 最后一次使用结束后释放占用。
3. 分别对 `L1` 和 `UB` 求峰值。

如果某个候选块的工作集接近容量，应优先拆分该块，避免 Step2 产生大量换入换出。

## 6. 初始分区方案

分区是本题的核心决策，不能只把拓扑序等分。候选生成器应把“切割边代价、核心负载、收缩图无环、片上容量”同时作为约束或评价指标。每个候选生成后立即调用 `validate_multicore_plan`，非法候选直接丢弃，不进入昂贵评估。

### 6.1 方案 A：单子图基线

将所有非 COPY Op 放入一个子图，分配到核心 0。该方案用于得到单块参考时间、验证核内 Step1/2/3、计算问题 2/3 的零额外切图基线。它不是追求并行度的最终方案，但必须始终保留。

### 6.2 方案 B：拓扑连续分块

按确定性拓扑序连续切块，块内工作量按 `cycles` 和 Pipe 负载计算，而不是只按 Op 数。该方案适合深窄图，或作为所有 case 的稳定保底方案；它不能作为默认最优方案，因为宽浅图上 producer/consumer 可能被切到不同块，造成大量重复搬运。

### 6.3 方案 C：波次和块对齐分区

对宽浅、重复模板图，先识别拓扑层、独立分支组或重复块索引，再尝试以下规则：

- `wave_k`：按连续拓扑层组成波次，再分配到核心；
- `stride_k`：同一分支组的第 `i` 个块分到 `i mod K` 的核心；
- `aligned_block_k`：保持同一重复块的 producer、内部 Op 和 consumer 尽量在同一子图或同一核心。

交错规则必须在每次移动或合并后检查收缩子图图无环；若产生环，立即回退。不能因为原始 Op-DAG 无环就默认分区后的子图图也无环。

### 6.4 方案 D：连通组件与多级图划分

优先识别弱连通组件，独立组件之间可以分别平衡到不同核心。对大型组件，采用多级图划分思想：顶点权为 `cycles`，并按 `PIPE_M/PIPE_V/PIPE_MTE2/PIPE_MTE3` 分开统计；Tensor 边权为 `size(t)`，目标是减少加权 cut，同时保持每核工作量和 L1/UB 峰值接近。

如果环境允许使用 METIS/KaHIP，先用其产生候选，再执行自定义约束修复；如果不引入第三方库，实现“粗化—初始划分—FM 移动细化”的简化版本。标准图划分结果仍必须经过收缩图无环和评估器校验。

### 6.5 目标块数和候选预算

不再机械扫描 `K、2K、4K、8K`。先根据画像分档：

- 宽浅图：优先尝试 `K` 个对齐块、`2K` 个波次/交错块，再补一个细粒度 LPT 候选；
- 深窄图：优先连续分块和少量波次分块，每核 1～2 个主要块；
- 高 spill 图：限制每核活跃工作集，增加连续窗口和组件候选；
- 低规模图：可允许更多候选，由官方评估器选优。

每类候选都必须记录分区规则、块数、合法性结果和预估切割字节，便于论文做消融，而不是只保留最终数字。

## 7. 问题 1：独立 Task 调度

### 7.1 建模重点

问题 1 中每个子图都是独立 Task，子图之间不保留片上数据，跨子图边都经过 DDR。

设子图为 \(q\)，其开始和结束时间为 \(A_q,E_q\)，其核心为 \(core(q)\)。问题一中，分区决定哪些边变成子图边界并产生额外 DDR 搬运；固定分区后，改变核心编号不会改变这些搬运量，只会改变并行关系与等待时间。因此要分别评价切图代价和核分配代价，不能让 DDR 字节数影响一个只改变核心编号的候选排序。

记 `prev(q)` 为 q 在所属核心队列中的前一个 Task，`Pred(q)` 为 q 的依赖前驱，则准确的最早开始时间受以下约束：

\[
A_q=\max\left\{
E_{prev(q)}+100,\;
\max_{p\in Pred(q),\,core(p)=core(q)}E_p,\;
\max_{p\in Pred(q),\,core(p)\ne core(q)}(E_p+1000)
\right\}.
\]

不存在的前驱项忽略。特别地，同核前一个 Task 即使与 q 没有数据依赖，也要等待 100 周期；跨核依赖前驱要等待 1000 周期。子图内部仍由官方核内调度器处理。

与评估器实现逐条核对后的精确口径（`multicore_cut_evaluate_problem_1.py` 的 `task_release_time`）：

- 每核第一个 Task 从时刻 0 起算，无 100 周期等待；之后每个 Task 的释放时刻为“该核上一个完成 Task 的结束时刻 + 100”。核内 Task 严格串行，因此公式中的 `E_prev(q)` 就是该核最近完成 Task 的结束时刻。
- 跨核 1000 周期按**前驱 Task 对去重**计：前驱集合是子图依赖对的集合，同一对子图之间的多条依赖边只产生一次 1000 周期等待。近似评分中的 `N_cross(P)` 应按去重后的 Task 对计数，不能按依赖边计数。
- 同核依赖前驱没有额外等待项；该约束已被核内串行顺序覆盖（同核前驱必在 q 之前完成）。

边界 COPY 的插入粒度决定切图搬运的计价方式（`_build_scene_a_tasks`）：

- `COPY_IN` 按 `(Task, tensor)` 去重：同一 Task 内多个 Op 消费同一外部 tensor 只插入一次 `COPY_IN`，消费方共享同一片上副本。
- `COPY_OUT` 同样每个 `(Task, tensor)` 一次：tensor 被多少个外部 Task 消费不改变 `COPY_OUT` 的数量。

因此对跨越子图边界的 tensor \(t\)，记消费它但不生产它的 Task 数为 \(c_t\)，它是否需要向 Task 外输出（存在 Task 外消费者或属于原图最终输出）为 \(o_t\in\{0,1\}\)，则问题 1 切图搬运可近似计为：

\[
\widehat{Bytes}_1=\sum_{t\ \mathrm{跨边界}} size(t)\cdot(c_t+o_t)
\]

再与原图已有 COPY 字节作差得到 `partition_added_copy_bytes` 的估计。要点是 \(c_t\) 按 Task 去重而不按消费 Op 个数增长：把同一 tensor 的多个消费者聚合到同一子图，搬运量按子图数而非边数下降，这是“聚合高扇出 tensor 消费者”在问题 1 中的定量依据。

问题一的最终优化目标是最小化官方评估器返回的 `Makespan`。切图确定边界搬运量，分核决定并行和等待；两组决策分开搜索、最终通过官方评估器比较。跨子图数据搬运和核心负载差异是解释结果的诊断指标，不应混为一个未经校准的目标。

### 7.2 近似目标函数

最终目标是让官方评估器返回的 `Makespan` 尽可能小。近似评分只用于在调用评估器前快速筛掉明显较差的候选，不能代替正式目标，也不能直接作为论文中的最终成绩。

固定一个切图后，核分配预筛只使用周期量，不加入 DDR 搬运量：

\[
\widehat T_1=\max\left\{\max_{k,p}Load_{k,p},\ \max_{P\in\mathcal P}\left[Cycles(P)+100N_{same}(P)+1000N_{cross}(P)\right]\right\}
\]

其中 `Load_{k,p}` 是分配给核心 `k`、流水线 `p` 的 Op 周期数总和；`\mathcal P` 是收缩后 Task DAG 的路径集合；`Cycles(P)` 是路径上 Task 的估计计算周期；`N_same(P)` 和 `N_cross(P)` 分别是路径上同核队列交接数与跨核依赖等待数。该估计不替代模拟器，也不声称准确预测共享带宽与流水线重叠。固定切图时，DDR 总搬运时间是与分核无关的公共项，可单独报告，不用于给核心编号排序。

该估计只用于同一切图下的分核候选初筛。不同切图之间另行比较 `partition_added_copy_bytes`、预计 spill 和合法性；最终所有指标由官方评估器确认并以其 `Makespan` 排名。

### 7.3 核心分配

对块按 `load` 降序排序，用 LPT 分配到当前估计结束时间最小的核心：

```python
for block in sorted(blocks, key=block_load, reverse=True):
    core = argmin(estimated_core_load)
    assign(block, core)
    update_load(core, block)
```

分配时用预计各流水线负载与关键依赖等待预估候选方案；如果预估完成时间接近，再用预计跨核等待和片上内存风险作次级排序。不要把字节数直接与周期相加：

```text
score(core)
 = max(projected_pipe_load_cycles,
       projected_dependency_path_cycles
       + 100 * projected_same_core_task_handoffs
       + 1000 * projected_cross_core_predecessors)
```

这里的 `score` 仍是启发式估计，候选方案的最终排序由官方评估器决定。

### 7.4 局部搜索

围绕当前最优方案进行：

- 块移动：将一个块移到另一个核心。
- 块交换：交换两个核心上的块。
- 块合并：合并通信量较大的相邻块。
- 块拆分：拆分超大或工作集超限的块。
- 同核顺序交换：保持依赖合法的前提下调整顺序。

每次操作后：

1. 快速检查拓扑和内存约束。
2. 用近似目标函数预筛。
3. 只有有希望的方案才调用官方评估器。
4. 以 Makespan 为第一排序键。

## 8. 问题 2：跨核通信和片上复用

### 8.1 建模重点

问题 2 中每个核心的所有子图合并为一个 Task：

- 同核边可以保留片上数据。
- 跨核边会插入 `COPY_OUT/COPY_IN`。
- 跨核传输增加 500 周期同步延迟。
- `core_schedules` 的子图顺序会直接影响核内 Op 执行顺序：评估器按给定子图顺序给 Step1 序列分桶，桶内保持 Step1 顺序、桶间按子图 rank 排序，再检查新序列是否仍满足拓扑依赖。若顺序造成核内依赖倒置，评估器直接拒绝该方案。

因此，问题 2/3 的子图粒度决定执行顺序可调的分辨率：切得较细时可以更精细地安排核内阶段，但过细会增大方案复杂度和切图搬运；排序只是可控杠杆之一，不能误以为它能任意重排 Op。每次改变 `core_schedules` 后，都要按评估器的分桶规则先做本地拓扑校验。

与评估器核对后的跨核 COPY 规则（`multicore_cut_evaluate_problem_2.py`，问题 3 相同）：

- 插入粒度为每 `(tensor, 源核, 目标核)` 一对：目标核上 \(k\) 个 Op 消费同一 tensor 只插入 1 对 `COPY_OUT/COPY_IN`；同一 tensor 被 \(m\) 个不同核心消费，则源核插 \(m\) 个 `COPY_OUT`、每个消费核各 1 个 `COPY_IN`。
- 原图输入 tensor 按消费核各计 1 个 `COPY_IN`；原图最终输出按生产核各计 1 个 `COPY_OUT`。
- 500 周期同步延迟加在目标 `COPY_IN` 的释放时刻（`源 COPY_OUT 完成时刻 + 500`），每条跨核 link 计一次；该延迟不占用 Pipe 与带宽池。
- `COPY_OUT` 挂在源核**最后**一个生产者子图，`COPY_IN` 挂在目标核**最早**一个消费者子图；核内子图顺序因此会决定 COPY 操作在时间轴上的位置。
- 合并 Task 从时刻 0 开始参与调度，**没有问题 1 的 100/1000 周期 Task 等待**；问题 2/3 中唯一的同步等待就是跨核 `COPY_IN` 的 500 周期。不要把问题 1 的激活公式套进问题 2/3 的近似评分。

因此对跨核 tensor \(t\)，记消费它的远端核心数为 \(m_t\)，近似代价应按 `(tensor, 源核, 目标核)` 三元组计价，而不是按 Op 依赖边计价：

\[
cost_2(t)=m_t\cdot\left(500+\frac{size(t)}{60}\right)
\]

聚合消费者的收益同样按 \(m_t\) 衡量：把一个远端核上的全部消费者并回 producer 所在核，收益为 \(500+size(t)/60\)；只迁走部分消费者时 \(m_t\) 不变、收益为零。这个“全有或全无”的计价结构是问题 2 分区搜索与问题 1 最大的差别。

### 8.2 优化重点

问题 2 的核心不是单纯均匀切图，而是：

- 尽量将高权 Tensor 的 producer 和 consumers 放在同一核心。
- 尽量减少关键路径上的跨核边。
- 让核心负载接近。
- 控制每个核心的 L1/UB 活跃工作集。
- 平衡四条 Pipe 的瓶颈。

高复用 Tensor 的权重：

\[
reuse\_weight(t)=size(t)\cdot(fanout(t)-1)
\]

如果一个 Tensor 的多个消费者分散到多个核心，就会产生更多跨核访问，应优先把这些消费者聚合到一个核心。

### 8.3 问题 2 的候选筛选

问题二沿用“先做可行性检查、再估计完成时间”的流程。候选核心的评分不能把周期、字节数和风险分数直接相加；应先把可估计的计算、搬运和依赖等待都换算为周期，按第 7.2 节的方式预筛。跨核字节数、L1/UB 峰值和 spill 预计量用于解释或打破相近候选的平手，最终仍按官方评估器的 `Makespan` 排序。

### 8.4 问题 2 的局部搜索

从问题 1 最优方案或拓扑连续分块方案开始，重复进行：

1. 找出跨核字节数最大的 Tensor。
2. 尝试把其消费者移动到 producer 所在核心。
3. 如果导致核心严重不平衡，则尝试交换其他块。
4. 如果 L1/UB 超限，则拆分块或撤销移动。
5. 用官方评估器比较 Makespan。

问题 2 中不宜把所有 Op 粗暴合并成一个巨大子图，因为过长驻留会造成 Step2 大量 spill。建议使用滑动窗口约束：

\[
Peak_{L1}(k,window)\le 0.8L1
\]

\[
Peak_{UB}(k,window)\le 0.8UB
\]

0.8 是启发式安全系数，最终仍以评估器结果为准。

## 9. 问题 3：FIFO Cache 感知调度

### 9.1 建模重点

问题 3 在问题 2 的基础上增加共享只读 FIFO Cache：

- Cache 容量约为 `1 MB`。
- Cache 带宽为 `250 bytes/cycle`。
- 只有 `COPY_IN` 可以命中。
- 命中时不占用 DDR。
- FIFO 按访问顺序淘汰。

对一次 `COPY_IN` 访问 \(r\)，定义：

\[
h_r=
\begin{cases}
1,& \text{Cache 命中}\\
0,& \text{Cache 未命中}
\end{cases}
\]

命中时传输时间近似为：

\[
\frac{s_r}{250}
\]

未命中时传输时间近似为：

\[
\frac{s_r}{60}
\]

这两个近似都假设独占带宽。评估器中 DDR 与 `CACHE_READ` 是两个**全局共享**带宽池（所有核心共用一个池，池内按在途搬运数公平分摊），并发为 \(n\) 时每次搬运实际只得到 \(1/n\) 带宽。Cache 命中的真实收益是“改用独立的 250 池、不再挤占 60 的 DDR 池”，而不只是单次传输变快；近似评分可用独占带宽给候选排序，但论文解释收益时必须按共享池口径说明。

### 9.2 Cache 感知目标

问题三最终也按官方评估器返回的 `Makespan` 比较方案。Cache 命中字节数能说明数据复用是否增强，但它和周期不是同一单位，不能不经换算就放入加权和，也不能假设命中率提高一定缩短总时间。只有在准确复现 FIFO 的全局访问顺序后，才能把估计的 Cache 命中用于周期预估；否则它只用于挑选值得送入官方评估器的候选。

开始优化前先做 Cache 命中潜力探针：按逻辑 Tensor id 统计评估器中 `COPY_IN` 访问次数、字节数和访问间隔，并区分切图重复读取与 Step2 spill 换入。若某 case 几乎没有重复 `COPY_IN`，就复用问题 2 方案并记录 Cache 对照，不浪费搜索预算。

只有同一逻辑 Tensor 的重复 `COPY_IN` 才有命中机会。高扇出片上 Tensor 可用于问题 2 的同核复用分组，但不能直接当成 Cache 收益。若多次访问在 FIFO 淘汰前再次发生，才将其列为高优先级；超过 Cache 容量的 Tensor 不进入 Cache。

### 9.3 FIFO 模拟

对每次逻辑 Tensor 的 `COPY_IN` 访问按**完成时刻**顺序送入 FIFO。评估器在 `COPY_IN` 完成（retire）时才把未命中项写入 Cache，因此 FIFO 插入顺序是完成事件顺序，不是发射顺序；本地模拟必须从问题 3 的 Trace/结果中取完成事件排序，不能按发射时刻或计划顺序扫描：

```python
cache = deque()
used_bytes = 0

for access in copy_in_order:
    if access.tensor_id in cache:
        hit_bytes += access.size
        hits += 1
    else:
        misses += 1
        if access.size <= CACHE_CAPACITY:
            while used_bytes + access.size > CACHE_CAPACITY:
                old = cache.popleft()
                used_bytes -= old.size
            cache.append(access)
            used_bytes += access.size
```

实际实现必须以评估器的逻辑 Tensor 标识和 FIFO 行为为准，不能仅用 Tensor 大小判断命中。Cache hit 不更新 FIFO 入队顺序；多核 `COPY_IN` 访问可能交错，单纯按某个核心的计划顺序扫描不等于全局事件顺序。初期以问题 3 评估器的 `cache_stats` 和 Trace 为准，只有复现相同访问事件顺序后，本地 FIFO 模拟才可用于预筛。

与代码核对后的补充口径：

- 插入发生在完成时刻，所以“FIFO 窗口”是完成时刻轴上的窗口，两次访问的间隔应按完成时刻差估计。
- `COPY_OUT` 完全不接触 Cache：不查询、不插入、不失效；不要设计“写回刷新 Cache”之类的策略。
- spill 换入的 `COPY_IN` 与切图重复读取**共享同一 Cache 键空间**（都是原图逻辑 tensor id），两类访问会互相挤占 FIFO 位置；spill 换出的 tensor 若原本来自原图 `COPY_IN`，则不生成新 `COPY_OUT`，换入时直接重读原 DDR tensor，这类 spill 的搬运成本只有换入一份。
- 官方 `cache_stats.hit_rate` 按**字节**计算（`hit_bytes/(hit_bytes+miss_bytes)`），不是按次数；论文引用时不要写成次数命中率。
- 官方结果还包含 `cache_events`（插入/淘汰事件）、`cache_final_entries`、`cache_used_bytes_final`；时间线条目带 `memory_path`（`DDR`/`CACHE_READ`/`ON_CHIP`）、`cache_hit`、`cache_tensor_id`，可直接用于核对本地 FIFO 模拟。

还要注意，多核上的 `COPY_IN` 可能交错发生。仅按 `core_schedules` 列表顺序串行扫描，不一定等于官方模拟器的真实全局访问顺序。初期应直接用问题 3 评估器产生的 Cache 统计和事件时间线作准；只有在确认自己复现了评估器的访问事件顺序后，才用本地 FIFO 程序预筛候选。

### 9.4 问题 3 的优化步骤

1. 读取问题 2 的最优方案，并运行命中潜力探针。
2. 将重复读取分成切图重复读取和 spill 换入两类，统计 Tensor id、字节数和访问间隔。
3. 只对确有重复访问的 case 尝试调整分区与核内子图顺序，使相同 id 的访问更可能落在 FIFO 窗口内。
4. 对高 spill case 单独比较“减少 spill”与“利用 spill 的重复换入命中”，以 Makespan 和额外搬运共同解释结果。
5. 每个候选仍由问题 3 官方评估器决定；不假设问题 3 一定快于问题 2。

Cache 带宽更高通常会带来问题 3 的加速，但不应把 `T3 <= T2` 写成硬约束。FIFO 淘汰顺序、额外访问和核心负载都可能导致个别 case 出现例外。

## 10. 推荐代码结构

在附件的 `code` 目录新增：

```text
solver_common.py
graph_features.py
case_profile.py
initial_partition.py
partition_search.py
core_assignment.py
local_search.py
evaluation_budget.py
solve_all.py
batch_experiment.py
```

建议接口：

```python
build_op_dag(graph)
extract_features(graph, dag)
make_initial_blocks(graph, dag, target_blocks)
assign_problem_1(blocks, features, num_cores)
assign_problem_2(blocks, features, num_cores)
refine_plan(graph, plan, problem, num_cores)
evaluate_plan(graph, plan, problem)
validate_plan(graph, plan, num_cores)
```

推荐模块职责：

### `solver_common.py`

- JSON 读取。
- 配置读取。
- 方案 JSON 生成。
- 统一日志。
- 结果字段提取。

### `graph_features.py`

- Tensor 到 producer/consumers 的索引。
- Op-DAG 构造。
- 拓扑排序。
- 深度、宽度、关键路径。
- Tensor 生命周期。
- fan-out、复用权重。
- L1/UB 工作集估计。

### `case_profile.py`

- 输出每个 case 的层数、平均/最大宽度、关键路径周期、各 Pipe 周期、L1/UB 峰值估计、最大 Tensor 和 spill 诊断指标。
- 额外扫描 `COPY_IN` 访问，按逻辑 Tensor id 统计重复次数和访问间隔，形成 Cache 命中潜力画像。

### `partition_search.py`

- 生成连续、wave、stride/块对齐、连通组件、多级加权图划分候选。
- 对每次交错分区或 FM move 增量检查收缩子图 DAG 是否成环；有环立即撤销。
- 对候选计算按 Pipe 分开的顶点负载、加权 cut、每核 L1/UB 压力，并把所有诊断写入候选记录。

### `initial_partition.py`

- 单子图方案。
- 拓扑连续分块。
- 连通组件分块。
- 目标块数生成。
- 高权 Tensor 边合并。

### `core_assignment.py`

- LPT。
- HEFT 风格分配。
- 关键路径优先。
- 通信惩罚。
- 内存压力惩罚。

### `local_search.py`

- move、swap、merge、split。
- 合法性检查。
- 候选缓存。
- 无改善停止。

### `solve_all.py`

- 单 case 求解。
- 生成问题 1、2、3 方案。
- 输出方案 JSON 和结果 JSON。

### `batch_experiment.py`

- 批量跑 100 个 case。
- 批量跑 2～5 核。
- 收集运行时间和失败信息。
- 输出 CSV/JSON 汇总。

### `evaluation_budget.py`

- 对每个 `(case, 核数, 问题, 方案哈希)` 缓存官方评估结果，避免重复运行。
- 根据 Op 数分档限制昂贵评估调用；先近似排序，再对 top-k 做精确回放。
- 支持多进程并行独立 case，记录每次运行耗时、超时和失败原因。

## 11. 方案合法性检查

每个候选方案在调用官方评估器前都执行：

```text
1. JSON schema 正确：`node_to_subgraph` 值为非负整数，`core_schedules` 为长度等于核数的二维整数数组。
2. `node_to_subgraph` 是否覆盖所有非 COPY Op，键是否为十进制 Op id 字符串。
3. 是否误将原始 COPY 节点放入 mapping。
4. 子图 id 是否唯一且建议连续；每个 Op 是否只属于一个子图。
5. 每个子图是否在某个核心顺序中恰好出现一次；空核是否显式写 `[]`。
6. 收缩子图依赖图是否无环；这是候选生成阶段的硬约束，不是事后统计项。
7. 每个核心子图顺序是否满足直接依赖；对问题 2/3，按评估器分桶规则（桶内保持 Step1 顺序的稳定排序）重排后的 Op 序列是否仍满足拓扑依赖。注意问题 2/3 实际是三道独立检查：子图级同核顺序检查、分桶重排后的 Op 级拓扑检查、以及包含 Pipe FIFO 顺序与内存复用补边（WAR/WAW）的全局 Kahn 判环；即使前两项通过，Pipe 顺序与补边组合出的等待环也会被评估器拒绝（报 `dependency cycle`），本地预检应复现第三道检查。
8. 官方 `validate_multicore_plan` 是否通过，core 数是否与 `core_schedules` 长度一致。
9. 评估后是否出现 deadlock、time does not advance、异常结束或内存未释放。
```

每次修改方案 JSON 生成器后，先运行题面附录 B.7 最小示例，三个评估器都应得到 `makespan=6`、原始/调度搬运量 `32 bytes`、新增搬运量 `0`，问题 3 Cache 命中数为 `0`。这是强制回归门禁，之后再跑 `case_001`。

注意原始 DDR 边界 COPY 的语义：原始 `COPY_IN/COPY_OUT` 不应当作为普通 Op 加入子图映射，否则可能改变评估器自动补充边界 COPY 的行为。

## 12. 官方评估与结果字段

官方评估器是最终的精确模型。HEFT、局部搜索和生命周期估计只能用于生成和筛选候选方案。

建议统一保存以下字段：

```text
case_id
num_cores
problem
makespan
singlecore_makespan
speedup
original_graph_copy_bytes
scheduled_copy_bytes
added_copy_bytes
partition_added_copy_bytes
spill_added_copy_bytes
cross_core_transfers
memory_peak_by_core
step3_by_core.local_makespan
step3_by_core.memory_dependency_count
step3_by_core.pipe_op_counts
estimated_pipe_cycle_loads  # 自己从分配到各核的原图 Op cycles 汇总，不是官方原生字段
cache_accesses
cache_hits
cache_hit_bytes
cache_miss_bytes
cache_hit_rate  # 官方口径为字节命中率 hit_bytes/(hit_bytes+miss_bytes)，非次数命中率
cache_events / cache_final_entries / cache_used_bytes_final  # 问题 3 的 FIFO 插入/淘汰事件与最终驻留
memory_path / cache_hit / cache_tensor_id  # 问题 3 时间线条目字段，用于核对本地 FIFO 模拟
runtime_seconds
status
error_message
```

问题 3 的典型字段位于：

```text
cache_stats.copy_in_hits
cache_stats.copy_in_misses
cache_stats.hit_bytes
cache_stats.miss_bytes
cache_stats.hits
cache_stats.accesses
cache_stats.hit_rate
```

官方 `step3_by_core[*].pipe_op_counts` 是各 Pipe 的 Op **数量**，不是 Pipe 周期负载。若论文要报告 Pipe 周期，应从图数据和核分配自行按 Pipe 累加 `cycles`，或从 Trace 统计事件时长，并明确标成自算字段。不可把 `PIPE_M_load` 等不存在的键假装成评估器输出。

## 13. 实验设计

### 13.1 核心数

对每个 case 测试：

```text
1、2、3、4、5 核
```

单核结果由 `singlecore_evaluate.py` 提供，作为加速比基线。

### 13.2 加速比

\[
S_p(K)=
\frac{T_{\mathrm{single}}}{T_p(K)}
\]

问题 3 相对于问题 2 的 Cache 加速比：

\[
S_{\mathrm{cache}}(K)=
\frac{T_{\mathrm{problem2}}(K)}
{T_{\mathrm{problem3}}(K)}
\]

### 13.3 统计方式

不要只报告算术平均。建议同时报告：

- 算术平均
- 中位数
- 几何平均
- p90
- 失败数
- 平均求解时间

超大 case 可能支配算术平均，因此中位数和几何平均很重要。

先逐 case 计算加速比，再对 case 级加速比取几何平均：

\[
G(K)=\exp\left(\frac1N\sum_{i=1}^{N}\log\frac{T_{single,i}}{T_{p,i}(K)}\right).
\]

不要用 `sum(T_single)/sum(T_p)` 代替，因为超大 case 会支配整体结果。需要同时报告慢于单核的 case 数，并如实展示负加速样例。

### 13.4 下界与达成差距

对每个 case 计算轻量乐观下界：关键路径周期 `CP`、最大 Pipe 总周期除以核心数、以及原始 DDR 总搬运字节除以带宽，取三者最大值作为诊断基准：

\[
LB(K)=\max\left(CP,\frac{\max_p\sum_i cycles_{i,p}}{K},\frac{DDRBytes}{B_{DDR}}\right).
\]

报告 `T_eval/LB` 作为“距乐观下界的比值”，越接近 1 越好；它不是最优性证明，也不是官方评分。图表应按层数/宽度分组，展示分区策略效果为什么随图结构改变。

### 13.5 分组分析

按图结构分组：

```text
宽浅图
深窄图
混合图
高复用图
高 L1 压力图
高 UB 压力图
```

建议图表：

1. 不同核心数的 Makespan 曲线。
2. 不同问题的加速比柱状图。
3. 跨核字节数与 Makespan 的散点图。
4. Cache 命中率和命中字节数。
5. L1/UB 峰值箱线图。
6. 四条 Pipe 的负载分布。
7. 代表性 case 的任务甘特图。
8. 高权 Tensor 的跨核通信热力图。

### 13.6 官方评估调用预算

单次评估耗时可能从不足 1 秒到数十秒甚至更久，必须做预算和缓存。初步上限可按规模设为：≤1,000 个非 COPY Op 最多 200 次候选评估，1,000～10,000 个最多 20 次，>10,000 个最多 5 次；这是老师给出的预算起点，先用代表性 case 测量耗时，再结合总机时调低。执行时先用近似评分筛选 top-k，按方案 JSON 哈希复用已算结果，独立 case 多进程并行，并设置每 case 的墙钟上限。最终阶段优先冻结一组可提交结果，再逐步扩展到 100 case 和 1～5 核；避免在超大图上无预算地调用局部搜索。

## 14. 七天执行计划

### Day 1：读题、统计和基线

- 阅读 README、核内调度文档和多核模拟文档。
- 先用官方 B.7 最小例子跑问题 1/2/3，核对 Makespan=6、COPY=32 bytes、新增搬运=0、Cache 命中=0。
- 扫描 100 个 case 的 Op、Tensor、边数、Pipe、pos、层数/宽度、关键路径和 L1/UB 压力，生成 `case_profile.csv`。
- 实现 `build_op_dag`。
- 跑通 `singlecore_evaluate.py` 和 case_001 的三个评估器。
- 测量不同规模 case 的单次评估用时，形成分档预算。

交付物：

```text
graph_features.py 初版
case_summary.csv
case_profile.csv
B.7 三问题回归结果
官方评估器回归结果
```

### Day 2：初始分区与问题 1

- 实现连续、wave、块对齐/stride、连通组件等结构化分区。
- 实现收缩子图 DAG 检查，交错候选成环时立即回退。
- 实现按 Pipe 负载统计的 LPT/HEFT 初始核心分配。
- 先在 case_001、010、026、051、074、083 等宽浅/深窄/高 spill 代表图做规则对照。
- 所有候选先跑合法性过滤，再按规模预算调用官方评估器。

交付物：

```text
initial_partition.py
core_assignment.py
problem1_initial.py
problem1 初始方案和结果：day2_results/problem1_summary.csv
```

Day 2 已完成首轮代表 case 实验：`case_001、010、026、051、074、083` 共 30 个候选（5 类分区规则 × 6 个 case）全部通过官方合法性校验和问题 1 评估。候选方案、元数据及完整评估结果保存在 `通用神经网络处理器下的多核调度问题  附件/day2_results/`。首轮结果显示，单子图方案可作为零切图基线；连续、wave、块对齐和连通组件方案均已接通官方评估器，后续 Day 3 应以 `problem1_summary.csv` 中的 `makespan` 和额外搬运共同筛选局部搜索起点，不能只按近似分数排序。

复现实验命令：

```powershell
Set-Location -LiteralPath 'D:\Downloads\dive-into-llms-main\A题\通用神经网络处理器下的多核调度问题  附件'
$env:PYTHONPATH='code'
python code/problem1_initial.py --data-dir data --output-dir day2_results --num-cores 4
```

### Day 3：问题 1 局部搜索

- 实现 move、swap、merge、split。
- 增加方案合法性检查。
- 增加官方结果缓存。
- 先验证小图和 case_001，再扩展到代表性 case。

交付物：

```text
local_search.py 初版
local_search_batch.py
problem1 最优方案
问题1多起点对比表
```

Day 3 已完成首轮局部搜索回放：`case_001` 从连续分块起点开始，20 次预算内保持 `58984` cycles；`case_010` 从连通组件起点开始，经一次 `move` 改善到 `29402` cycles。搜索过程使用官方评估器缓存，完整方案、结果和历史记录位于 `通用神经网络处理器下的多核调度问题  附件/day3_results/`，汇总为 `day3_results/local_search_summary.csv`。两个最优方案均通过非 COPY Op 覆盖、子图依赖无环和核心调度覆盖校验。

复现实验命令：

```powershell
$env:PYTHONPATH='code'
python code/local_search_batch.py --cases case_001 case_010 --data-dir data --input-dir day2_results --output-dir day3_results --max-rounds 2 --max-evaluations 20 --proposal-limit 24
```

### Day 4：问题 2

- 单独生成问题 2 候选；可把问题 1 解作为候选之一，但不假设问题 1 的最优切图适用于问题 2。
- 加入跨核通信和 500 周期惩罚。
- 聚合高复用 Tensor 的消费者。
- 加入 L1/UB 工作集滑窗约束。
- 批量跑代表性 case 和 2～5 核。

交付物：

```text
problem2 求解器
跨核通信统计
memory_peak 统计
问题2最优方案
```

### Day 5：问题 3

- 对问题 2 结果做 `COPY_IN` 逻辑 Tensor 重复访问探针，区分切图搬运和 spill 换入。
- 只对有重复访问的 case 开展问题 3 调整；检查相同 id 的重复访问能否留在 FIFO 窗口内。
- 不把片上 Tensor 的普通多消费者直接记成 Cache 收益。
- 用官方问题 3 评估器确认命中率和 Makespan。

交付物：

```text
problem3 求解器
cache_stats 汇总
问题2/问题3对比表
```

### Day 6：批量验证和作图

- 按预算跑完 100 个 case、1～5 核、三个问题；缓存评估结果并行运行独立 case。
- 汇总失败方案和评估器错误。
- 独立回放合法性检查。
- 检查 Pipe FIFO、内存峰值、跨核传输和结果 JSON。
- 绘制甘特图、加速比曲线、Cache 命中图和通信图。

交付物：

```text
batch_results.csv
batch_results.json
paper_figures/
failure_cases.csv
```

### Day 7：论文与冻结结果

- 整理符号、约束、目标函数。
- 写三题递进模型。
- 写算法伪代码和复杂度。
- 完成结果、消融和敏感性分析。
- 固定最终方案和结果文件。
- 复核所有表格中的 case 数、核心数和指标字段。

## 15. 论文写作结构

第 23 届 Word 模板给出的核心写作要求是“事实—模型—结果—检验—结论”证据链，并建议摘要写清问题、方法、结果、验证和应用价值。2025 年 A 题优秀论文与本题同属 NPU 调度方向，可参考它逐问拆解、给出伪代码和复杂度、再用定量表格和曲线分析结果的组织方式；它解决的是另一版核内子问题，不能直接照搬其模型或数字。建议把今年题目的分区、分核、排序决策作为主线，按模板适配为：

### 摘要与关键词

摘要按“三问分别写方法与可复核结果”的顺序组织，给出代表性加速比、额外搬运、Cache 命中和验证口径。所有数字从最终汇总 CSV 自动生成，结果未跑完时不提前写占位结论。

### 第一章：赛题理解与任务定义

- 题目背景、输入输出和每问要优化什么。
- 把切图、核心分配、每核顺序分别定义为决策。
- 明确 Makespan 为评价主指标，搬运、spill、Cache 和内存为解释指标。

### 第二章：数据画像与统一建模框架

- case 数量、图规模、层数/宽度和 Pipe 周期统计。
- 原图 COPY、DDR 输入 fan-out、Tensor size 和片上内存压力。
- 画出“画像—候选分区—可行性过滤—官方评估—局部改进”的流程图。

### 第三章：公共图模型、约束和下界

- Tensor-Op 图收缩为 Op-DAG 的方法。
- 加权 cut、按 Pipe 统计的核心负载、收缩子图无环约束。
- L1/UB 容量、Task 等待、跨核同步和 FIFO 访问规则。
- 关键路径、Pipe 工作量和 DDR 搬运下界，并说明它们是乐观基准。

### 第四章：问题一的切图与 Task 调度

- 切图决定额外搬运、分核决定并行和 100/1000 周期等待的分离建模。
- 连续、波次、块对齐/交错等候选规则及其无环过滤。
- 同核前一 Task 固定 100 周期的激活公式。
- 固定分区下的 LPT/关键路径分核与评估器闭环。

### 第五章：问题二的跨核通信优化

- 每核合并 Task 后跨核边如何产生 COPY 和搬运成本。
- `core_schedules` 顺序如何影响 Step1 序列分桶与核内依赖合法性。
- 多级加权 cut、负载平衡、片上复用和 spill 风险。
- 问题一方案、单子图方案与问题二最终方案的对比。

### 第六章：问题三的 FIFO Cache 优化

- Cache 仅服务 `COPY_IN` 的访问机制。
- 重复逻辑 Tensor 访问探针，分离切图读取与 spill 换入。
- 命中潜力、FIFO 淘汰和 Cache 带宽的作用。
- 用官方 `cache_stats` 验证命中是否转化为 Makespan 改善。

### 第七章：算法实现与复杂度

- 多候选生成、收缩 DAG 增量无环检查、FM 局部细化。
- 近似预筛、官方评估缓存、并行批处理和规模分档预算。
- 分别给出每个关键算法伪代码和复杂度；区分自己实现的近似算法与官方模拟器。

### 第八章：实验设计、结果和消融

- 1～5 核、单核基线、分区规则对照、问题 1/2/3 递进结果。
- 每 case 加速比的几何平均、中位数、p90、负加速比例、下界比值。
- 按宽浅/中等/深窄/高 spill 分组；报告 Makespan、搬运拆分、内存峰值、Cache 统计和评估耗时。
- 消融至少比较连续分块、对齐/波次切分、多级图划分、加入 DAG 无环修复、加入 Cache 探针前后。

### 第九章：稳健性、模型评价和局限

- 改变核心数、分区候选预算和关键启发式参数，检查排序是否稳定。
- 说明多核不保证加速，单独解释负加速 case。
- 说明启发式非全局最优、超大 case 评估耗时和 FIFO 命中顺序等局限。

### 第十章：结论

- 用数字回答三问；明确哪些结构适合并行切分、哪些深链适合保留局部性、Cache 在哪些重复访问场景有用。
- 附参考文献、复现命令、参数配置和核心代码；附录只放必要代码，不让长代码挤占正文论证。

每张图表都要在正文引用并解释，结果段落按“数字—与基线比较—原因—含义”写。图表中的每个数必须能回溯到结果 JSON 或汇总脚本。

## 16. 最终验收清单

提交前逐项确认：

- [ ] 100 个 case 均完成统计。
- [ ] 2～5 核均有结果。
- [ ] 问题 1、2、3 均生成合法方案。
- [ ] 所有非 COPY Op 恰好覆盖。
- [ ] 所有子图恰好调度一次。
- [ ] 无环、无死锁、无时间不推进。
- [ ] L1/UB 峰值字段完整。
- [ ] 问题 3 的 Cache 字段完整。
- [ ] 官方 B.7 最小图三个问题均通过，输出为 6 cycles / 32 bytes / 0 added bytes。
- [ ] 单核基线和 speedup 计算一致。
- [ ] 结果文件没有覆盖原始输入。
- [ ] 代码可以从干净目录重新运行。
- [ ] 论文中的表格可以由结果文件复现。
- [ ] 论文中没有把近似估计误写成官方精确结果。

## 17. 推荐的最终运行入口

最终建议统一使用一个入口：

```powershell
python code/solve_all.py `
  --input-dir data `
  --output-dir results `
  --cores 1 2 3 4 5 `
  --problems 1 2 3 `
  --config data/config.txt `
  --seed 2026
```

批量实验：

```powershell
python code/batch_experiment.py `
  --input-dir data `
  --result-dir results `
  --summary results/summary.csv `
  --cores 1 2 3 4 5 `
  --problems 1 2 3
```

最终原则是：启发式算法负责快速提出候选方案，官方评估器负责给出最终成绩；论文中的结论只引用官方评估结果，近似目标函数和生命周期估计用于解释算法行为和指导搜索。

## 18. 参考优秀论文和模板后的复盘与升级

### 18.1 哪些经验适用于本题

最值得参考的是 2025 年 A 题《通用神经网络处理器下的核内调度问题》：它与本题属于同一类 NPU 调度问题，论文按“问题分析—数学模型—算法步骤—算例结果”逐问展开。它处理的是核内缓存驻留、缓存分配和 SPILL；本题进一步增加了子图切分、核心映射、跨核通信和多核并行模拟。因此可以借用它把模型和代码步骤对应起来的写法，不能直接搬用它的单核模型、参数或数值结果。

2023 年 A 题把理论计算与离散事件仿真结果相互核对，2024 年 A 题展示了多种方案对照、多个指标和结果解释。这些经验适合转成当前题目的验证方法：让每个结论都能追溯到官方评估器输出、对照方案和代码运行记录。其他年份及 B–F 题涉及的模型差异很大，适合借鉴表达和论证结构，不应把无关领域的模型直接移植到 NPU 调度题。

### 18.2 把“精确模型”和“搜索代理”分开

论文中的正式模型应明确写成：决策包括 Op 到子图的归属、子图到核心的归属、各核心内子图的执行顺序。令 `x_{u,g}=1` 表示普通 Op `u` 属于子图 `g`，令 `y_{g,k}=1` 表示子图 `g` 分配给核心 `k`，则基本覆盖关系为：

\[
\sum_g x_{u,g}=1\quad(\forall u),\qquad
\sum_k y_{g,k}=1\quad(\forall g)
\]

此外，原始 COPY 按题目规则处理，子图依赖和核心内顺序必须合法，核心编号和内存规则必须满足评估器要求。对问题 `q`，以官方模拟器定义完成时间：

\[
\min_{\mathcal S} T_q(\mathcal S),\qquad
T_q(\mathcal S)=\operatorname{Evaluator}_q(G,\mathcal S)
\]

其中 `\mathcal S` 表示完整排班方案。这样写清楚了“我们到底在优化什么”和“约束从哪里来”；复杂的共享带宽、流水线、等待、缓存行为则由给定的官方评估器精确计算。前面的 `\widehat T_1`、通信权重、复用分数属于搜索代理或启发式，不要把它们误称为官方 Makespan 的精确公式。

### 18.3 先做可信基线，再逐步加功能

当前记录的 `stub_multicore_cut_and_schedule.py` 是随机方案，运行成功只说明输入、输出和评估器链路基本打通。它不是强基线：在 `case_001` 上，随机问题一结果 `256220` 周期还高于单核 `233110` 周期，这说明随意切块和分配未必能加速。

建议用同一评估器建立逐级对照：

1. 单核、单子图：检查核内流程和结果格式。
2. 简单多核：确定性拓扑排序后按工作量切块，再用 LPT 分配。
3. 通信感知：在简单多核方案上减少关键路径和大 Tensor 的跨核传输。
4. Cache 感知：从问题二较好方案开始，调整高复用 Tensor 的跨核读取顺序。
5. 完整搜索：加入 move、swap、merge、split，并与前四级方案比较。

每一级都保存方案 JSON、评估器原始结果、运行参数和随机种子。这样才能说明负载均衡、通信感知、Cache 感知和局部搜索各自带来了多少变化，而不只是报告一个最终数字。

### 18.4 补上公平、稳定的实验设计

先在宽浅、深窄、混合三类代表 case 上调通流程，再扩大到全体 case。参数选择阶段固定一组开发 case；报告阶段使用没有参与参数选择的 case 检验效果，最后再对全部 100 个 case 运行冻结版本。带随机性的搜索应固定种子，并在代表 case 上使用多个种子，至少报告中位数和最好、最差结果。

最终结果表以 `Makespan` 和相对同核心数基线的加速比为主，同时列出额外搬运、跨核传输、L1/UB 峰值、Cache 命中字节数、运行时间和失败状态。总体报告中位数、几何平均、p90 和最差 case；分组解释宽浅、深窄、混合图上的得失。再做消融实验，例如分别关闭通信惩罚、内存风险惩罚、Cache 感知和局部搜索。若某些 case 变慢或搬运变多，应如实展示并解释原因。

### 18.5 按 Word 模板改写论文，不照抄模板示例

模板强调“问题定义—模型—结果—检验—结论”的证据链、逐问结果、基线比较、敏感性分析和图表解释。对本题可整理成：摘要；赛题理解和符号；输入图与评估器；问题一模型和算法；问题二模型和算法；问题三模型和算法；实验、基线和消融；模型评价；结论；参考文献和代码附录。模板中关于回归、预测、分类和一般数据清洗的示例不适合本题，应换成图结构统计、拓扑合法性、内存峰值、通信量和调度甘特图。

每段结果建议按“数值—对比—原因—意义”写。例如：“4 核下该策略使中位 Makespan 比拓扑分块基线下降 X%；收益主要来自减少关键路径上的跨核等待；但深窄组改善较小，因此该策略更适合宽浅图。”图表应由程序结果自动生成，正文、摘要和结论中的数字统一从汇总表读取。

### 18.6 写作时重点防止的错误

2025 年 A 题论文虽然与本题最接近，但摘要中 `Matmul_Case0` 算例名称有重复，问题一表格列出的 7660 与正文描述的 7760 也不一致。参考它的结构即可，所有算法、数据和结论都应从本题附件和自己运行的评估器重新核对。尤其避免以下问题：

- 把启发式近似分数当成最终目标，或把不同单位的指标不加说明地相加。
- 只报告最优的几个 case，不报告退化、失败和运行时间。
- 没有基线、消融和固定种子，却声称某个模块“显著有效”。
- 把命中率提高直接等同于 Makespan 一定下降。
- 只写“算法先进、效果良好”，不给可复算的数字、图表和评估器输出。
- 把往年论文里的算法、参数或结果直接移植到今年的多核评估语义中。

获奖无法由模板或算法名称保证；最能提高论文可信度的，是一套可复现的方案生成程序、严格合法性检查、覆盖不同图结构的对照实验，以及与代码完全一致的数字和解释。

### 18.7 老师审阅意见落实表

| 审阅意见 | 已落实位置或动作 |
|---|---|
| 方案 JSON 类型不符合 B.5 | §2.2 已改为整数 sgid 和二维整数 `core_schedules`；新增 `_probe_b7` 官方回归文件 |
| 问题一搬运与分核目标混淆 | §7.1～7.3 拆开切图搬运与固定切图下的核心分配评分 |
| 问题 2/3 顺序控制被遗漏 | §8.1 写明按子图 rank 分桶重排和依赖校验 |
| 连续分块不能作为默认最优 | §6 改为结构化候选；宽浅、深窄和高 spill 图分别选策略 |
| 交错分区可能收缩成环 | §6、§10、§11 增加候选生成阶段的无环硬过滤 |
| Cache 收益来源误判 | §3、§9 改为只统计重复逻辑 Tensor 的 `COPY_IN`，并区分 spill 换入 |
| 缓存压力随规模变化 | §3.1 加入 spill 诊断分档，并要求结合峰值与额外搬运确认 |
| 评估器调用可能超预算 | §10、§13.6 新增哈希缓存、top-k、并行和分档上限 |
| 缺少官方最小示例 | §4.3、§11 固定 B.7 为回归门禁；当前三个评估器实跑通过 |
| Pipe 字段名错误 | §12 去掉虚构原生负载字段，改用真实 `step3_by_core` 字段并区分自算周期 |
| 问题一漏同核前序等待 | §7.1 纳入无依赖同核前序 Task 的 100 周期等待 |

### 18.8 本轮对照官方代码的模型修正清单

以当前附件代码为准逐条核对后，对计划中的模型做了以下修正（与上文修改一一对应）：

| 核对发现（代码证据） | 修正位置 |
|---|---|
| Op/Tensor 共享 ID 空间，不能按 ID 大小区分节点类型；禁止重复边与 Tensor→Tensor 边 | §2.1 新增输入硬约束 |
| 问题 1 跨核 1000 周期按前驱 Task 对去重；每核首 Task 从 0 起算 | §7.1 激活公式精确口径 |
| 问题 1 边界 COPY 按 `(Task, tensor)` 去重，切图搬运应按 \(c_t+o_t\) 计价而非按消费边数 | §7.1 新增计价公式 |
| 问题 2/3 跨核 COPY 按 `(tensor, 源核, 目标核)` 成对，聚合收益对 \(m_t\) “全有或全无”；合并 Task 无 100/1000 等待 | §8.1 重写 `cost_2` |
| DDR 与 CACHE_READ 均为全局共享池，命中收益是“换池”而非单纯加速 | §9.1 共享池注记 |
| FIFO 按 COPY_IN **完成时刻**插入；`COPY_OUT` 不触 Cache；spill 换入与切图读取同键空间；`hit_rate` 按字节 | §9.3 补充口径 |
| 问题 2/3 合法性是三道检查，Pipe FIFO 与 WAR/WAW 补边也参与全局判环 | §11 检查清单第 7 条 |
| 官方结果含 `cache_events`/`cache_final_entries`/`memory_path` 等核对字段 | §12 字段表 |
| 关键路径两个递推方向相反，原文未说明 | §5.4 方向注记 |

本次重点核对的参考材料：`C:\Users\xjw\Desktop\优秀论文\2025\A\优秀论文_2025_A_01.pdf`、`C:\Users\xjw\Desktop\优秀论文\2023\A\优秀论文_2023_A_01.pdf`、`C:\Users\xjw\Desktop\优秀论文\2024\A\优秀论文_2024_A_01.pdf`、`C:\Users\xjw\Desktop\优秀论文\2024\A\优秀论文_2024_A_02.pdf`，以及 `C:\Users\xjw\Desktop\2026 华为杯论文模板\第23届华为杯word模板.docx`。
