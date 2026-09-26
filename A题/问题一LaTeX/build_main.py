"""Build the problem-one paper using the official 2026 template preamble."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
template = (HERE / "template2026" / "example.tex").read_text(encoding="utf-8")
preamble = template[: template.index(r"\begin{document}")]
replacements = {
    r"\title{": r"\title{通用神经网络处理器下的多核调度问题\\问题一：面向5核的切图与 Task 调度优化}",
    r"\baominghao{": r"\baominghao{}",
    r"\schoolname{": r"\schoolname{}",
    r"\membera{": r"\membera{}",
    r"\memberb{": r"\memberb{}",
    r"\memberc{": r"\memberc{}",
}
lines = []
for line in preamble.splitlines():
    stripped = line.lstrip()
    for prefix, replacement in replacements.items():
        if stripped.startswith(prefix):
            line = line[: len(line) - len(stripped)] + replacement
            break
    lines.append(line)
preamble = "\n".join(lines) + "\n"

body = r"""\begin{document}
\maketitle
\begin{abstract}
针对通用神经网络处理器上的多核切图与任务调度问题，本文研究问题一：每个子图作为独立 Task，跨子图数据通过 DDR 传输。我们将 Tensor--Op 二部图收缩为 Op-DAG，以子图归属、核心归属和核心内 Task 顺序作为决策变量，以官方离散事件评估器返回的 Makespan 为最终目标。算法采用连续链簇、负载加权、轮转和 wave 规则生成候选，再对最难的 20 个用例使用遗传搜索与模拟退火混合局部搜索。最终 5 核方案覆盖 100 个用例，结构校验和官方评估均无错误，算术平均加速比为 $3.351113$，几何平均为 $2.982523$，总周期比为 $3.633128$。
\keywords{多核调度\quad Tensor--Op 图\quad Task 切分\quad 离散事件仿真\quad 混合搜索}
\end{abstract}
\pagestyle{plain}
\maketoc
\clearpage
\input{problem1_section}
\end{document}
"""
(HERE / "main.tex").write_text(preamble + body, encoding="utf-8")
print("main.tex generated from the official 2026 template")
