"""Regenerate the v4/v5 comparison figure from the archived official results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
RESULTS = next(WORKSPACE.rglob("chain_generator_5core_v4_summary.csv")).parent


def read_summary(name: str):
    with (RESULTS / name).open(newline="", encoding="utf-8-sig") as handle:
        return {row["case"]: row for row in csv.DictReader(handle)}


v4 = read_summary("chain_generator_5core_v4_summary.csv")
v5 = read_summary("chain_generator_5core_v5_summary.csv")
cases = sorted(v5)
x = [float(v4[case]["speedup"]) for case in cases]
y = [float(v5[case]["speedup"]) for case in cases]
changed = [case for case in cases if float(v5[case]["problem1_makespan"])
           < float(v4[case]["problem1_makespan"])]
indices = [cases.index(case) for case in changed]

plt.rcParams.update({
    "font.family": "SimSun",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=180)
ax.scatter(x, y, s=18, c="#4C78A8", alpha=0.65, label="其余用例")
ax.scatter([x[i] for i in indices], [y[i] for i in indices], s=38,
           c="#E45756", label="混合搜索接受改进")
lo, hi = min(x + y), max(x + y)
ax.plot([lo, hi], [lo, hi], ls="--", c="gray", lw=1, label="$y=x$")
ax.set_xlabel("v4 加速比")
ax.set_ylabel("v5 加速比")
ax.set_title("5核100个用例：v4与v5加速比对比")
ax.grid(alpha=0.2)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "figures" / "v4_v5_speedup.pdf", bbox_inches="tight")
fig.savefig(HERE / "figures" / "v4_v5_speedup.png", bbox_inches="tight")
plt.close(fig)
