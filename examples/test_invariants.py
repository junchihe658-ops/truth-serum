"""闸门之间的算术不变量 —— 防止「数量级还在、方向也对，只是大了几倍」的错

## 为什么单独一个文件

2026-09-05 抓到一个 bug：② 号闸门跨标的聚合时，判据写成 `k.startswith("n")`，
本意是匹配笔数 n_all / n_uni，但 net_all / net_uni 也是 n 开头 ——
于是净期望被【求和】而不是取均值，4 个标的就报成 4 倍。

这种错最难自己发现：数量级还在、正负号也对，只是大了几倍。跑一次没人看得出来，
要拿另一道闸门的数字去对，才会发现「差 4 倍」。

所以这里的测试不看单个闸门算得对不对，只看**闸门之间对不对得上**、以及
**加标的时数字该不该变**。全部合成数据，秒级，不联网。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from truthserum.audits.overlap import OverlapAudit
from truthserum.core import Context, Costs, FuncStrategy
from truthserum.indicators import rsi

FAIL = []


def ck(name, cond, detail=""):
    if not cond:
        FAIL.append(name)
    print(f"  {'✅' if cond else '❌'} {name}" + (f"   {detail}" if detail else ""))


def bars_synth(n=1800, seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    px = 100 * (1 + 0.05 * np.sin(t / 31) + np.cumsum(rng.normal(0, 0.0015, n)))
    return pd.DataFrame(
        {"open": px, "high": px * (1 + rng.uniform(0, .004, n)),
         "low": px * (1 - rng.uniform(0, .004, n)), "close": px,
         "volume": rng.uniform(50, 150, n)},
        index=pd.date_range("2026-01-01", periods=n, freq="h"))


def strat(bars):
    r = rsi(bars["close"])
    return pd.Series(np.where(r > 60, 1.0, np.where(r < 40, -1.0, 0.0)),
                     index=bars.index)


B = bars_synth()
COSTS = Costs()
CPCT = COSTS.round_trip * 100


def agg_for(symbols):
    ctx = Context(bars={s: B for s in symbols},
                  strategy=FuncStrategy("t", strat), costs=COSTS,
                  barrier_mult=1.5, horizon=12)
    return OverlapAudit()._agg(ctx.strategy, ctx)


print("=" * 74)
print("【1】② 号闸门：净 必须等于 毛 − 成本")
print("=" * 74)

a1 = agg_for(["A"])
for tag in ("all", "uni"):
    g, n = a1[f"gross_{tag}"], a1[f"net_{tag}"]
    ck(f"{tag}: 净 == 毛 − 成本", abs((g - CPCT) - n) < 1e-9,
       f"毛 {g:+.4f}% − {CPCT:.4f}% = {g-CPCT:+.4f}%，报 {n:+.4f}%")

ck("去重叠后笔数不多于全部", a1["n_uni"] <= a1["n_all"],
   f"{a1['n_uni']} ≤ {a1['n_all']}")

print()
print("=" * 74)
print("【2】加标的时：笔数该变，期望不该变")
print("=" * 74)
print("   （这一条就是 2026-09-05 那个 bug 的正面靶子：")
print("    净期望曾被跨标的求和，3 个标的报成 3 倍）")
print()

a3 = agg_for(["A", "B", "C"])
ck("笔数按标的数放大 3 倍", a3["n_all"] == a1["n_all"] * 3,
   f"{a1['n_all']} → {a3['n_all']}")
for key in ("gross_all", "gross_uni", "net_all", "net_uni"):
    ck(f"{key} 与单标的相同", abs(a3[key] - a1[key]) < 1e-9,
       f"1 个标的 {a1[key]:+.4f}%  vs  3 个标的 {a3[key]:+.4f}%")

print()
print("=" * 74)
print(f"{'✅ 全部通过' if not FAIL else f'❌ {len(FAIL)} 项失败'}")
for f in FAIL:
    print("   ", f)
print("=" * 74)
sys.exit(1 if FAIL else 0)
