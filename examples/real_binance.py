"""在真实币安行情上跑四道闸门

合成数据只能证明机器能转。真正有说服力的是这个问题：

    一个【看起来完全合理、没有任何作弊】的策略，
    在真实数据上能不能过关？

三个受检对象：
  A. 诚实的 RSI 动量        —— 教科书策略，作者本人的生产系统就用它
  B. 偷看下一根收盘价       —— 明显作弊，前瞻检测必须抓到
  C. 「优化过」的多因子     —— 在全样本上挑出最好的参数组合再回测（最常见的自欺）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from truthserum import check
from truthserum.data import describe, load

SYMS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def honest_rsi(bars):
    """诚实的 RSI 顺势：RSI>60 做多、<40 做空。只用当前及之前的数据。"""
    r = rsi(bars["close"])
    return pd.Series(np.where(r > 60, 1.0, np.where(r < 40, -1.0, 0.0)),
                     index=bars.index)


def peeking(bars):
    """作弊：直接看下一根收盘价"""
    nxt = bars["close"].shift(-1)
    return pd.Series(np.sign(nxt - bars["close"]).fillna(0).to_numpy(),
                     index=bars.index)


def overfit_multi(bars):
    """「优化过」的多因子 —— 最常见、也最难自察的一种自欺

    在【全样本】上试了一堆阈值组合，挑出表现最好的那一套，然后回测。
    没有偷看未来（每根的计算都只用过去），但**参数是用全部数据挑的**。
    这类策略前瞻检测查不出来，只有零假设本底能拆穿。
    """
    c = bars["close"]
    r = rsi(c)
    ema_f, ema_s = c.ewm(span=12).mean(), c.ewm(span=48).mean()
    mom = c.pct_change(24)
    # 这些阈值是在全样本上挑出来的 —— 这就是问题所在
    long_ = (r > 58) & (ema_f > ema_s) & (mom > 0.004)
    short = (r < 42) & (ema_f < ema_s) & (mom < -0.004)
    return pd.Series(np.where(long_, 1.0, np.where(short, -1.0, 0.0)),
                     index=bars.index)


print("拉取真实币安行情（优先读本地缓存）…")
bars, prov = load(SYMS, "1h")
print(describe(prov))
print()

CASES = [
    ("A 诚实的 RSI 动量（教科书策略）", honest_rsi, "1h 周期 4 个币，回测年化 +38%"),
    ("B 偷看下一根收盘价", peeking, "回测年化 +9400%"),
    ("C 全样本调参的三因子共振", overfit_multi, "精选参数，回测年化 +112%"),
]

summary = []
for label, fn, claim in CASES:
    rep = check(bars, fn, name=label, claimed=claim)
    print(rep.render())
    print("\n" + "─" * 78 + "\n")
    summary.append((label, rep))

print("=" * 78)
print("  汇总")
print("=" * 78)
for label, rep in summary:
    marks = "".join({"clean": "✅", "failed": "❌", "unusable": "⛔",
                     "skipped": "⏭"}[r.verdict.value] for r in rep.results)
    print(f"  {marks}  {label}")
print()
print("  闸门顺序：① 前瞻  ② 重叠  ③ 本底  ④ 组合层")
