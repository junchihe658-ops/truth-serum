"""生成 HTML 体检报告 —— 这是录视频/给评审看的那个界面

跑完会在 reports/ 下生成三份，对应三种自欺：
  overfit.html   全样本调参（①② 全过，③ 拆穿）—— 最常见、最难自察
  peeking.html   偷看未来（只有 ① 抓到）
  honest.html    诚实的教科书策略（① 干净，②③④ 依然拦下）
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
from truthserum.report_html import save_html

SYMS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def honest(bars):
    r = rsi(bars["close"])
    return pd.Series(np.where(r > 60, 1.0, np.where(r < 40, -1.0, 0.0)),
                     index=bars.index)


def peeking(bars):
    nxt = bars["close"].shift(-1)
    return pd.Series(np.sign(nxt - bars["close"]).fillna(0).to_numpy(),
                     index=bars.index)


def overfit(bars):
    c = bars["close"]
    r, ef, es = rsi(c), c.ewm(span=12).mean(), c.ewm(span=48).mean()
    mom = c.pct_change(24)
    lo = (r > 58) & (ef > es) & (mom > 0.004)
    sh = (r < 42) & (ef < es) & (mom < -0.004)
    return pd.Series(np.where(lo, 1.0, np.where(sh, -1.0, 0.0)), index=bars.index)


bars, prov = load(SYMS, "1h")
note = describe(prov)          # 多行保留 —— 排版交给渲染层

CASES = [
    ("overfit", "全样本调参的三因子共振", overfit, "精选参数，回测年化 +112%"),
    ("peeking", "用了明天的收盘价", peeking, "回测年化 +9400%"),
    ("honest", "诚实的 RSI 动量", honest, "教科书策略，回测年化 +38%"),
]

print("生成报告…")
for slug, name, fn, claim in CASES:
    rep = check(bars, fn, name=name, claimed=claim)
    p = save_html(rep, f"reports/{slug}.html", data_note=note)
    marks = "".join({"clean": "✅", "failed": "❌", "unusable": "⛔",
                     "skipped": "⏭"}[r.verdict.value] for r in rep.results)
    print(f"  {marks}  {name:<24} → {p}")
print("\n闸门顺序：⓪ 数据出处  ① 前瞻  ② 重叠  ③ 本底  ④ 组合层")
