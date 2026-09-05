"""⓪ 号闸门：数据出处核验 —— 单独演示

流程（这就是「agent 用 MCP 干活」的真实样子）：
  1. agent 通过 Binance MCP Server 拉一段 K 线，存成参照样本
  2. 本地缓存的行情逐根与参照比对
  3. 对不上就报警 —— 无论它来自哪个"看起来很官方"的接口

自检会先把参照人为改动 0.01%，确认检测器能抓到，再给结论。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from truthserum.audits.provenance import ProvenanceAudit, load_reference
from truthserum.core import Context, Costs
from truthserum.data import load

SYM = "ETHUSDT"

print("拉取本地缓存的行情…")
bars, prov = load([SYM], "1h", refresh=True)
df = bars[SYM]
print(f"  {SYM}: {len(df)} 根  {df.index[0]} ~ {df.index[-1]}")

ref = load_reference("./.cache", SYM, "1h")
if ref is None:
    print("❌ 没有 MCP 参照样本，先让 agent 通过 Binance MCP 抓一段")
    sys.exit(2)
print(f"  MCP 参照: {len(ref)} 根  {ref.index[0]} ~ {ref.index[-1]}")
print(f"  重叠区间: {len(df.index.intersection(ref.index))} 根")


class _Noop:
    name = "noop"
    def signal(self, b):
        return pd.Series(np.zeros(len(b)), index=b.index)


ctx = Context(bars={SYM: df}, strategy=_Noop(), costs=Costs())

print()
print("=" * 78)
print("场景 1：缓存与官方 MCP 数据一致时")
print("=" * 78)
print(ProvenanceAudit().report(ctx).render())

print()
print("=" * 78)
print("场景 2：缓存被污染时（模拟「模拟盘假影线」——最低价被压低 0.5%）")
print("=" * 78)
print("  2026-08 真实案例：模拟盘同一分钟的最低价比生产端低 0.52%，")
print("  多出一根真实市场不存在的下影线，止损因此被【不存在的价格】打掉。")
print()
dirty = df.copy()
hit = dirty.index.intersection(ref.index)
if len(hit):
    dirty.loc[hit[len(hit) // 2], "low"] *= 0.995
ctx2 = Context(bars={SYM: dirty}, strategy=_Noop(), costs=Costs())
print(ProvenanceAudit().report(ctx2).render())
