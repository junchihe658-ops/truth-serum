"""把四道闸门串起来"""
from __future__ import annotations

import pandas as pd

from .audit import TruthReport
from .audits.lookahead import LookaheadAudit
from .audits.nulltest import NullTestAudit
from .audits.overlap import OverlapAudit
from .audits.portfolio import PortfolioAudit
from .audits.provenance import ProvenanceAudit
from .core import Context, Costs, FuncStrategy, Strategy

#: 顺序有意为之，从「最底层的前提」往上查：
#:   ⓪ 数据本身对不对   —— 数据错了，后面全部没意义
#:   ① 有没有偷看未来   —— 偷看了，后面的数字也没意义
#:   ② 每笔期望有没有灌水
#:   ③ 随机信号能不能也做出来
#:   ④ 账户里最后剩多少钱  ← 唯一有资格下结论的那个
AUDITS = [ProvenanceAudit, LookaheadAudit, OverlapAudit,
          NullTestAudit, PortfolioAudit]


def check(bars: dict[str, pd.DataFrame] | pd.DataFrame,
          strategy: Strategy | callable,
          *, name: str | None = None, claimed: str = "",
          costs: Costs | None = None, **kw) -> TruthReport:
    """给一个策略做体检。

    bars     : {symbol: OHLCV DataFrame}，或单个 DataFrame
    strategy : 有 .signal(bars) 的对象，或一个 bars -> Series 的函数
    claimed  : 这个策略自称的成绩，会印在报告顶部作对照
    """
    if isinstance(bars, pd.DataFrame):
        bars = {"ASSET": bars}
    if not hasattr(strategy, "signal"):
        strategy = FuncStrategy(name or getattr(strategy, "__name__", "strategy"),
                                strategy)
    ctx = Context(bars=bars, strategy=strategy,
                  costs=costs or Costs(), **kw)
    results = [A().report(ctx) for A in AUDITS]
    return TruthReport(
        strategy_name=name or getattr(strategy, "name", "strategy"),
        results=results, claimed=claimed)
