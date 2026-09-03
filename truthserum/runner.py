"""把四道闸门串起来"""
from __future__ import annotations

import pandas as pd

from .audit import TruthReport
from .audits.lookahead import LookaheadAudit
from .audits.nulltest import NullTestAudit
from .audits.overlap import OverlapAudit
from .audits.portfolio import PortfolioAudit
from .core import Context, Costs, FuncStrategy, Strategy

#: 顺序有意为之：前瞻是【最致命】的，先查。
#: 一个偷看未来的策略，后面三项跑出来的所有数字都没有意义。
AUDITS = [LookaheadAudit, OverlapAudit, NullTestAudit, PortfolioAudit]


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
