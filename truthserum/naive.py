"""天真回测器 —— 这【不是】我们的算法，这是我们要拆穿的那个算法

## 为什么要专门写一个错的

演示需要一个漂亮的数字作对照。编一个是最省事的 —— 但一个讲「不要自欺」
的工具，在自己的演示里摆一个编造的数字，评审一句「这 112% 哪来的」就塌了。

所以这里老老实实实现一遍**大多数人自己写的那个回测器**，在同一份真实
行情上算。出来的数字是真的，只是算法是错的。错在哪，下面逐条写明。

## 它犯的罪，一条条列清楚

1. **不扣手续费和滑点。**（`cost=0` 时）
   最常见的一条。币安永续 taker 0.05%/边，来回加滑点约 0.19%。
   一年几百笔，这一项就能把 +62% 变成 −41%。

2. **只报表现最好的那个标的。**（`best_of()` 干的事）
   同一个策略在 4 个币上跑，挑最好的那个写进宣传材料 ——
   这就是选择偏差，③ 号闸门专门抓它。

3. **单账户、单仓位、不考虑资金池。**
   实盘同时持有多个币时，仓位要分、保证金要占、回撤会叠加。
   ④ 号闸门做的才是账户层面的模拟。

4. **持满 horizon 才找下一笔。**
   这一条反而让它躲开了「重叠灌水」（② 号闸门抓的那个）——
   天真不等于每条都错，所以拆穿它必须逐条来，不能笼统说「你算错了」。

## 用法

    from truthserum.naive import naive_backtest, best_of
    rows = naive_backtest(bars, fn)          # 每个标的一行
    claim = best_of(rows, with_costs=False)  # 「他会拿去宣传的那个数字」
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import Costs, barrier_outcomes


@dataclass
class NaiveRow:
    symbol: str
    trades: int
    win_rate: float          # 小数
    total_return: float      # 小数，整段区间
    annualized: float        # 小数
    years: float
    cost_per_trade: float    # 小数，0 表示这一行没扣成本

    @property
    def label(self) -> str:
        tag = "未扣成本" if self.cost_per_trade == 0 else "已扣成本"
        return (f"{self.symbol} {tag} 年化 {self.annualized*100:+.1f}%"
                f"（{self.trades} 笔，胜率 {self.win_rate*100:.1f}%）")


def _one(df: pd.DataFrame, fn, mult: float, horizon: int,
         cost: float, symbol: str) -> NaiveRow | None:
    """单仓顺序持有：有信号就入场，持满 horizon 再看下一笔。

    ⚠ 这就是大多数人写的那个循环。它没有资金池、没有并发仓位、
      没有按权益定仓 —— 每一笔都当成「全部本金押上去」在复利。
    """
    sig = np.asarray(fn(df), dtype=float).reshape(-1)
    if len(sig) != len(df):
        raise ValueError(f"{symbol}: 策略返回 {len(sig)} 个值，"
                         f"与 {len(df)} 根 K 线不等长")
    rl, rs, _ = barrier_outcomes(df, mult, horizon)
    n, i, rets = len(sig), 0, []
    while i < n - horizon:
        s = sig[i]
        r = rl[i] if s > 0 else (rs[i] if s < 0 else np.nan)
        if s == 0 or not np.isfinite(r):
            i += 1
            continue
        rets.append(r - cost)
        i += horizon
    if not rets:
        return None
    rets = np.asarray(rets)
    eq = float(np.prod(1 + rets))
    years = (df.index[-1] - df.index[0]).total_seconds() / 31_536_000
    # 本金亏光之后没有「年化」可言，直接记 −100%，不许开根号开出复数
    ann = eq ** (1 / years) - 1 if eq > 0 and years > 0 else -1.0
    return NaiveRow(symbol, len(rets), float((rets > 0).mean()),
                    eq - 1, ann, years, cost)


def naive_backtest(bars: dict[str, pd.DataFrame], fn, *,
                   mult: float = 1.5, horizon: int = 12,
                   costs: Costs | None = None) -> list[NaiveRow]:
    """每个标的跑两遍：不扣成本、扣成本。返回全部行。

    两遍都跑是有意的 —— 差额就是「漏掉手续费」这一条罪的价码，
    演示时要能一眼看见它值多少钱。
    """
    rt = (costs or Costs()).round_trip
    rows: list[NaiveRow] = []
    for sym, df in bars.items():
        for c in (0.0, rt):
            row = _one(df, fn, mult, horizon, c, sym)
            if row is not None:
                rows.append(row)
    return rows


def best_of(rows: list[NaiveRow], *, with_costs: bool) -> NaiveRow | None:
    """挑年化最高的那一行 —— 也就是「他会拿去宣传的那个数字」。

    这个函数本身就是罪证：真要评估一个策略，看的是全部标的的分布，
    不是最好的那一个。③ 号闸门抓的正是这种「筛了很多个，最好的那个
    看起来不错」。
    """
    pool = [r for r in rows if (r.cost_per_trade == 0) == (not with_costs)]
    return max(pool, key=lambda r: r.annualized) if pool else None


def claim_line(rows: list[NaiveRow]) -> str:
    """生成「策略自称」那一行 —— 真算出来的，不是编的。"""
    b = best_of(rows, with_costs=False)
    if b is None:
        return "（天真回测没有产生任何交易）"
    return (f"{b.symbol} 单币回测，年化 {b.annualized*100:+.1f}%"
            f"（{b.trades} 笔，胜率 {b.win_rate*100:.1f}%）")
