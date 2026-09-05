"""调参 agent —— 目标单一，迭代搜索，全过程留痕

## 它做什么

给它一个策略族（RSI 阈值 + 周期 + 持仓时长）和一个目标（每笔期望最大化），
它自己搜完整个空间、挑出最好的一组，并把**每一次试验**记进 SearchLog。

## 为什么它是这个项目最好的靶子

这个 agent 完全"诚实"：不偷看未来、不改数据、每一步都可复现。
但它几乎必然把自己优化进**选择偏差** —— 搜两百组挑最好的那组，
那组的成绩里有多少是真本事、多少是运气，从它自己的视角完全看不出来。
它只知道"数字变好了"。

这正是人类调参时干的事，也是 ⑤ 号闸门存在的理由。

## 硬规矩：必须交出完整搜索日志

只交胜出者、不交试过多少组，⑤ 号闸门就无从判断，选择偏差直接消失在
视野之外。所以 `run()` 返回的永远是 `(策略, SearchLog)` 一对，
不提供"只要策略"的接口 —— 那个接口本身就是一条自欺的捷径。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import Costs, SearchLog, barrier_outcomes, net_expectancy
from ..indicators import rsi


@dataclass
class Candidate:
    period: int
    hi: float
    lo: float

    @property
    def label(self) -> str:
        return f"RSI({self.period}) >{self.hi:g} 做多 / <{self.lo:g} 做空"

    def make(self):
        p, hi, lo = self.period, self.hi, self.lo

        def signal(bars: pd.DataFrame) -> pd.Series:
            r = rsi(bars["close"], p)
            return pd.Series(np.where(r > hi, 1.0, np.where(r < lo, -1.0, 0.0)),
                             index=bars.index)
        return signal


class TunerAgent:
    """把一族 RSI 策略搜一遍，挑每笔期望最高的那组。

    periods / his / los 一改，搜索空间大小就变 —— 而搜索空间越大，
    选择偏差越强。这个 agent 不会替你隐瞒这一点：n_trials 原样进日志。
    """

    name = "调参 agent"

    def __init__(self, periods=(7, 14, 21), his=(55, 58, 60, 62, 65, 68, 70),
                 los=(30, 32, 35, 38, 40, 42, 45),
                 barrier_mult: float = 1.5, horizon: int = 12,
                 costs: Costs | None = None, verbose: bool = True):
        self.space = [Candidate(p, h, l)
                      for p in periods for h in his for l in los]
        self.barrier_mult, self.horizon = barrier_mult, horizon
        self.costs = costs or Costs()
        self.verbose = verbose

    def run(self, bars: dict[str, pd.DataFrame]):
        """搜完整个空间。返回 (胜出策略函数, SearchLog)。"""
        # 屏障结果只依赖 K 线，不依赖信号 —— 算一次，几百组参数复用。
        # 不做这一步，搜两百组要跑几十分钟。
        pre = {s: barrier_outcomes(df, self.barrier_mult, self.horizon)[:2]
               for s, df in bars.items()}

        if self.verbose:
            print(f"  [{self.name}] 搜索空间 {len(self.space)} 组，"
                  f"目标：去重叠后的净每笔期望最大化")

        scores, best, best_c = [], -np.inf, None
        for i, c in enumerate(self.space, 1):
            fn = c.make()
            vals = []
            for s, df in bars.items():
                z = np.asarray(fn(df), dtype=float).reshape(-1)
                v = net_expectancy(df, z, self.barrier_mult, self.horizon,
                                   self.costs.round_trip, pre=pre[s])
                if np.isfinite(v):
                    vals.append(v)
            score = float(np.mean(vals)) if vals else float("nan")
            scores.append(score)
            if np.isfinite(score) and score > best:
                best, best_c = score, c
                if self.verbose:
                    print(f"    第 {i:>3} 组刷新最好：{score:+.4f}%/笔  "
                          f"{c.label}")

        if best_c is None:
            raise RuntimeError("搜索没有产出任何有效结果")

        log = SearchLog(
            n_trials=len(self.space), scores=scores,
            best_score=best, best_label=best_c.label,
            space=f"RSI 周期 × 做多阈值 × 做空阈值 = {len(self.space)} 组")

        if self.verbose:
            print(f"  [{self.name}] 搜完 {len(self.space)} 组。"
                  f"最好 {best:+.4f}%/笔，中位 {log.median:+.4f}%")
            print(f"  [{self.name}] 胜出：{best_c.label}")
            print(f"  [{self.name}] ⚠ 我只知道「数字变好了」。"
                  f"这里面有多少是运气，我自己看不出来 —— 那是审计器的事。")

        return best_c.make(), log
