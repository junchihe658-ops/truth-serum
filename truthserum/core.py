"""核心数据模型：策略接口 + 体检上下文

## 策略接口故意做得很窄

    signal(bars) -> Series[-1/0/+1]

只有这一个方法，输入是一张 OHLCV 表，输出是等长的信号序列。
**窄接口是前瞻检测能通用的前提** —— 检测器只要把 bars 截短、重算、
比较截断前的那些行就行，完全不需要理解策略内部在算什么。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close", "volume")


class Strategy(Protocol):
    name: str

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """返回与 bars 等长的信号：+1 做多 / −1 做空 / 0 观望。

        ⚠ 合约：第 i 行的信号只能用第 i 行【及之前】的数据。
          违反了也没关系 —— LookaheadAudit 会抓到你。
        """
        ...


@dataclass
class FuncStrategy:
    """把一个普通函数包成策略，方便 demo 和 agent 生成的代码"""
    name: str
    fn: Callable[[pd.DataFrame], pd.Series]

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        s = self.fn(bars)
        if not isinstance(s, pd.Series):
            s = pd.Series(np.asarray(s), index=bars.index)
        return s.reindex(bars.index).fillna(0).clip(-1, 1)


@dataclass
class Costs:
    """成本模型。默认值来自 2026-09 在 OKX 永续上的实测。

    ⚠ 这些数字是【实测】不是拍脑袋：
      手续费 taker 0.05%/边（账本 190 笔中位 0.0998% 来回，与之吻合）
      滑点 入场 0.0464% / 止盈 0.0835% / 止损 0.0053%
    """
    fee_per_side: float = 0.0005
    slip_entry: float = 0.000464
    slip_tp: float = 0.000835
    slip_sl: float = 0.000053

    @property
    def round_trip(self) -> float:
        """一笔完整交易的来回成本（小数，非百分比）"""
        return (self.fee_per_side * 2 + self.slip_entry
                + (self.slip_tp + self.slip_sl) / 2)

    def describe(self) -> str:
        return (f"来回 {self.round_trip*100:.4f}%"
                f"（手续费 {self.fee_per_side*200:.3f}% + 滑点）")


@dataclass
class Context:
    """一次体检需要的全部东西"""
    bars: dict[str, pd.DataFrame]      # symbol -> OHLCV，index 是 DatetimeIndex
    strategy: Strategy
    costs: Costs = field(default_factory=Costs)
    barrier_mult: float = 1.5          # 止盈止损各 mult × ATR
    horizon: int = 12                  # 时间屏障：多少根 K 线
    top_quantile: float = 0.10         # 只做信号最强的前 q
    max_positions: int = 4
    pos_pct: float = 0.15              # 单笔名义 = 权益 × pos_pct × leverage
    leverage: int = 3
    n_folds: int = 8
    seed: int = 20260902

    def __post_init__(self):
        for s, df in self.bars.items():
            miss = [c for c in REQUIRED_COLS if c not in df.columns]
            if miss:
                raise ValueError(f"{s} 缺少列 {miss}")
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{s} 的 index 必须是 DatetimeIndex")
            if not df.index.is_monotonic_increasing:
                raise ValueError(f"{s} 的 index 必须按时间升序")

    @property
    def symbols(self) -> list[str]:
        return list(self.bars)


# ────────────────────────────────────────────────────────────
# 公用计算
# ────────────────────────────────────────────────────────────
def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def barrier_outcomes(df: pd.DataFrame, mult: float, horizon: int):
    """每根 K 线：做多 / 做空各自持到屏障或到点的收益率（不含成本）

    同一根内同时触及上下屏障时，保守地按【先触到不利的那边】算。
    """
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    a = atr(df)
    n = len(c)
    rl, rs = np.full(n, np.nan), np.full(n, np.nan)
    hit_time = np.full(n, np.nan)
    for i in range(n - horizon):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        b = mult * a[i]
        for d, arr in ((1, rl), (-1, rs)):
            tp, sl = c[i] + d * b, c[i] - d * b
            ex, j_hit = None, horizon
            for j in range(i + 1, i + horizon + 1):
                if d == 1:
                    if l[j] <= sl: ex, j_hit = sl, j - i; break
                    if h[j] >= tp: ex, j_hit = tp, j - i; break
                else:
                    if h[j] >= sl: ex, j_hit = sl, j - i; break
                    if l[j] <= tp: ex, j_hit = tp, j - i; break
            arr[i] = d * ((ex if ex is not None else c[i + horizon]) - c[i]) / c[i]
            if d == 1:
                hit_time[i] = j_hit
    return rl, rs, hit_time


def fold_bounds(index: pd.DatetimeIndex, n_folds: int, warmup_frac: float = 0.25):
    """走向前折：前 warmup_frac 留作最早一折的训练/预热，其余等分成 n_folds 段"""
    n = len(index)
    start = int(n * warmup_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_folds)]
