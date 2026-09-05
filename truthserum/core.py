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
class SearchLog:
    """一次参数搜索的全过程记录。

    ## 为什么必须记全过程，而不是只记胜出者

    「我试了 200 组，这组最好」和「我只试了这一组，它就是好」——
    成绩可能一模一样，可信度差着数量级。只交出胜出者，选择偏差就消失在
    视野之外了，而那正是最常见的自欺方式。

    所以搜索型 agent 必须把每一次试验都记下来，⑤ 号闸门才有东西可查。
    """
    n_trials: int                    # 一共试了多少组
    scores: list[float]              # 每组的成绩（去重叠净每笔期望 %）
    best_score: float
    best_label: str                  # 胜出参数的人话描述
    space: str = ""                  # 搜索空间的描述，写进报告
    #: 胜出者的信号（{标的: 数组}）。⑤ 号闸门要拿它构造本底 ——
    #: 用「当次提交的那个」构造本底是错的：不同策略的交易结构不同，
    #: 本底分布跟着变，p 值跨次不可比。实测会出现「试得越多反而越显著」。
    best_signal: dict | None = None

    @property
    def median(self) -> float:
        import numpy as _np
        f = [s for s in self.scores if _np.isfinite(s)]
        return float(_np.median(f)) if f else float("nan")


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
    # ⚠ 这里【故意】没有 top_quantile。
    #   曾经有一个 `top_quantile: float = 0.10  # 只做信号最强的前 q`，
    #   但全库搜下来它从没过滤过任何信号 —— 注释声称在干活，实际是死参数。
    #   任何人读 Context 都会以为审的是「最强的 10% 信号」，其实是全量。
    #   在一个讲「不要自欺」的项目里，代码里的一句假话比缺个功能严重得多。
    #   要么实现，要么删掉；留着最危险。删了。
    #   （真要按信号强度筛，得先让 Strategy 协议能表达强度 —— 它现在只允许
    #     -1/0/+1，强度根本无从表达。那是另一件事，不是加个参数能解决的。）
    max_positions: int = 4
    pos_pct: float = 0.15              # 单笔名义 = 权益 × pos_pct × leverage
    leverage: int = 3
    n_folds: int = 8
    seed: int = 20260902

    #: 搜索日志。策略是「搜出来的」时才有值 —— ⑤ 号闸门要靠它算选择偏差。
    #: 没有就是没有：⑤ 号会判「未检」，而不是假装查过了。
    search_log: "SearchLog | None" = None

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


def dedup_indices(sig: np.ndarray, valid: np.ndarray, horizon: int) -> np.ndarray:
    """去重叠：一笔持满 horizon 之前不再开新仓，返回保留下来的行号。

    ② 号和 ⑤ 号闸门都要用。**必须共用一份** —— 各写各的就是数字漂移的温床，
    2026-09-05 那个「净期望被跨标的求和」的 bug 就是这么来的。
    """
    idx = np.where(valid & (sig != 0))[0]
    keep, last_end = [], -1
    for i in idx:
        if i > last_end:
            keep.append(i)
            last_end = i + horizon
    return np.asarray(keep, dtype=int)


def net_expectancy(bars: pd.DataFrame, sig: np.ndarray, mult: float,
                   horizon: int, cost: float,
                   pre: tuple | None = None) -> float:
    """去重叠后的净每笔期望（百分数）。与 ②③ 口径一致。

    pre 传入预先算好的 (rl, rs) 可以省掉最贵的一步 —— 屏障结果只依赖 K 线，
    不依赖信号，所以搜索几百组参数时算一次就够。
    """
    rl, rs = pre if pre is not None else barrier_outcomes(bars, mult, horizon)[:2]
    valid = np.isfinite(rl) & np.isfinite(rs)
    k = dedup_indices(np.asarray(sig, dtype=float), valid, horizon)
    if len(k) < 30:
        return float("nan")
    r = np.where(sig[k] > 0, rl[k], rs[k])
    return float(r.mean() * 100 - cost * 100)


def fold_bounds(index: pd.DatetimeIndex, n_folds: int, warmup_frac: float = 0.25):
    """走向前折：前 warmup_frac 留作最早一折的训练/预热，其余等分成 n_folds 段"""
    n = len(index)
    start = int(n * warmup_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_folds)]
