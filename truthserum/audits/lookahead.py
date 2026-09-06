"""前瞻检测 —— 策略有没有偷看未来

## 判据

一个诚实的策略，第 i 根 K 线的信号只能由第 i 根及之前的数据决定。
所以：**把切点之后的 K 线全删掉、重算信号，切点之前每一根的信号必须一位不变。**

变了，就是它偷看了未来。这个判据不需要理解策略内部在算什么 ——
只要策略的接口是 `signal(bars) -> Series`，它就通用。

## 自检：给策略套一层「偷看一根」的外壳

检测器必须抓到这个人为植入的 bug。抓不到，就说明它自己瞎了，
它说的「干净」没有任何意义。

作者写这个检测器时连错三版，每一版都会在真实数据上报「未发现前瞻」——
全靠这一层自检拦下来。见 audit.py 的模块注释。

## 这道闸门专门防的那一类 bug

最难自察的一种：两份数据用了不同的时间基准（比如 K 线按收盘对齐、
衍生品按整点对齐，差几个小时），代码里那条防前瞻的缓冲就被悄悄抵消了，
回测于是能读到未来几个小时的数据。

它不会报错。它只会让走向前验证的成绩整体虚高一截，看着像策略变好了。
手工排查要几天，而这道闸门几秒钟就能定位到是哪个标的、哪一根 K 线。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..audit import Audit, AuditResult, SelfCheckFailed, Verdict


def _diff_before_cut(strategy, bars: pd.DataFrame, cut: int,
                     tail_pad: int) -> tuple[int, int, float]:
    """比较【两个不同长度的截断】在 cut 之前产生的信号

        短版 = signal(bars[:cut])          → cut 附近的行【拿不到】未来
        长版 = signal(bars[:cut+tail_pad]) → 同样的行【拿得到】未来
        比较 [0, cut) 的信号

    返回 (不同的根数, 比较的总根数, 最早差异距切点多少根)

    ⚠ 为什么不是「截断版 vs 全量版」：
      那样在两边都留了未来数据，短周期的偷看会被 padding 抹平 ——
      作者第一版就是这么写的，自检当场判定检测器失效。
      必须让短版真的缺少未来，差异才会显形。

    ⚠ tail_pad 的作用：长版要多留一段，这样它自己的尾部丢弃
      （指标预热、NaN）不会波及到 [0, cut) 这段比较区间。
    """
    short = strategy.signal(bars.iloc[:cut]).to_numpy(float)
    long_ = strategy.signal(bars.iloc[:cut + tail_pad]).to_numpy(float)
    m = min(cut, len(short), len(long_))
    if m <= 0:
        return 0, 0, np.nan
    a = np.nan_to_num(short[:m], nan=0.0)
    b = np.nan_to_num(long_[:m], nan=0.0)
    d = ~np.isclose(a, b, rtol=1e-9, atol=1e-12)
    if not d.any():
        return 0, m, np.nan
    first = int(np.argmax(d))
    return int(d.sum()), m, float(cut - first)


class _PeekWrapper:
    """把任意策略包成「偷看未来 k 根」的版本 —— 只用于自检"""

    def __init__(self, inner, k: int = 1):
        self.inner = inner
        self.k = k
        self.name = f"{getattr(inner, 'name', 'strategy')}+peek{k}"

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        base = self.inner.signal(bars)
        # 用未来 k 根的收益方向【覆盖】信号 —— 一个再明显不过的前瞻
        fut = bars["close"].shift(-self.k) - bars["close"]
        peek = np.sign(fut).fillna(0)
        return pd.Series(np.where(peek != 0, peek, base), index=bars.index)


class LookaheadAudit(Audit):
    name = "① 前瞻检测（策略有没有偷看未来）"
    catches = "任何让『未来数据改变过去信号』的实现错误：时区错位、居中窗口、bfill、跨周期错位对齐"

    def __init__(self, n_cuts: int = 6, tail_pad: int = 48):
        self.n_cuts = n_cuts
        self.tail_pad = tail_pad

    def _cuts(self, n: int) -> list[int]:
        lo, hi = int(n * 0.35), int(n * 0.92)
        return [int(x) for x in np.linspace(lo, hi, self.n_cuts)]

    def _scan(self, strategy, ctx) -> dict[str, tuple[int, int, float]]:
        worst: dict[str, tuple[int, int, float]] = {}
        for sym, bars in ctx.bars.items():
            if len(bars) < 500:
                continue
            for cut in self._cuts(len(bars)):
                nd, tot, dist = _diff_before_cut(strategy, bars, cut, self.tail_pad)
                if nd == 0:
                    continue
                prev = worst.get(sym, (0, 0, 0.0))
                if nd > prev[0]:
                    worst[sym] = (nd, tot, dist)
        return worst

    def _self_check(self, ctx) -> str:
        planted = _PeekWrapper(ctx.strategy, k=1)
        found = self._scan(planted, ctx)
        if not found:
            raise SelfCheckFailed(
                "给策略套上『偷看未来 1 根』的外壳后，检测器仍报告干净")
        sym, (nd, tot, dist) = max(found.items(), key=lambda kv: kv[1][0])
        return (f"在人为植入『偷看未来 1 根』的版本上抓到了 —— "
                f"{sym} 有 {nd}/{tot} 根信号被未来数据改变")

    def _run(self, ctx) -> AuditResult:
        found = self._scan(ctx.strategy, ctx)
        if not found:
            return AuditResult(
                name=self.name, verdict=Verdict.CLEAN,
                headline="未发现前瞻：删掉切点之后的全部 K 线后，之前每一根信号都一位不变",
                detail=[f"受检 {len(ctx.bars)} 个标的 × {self.n_cuts} 个切点，"
                        f"窗口越过切点再留 {self.tail_pad} 根（防止差异落进被丢弃的尾部）"],
                numbers={"symbols": len(ctx.bars), "cuts": self.n_cuts})
        worst_sym, (nd, tot, dist) = max(found.items(), key=lambda kv: kv[1][0])
        return AuditResult(
            name=self.name, verdict=Verdict.FAILED,
            headline=f"发现前瞻：{len(found)} 个标的的历史信号会被未来数据改变",
            detail=[f"最严重 {worst_sym}：{nd}/{tot} 根不一致，"
                    f"最早的差异出现在切点前 {dist:.0f} 根",
                    "这意味着回测里的每一个数字都包含了当时拿不到的信息。",
                    "常见成因：两套时间基准错位、居中滚动窗口、bfill/nearest 填充、"
                    "跨周期按索引而非收盘时间对齐。"],
            numbers={"symbols_affected": len(found), "worst_bars": nd,
                     "horizon_bars": dist})
