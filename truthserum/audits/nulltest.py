"""零假设本底 —— 这个成绩，随机信号能不能也做出来

## 问题

「我筛了 20 个想法，最好的那个年化 +9%」——
但**随机信号里最好的那个，也能做出 +14%**。

不跟本底比，「最好的那个」永远看起来不错，因为你就是按「最好」挑的。

## 真实案例（2026-09-02）

日线组合模拟，rsi 顺势 4×ATR/10天：平均折收益 **+9.47%**，看起来很好。
跑 20 次打乱：

    全打乱   均值 −5.09%   范围 [−18.90%, +28.06%]   4/20 次超过真实值
    块打乱   均值 +0.90%   范围 [ −7.62%, +15.02%]   2/20 次超过真实值

经验 p ≈ 0.14~0.24 —— **和本底完全区分不开。**
原因是 8 折总共只有约 60 笔交易，结果被少数几笔主导。

## 两种打乱，破坏的东西不同

  · 全打乱：连信号自身的持续性一起破坏
  · 块打乱：保留局部持续性，只破坏「信号 ↔ 未来」的对应关系  ← 更严格

两个都跑，取更严格的那个作判据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..audit import Audit, AuditResult, SelfCheckFailed, Verdict
from ..core import barrier_outcomes

N_DRAWS = 20


def _score(bars, z, mult, horizon, cost):
    rl, rs, _ = barrier_outcomes(bars, mult, horizon)
    ok = np.isfinite(rl) & np.isfinite(rs) & (z != 0)
    idx = np.where(ok)[0]
    if len(idx) < 30:
        return np.nan
    keep, last = [], -1
    for i in idx:
        if i > last:
            keep.append(i); last = i + horizon
    k = np.array(keep)
    r = np.where(z[k] > 0, rl[k], rs[k])
    return float(np.mean(r) * 100) - cost * 100


def _shuffle(z, rng, block=None):
    n = len(z)
    if block is None:
        return z[rng.permutation(n)]
    starts = np.arange(0, n, block)
    blocks = [np.arange(x, min(x + block, n)) for x in starts]
    order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
    return z[order[:n]]


class NullTestAudit(Audit):
    name = "③ 零假设本底（随机信号能不能也做出来）"
    catches = "「筛了很多个，最好的那个看起来不错」——把选择偏差当成发现"

    def _real_and_null(self, strategy, ctx, draws=N_DRAWS):
        rng = np.random.default_rng(ctx.seed)
        reals, nulls_full, nulls_block = [], [], []
        for sym, bars in ctx.bars.items():
            z = strategy.signal(bars).to_numpy(float)
            v = _score(bars, z, ctx.barrier_mult, ctx.horizon, ctx.costs.round_trip)
            if not np.isfinite(v):
                continue
            reals.append(v)
            for _ in range(draws):
                nulls_full.append(_score(bars, _shuffle(z, rng), ctx.barrier_mult,
                                         ctx.horizon, ctx.costs.round_trip))
                nulls_block.append(_score(bars, _shuffle(z, rng, ctx.horizon),
                                          ctx.barrier_mult, ctx.horizon,
                                          ctx.costs.round_trip))
        return (np.array(reals), np.array(nulls_full, float),
                np.array(nulls_block, float))

    def _self_check(self, ctx) -> str:
        """植入一个【含答案】的策略：它必须显著跑赢自己的本底。

        如果连「明知有 edge」的策略都跑不赢本底，说明本底构造有问题
        （比如打乱没真的破坏对应关系），那么它对真实策略的判定也不可信。
        """
        sym = ctx.symbols[0]
        bars = ctx.bars[sym]

        class _Oracle:
            name = "oracle"
            def signal(self, b):
                fut = b["close"].shift(-ctx.horizon) - b["close"]
                return pd.Series(np.sign(fut).fillna(0).to_numpy(), index=b.index)

        sub = type(ctx)(bars={sym: bars}, strategy=_Oracle(), costs=ctx.costs,
                        barrier_mult=ctx.barrier_mult, horizon=ctx.horizon,
                        top_quantile=ctx.top_quantile, seed=ctx.seed)
        real, nf, nb = self._real_and_null(_Oracle(), sub, draws=5)
        if real.size == 0 or not np.isfinite(np.nanmean(nb)):
            raise SelfCheckFailed("样本不足")
        beat = float(np.nanmean(nb >= real.mean()))
        if beat > 0.05:
            raise SelfCheckFailed(
                f"植入一个【直接看未来】的策略，本底仍有 {beat*100:.0f}% 的抽样超过它 —— "
                f"说明打乱没有真正破坏『信号↔未来』的对应关系")
        return (f"在人为植入的『直接看未来』策略上验证：真实 {real.mean():+.3f}% "
                f"vs 本底中位 {np.nanmedian(nb):+.3f}%，本底 0% 抽样超过它 —— 判别力有效")

    def _run(self, ctx) -> AuditResult:
        real, nf, nb = self._real_and_null(ctx.strategy, ctx)
        if real.size == 0:
            return AuditResult(name=self.name, verdict=Verdict.SKIPPED,
                               headline="信号太少，无法做本底检验")
        r = float(real.mean())
        det = []
        worst_p = 0.0
        for lbl, arr in (("全打乱", nf), ("块打乱", nb)):
            a = arr[np.isfinite(arr)]
            if a.size == 0:
                continue
            beat = int((a >= r).sum())
            p = (beat + 1) / (a.size + 1)
            worst_p = max(worst_p, p)
            det.append(f"{lbl}：{a.size} 次抽样，均值 {a.mean():+.4f}%，"
                       f"范围 [{a.min():+.4f}%, {a.max():+.4f}%]，"
                       f"超过真实值的 {beat} 次 → p ≈ {p:.3f}")
        det.append(f"真实策略净期望 {r:+.4f}%/笔")
        if worst_p >= 0.05:
            return AuditResult(
                name=self.name, verdict=Verdict.FAILED,
                headline=f"与随机信号区分不开（最差 p ≈ {worst_p:.3f}，需 < 0.05）",
                detail=det + ["随机打乱后的信号也能做出同等成绩，"
                              "说明这个结果可以用运气解释。"],
                numbers={"real": r, "p": worst_p})
        return AuditResult(
            name=self.name, verdict=Verdict.CLEAN,
            headline=f"显著优于随机本底（最差 p ≈ {worst_p:.3f}）",
            detail=det, numbers={"real": r, "p": worst_p})
