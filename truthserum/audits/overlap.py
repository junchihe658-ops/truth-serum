"""重叠计数检测 —— 「平均每笔期望」是不是被信号持续期灌水了

## 问题

一段趋势持续 20 根 K 线，这 20 根可能全都满足入场条件。
如果你把它们当成 20 笔独立观测去平均，就等于**按信号持续多久加权**，
而不是按你真正做了几笔加权。

而信号持续得久，恰恰是趋势最强、最赚钱的时候 —— 所以这个偏差是【系统性向上】的。

## 真实案例（2026-09-02）

同一个配置（4×ATR / 24h 屏障）：

    含重叠   4179 笔   毛期望 +0.2853%   净 **+0.0945%**   ← 看起来能赚
    去重叠    727 笔   毛期望 +0.1784%   净 **−0.0124%**   ← 其实亏
    组合模拟  978 笔                     净 **−0.1418%**

**足以把负的变成正的。** 当时作者据此差点得出「宽屏障能盈利」的结论。

## 判据

同一份数据、同一个策略，分别按「全部信号」和「一笔没结束不再开」统计。
两者差得越多，说明原始数字被灌水越严重。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..audit import Audit, AuditResult, SelfCheckFailed, Verdict
from ..core import barrier_outcomes

#: 净期望因去重叠而【变号】= 致命；仅缩水 = 警告
_INFLATION_WARN = 0.20      # 毛期望缩水超过 20% 就值得说


def _both_ways(bars, sig, mult, horizon, cost):
    rl, rs, _ = barrier_outcomes(bars, mult, horizon)
    z = sig.to_numpy(float)
    ok = np.isfinite(rl) & np.isfinite(rs) & (z != 0)
    idx = np.where(ok)[0]
    if len(idx) < 30:
        return None
    r_all = np.where(z[idx] > 0, rl[idx], rs[idx])
    keep, last_end = [], -1
    for i in idx:
        if i > last_end:
            keep.append(i)
            last_end = i + horizon
    k = np.array(keep)
    r_uni = np.where(z[k] > 0, rl[k], rs[k])
    return {
        "n_all": len(idx), "n_uni": len(k),
        "gross_all": float(np.mean(r_all) * 100),
        "gross_uni": float(np.mean(r_uni) * 100),
        "net_all": float(np.mean(r_all) * 100) - cost * 100,
        "net_uni": float(np.mean(r_uni) * 100) - cost * 100,
    }


class OverlapAudit(Audit):
    name = "② 重叠计数检测（每笔期望是不是被灌水）"
    catches = "把高度重叠的同一段行情当成多笔独立观测，从而系统性高估平均每笔收益"

    def _agg(self, strategy, ctx):
        rows = []
        for sym, bars in ctx.bars.items():
            r = _both_ways(bars, strategy.signal(bars), ctx.barrier_mult,
                           ctx.horizon, ctx.costs.round_trip)
            if r:
                rows.append(r)
        if not rows:
            return None
        # ⚠ 笔数跨标的【求和】，期望跨标的【取均值】。
        #   原先判据写的是 k.startswith("n") —— 本意匹配 n_all / n_uni，
        #   但 net_all / net_uni 也是 n 开头，于是净期望被求和了：
        #   4 个标的就报成 4 倍。毛期望不受影响（不以 n 开头），所以
        #   「缩水 %」一直是对的，只有「净」那两个数字是错的。
        #   这种错法最阴险 —— 数量级还在，方向也对，只是大了几倍。
        counts = ("n_all", "n_uni")
        return {k: (sum(r[k] for r in rows) if k in counts
                    else float(np.mean([r[k] for r in rows])))
                for k in rows[0]}

    def _self_check(self, ctx) -> str:
        """植入一个【极度持续】的信号：它必然产生大量重叠。

        检测器必须报告「去重叠后笔数大幅下降」。报不出来说明它没在算重叠。
        """
        sym = ctx.symbols[0]
        bars = ctx.bars[sym]

        class _Sticky:
            name = "sticky"
            def signal(self, b):    # noqa: D401 - 恒定做多，重叠必然最大化
                return pd.Series(np.ones(len(b)), index=b.index)

        r = _both_ways(bars, _Sticky().signal(bars), ctx.barrier_mult,
                       ctx.horizon, ctx.costs.round_trip)
        if r is None:
            raise SelfCheckFailed("样本不足，无法验证")
        ratio = r["n_uni"] / max(r["n_all"], 1)
        expect = 1.0 / ctx.horizon
        if not (ratio < expect * 2.5):
            raise SelfCheckFailed(
                f"恒定信号下去重叠笔数应约为总数的 1/{ctx.horizon}，"
                f"实测比例 {ratio:.3f}，说明没有真正在去重叠")
        return (f"在恒定信号（重叠最大化）上验证：{r['n_all']} 笔 → "
                f"去重叠后 {r['n_uni']} 笔（≈1/{ctx.horizon}），去重逻辑有效")

    def _run(self, ctx) -> AuditResult:
        a = self._agg(ctx.strategy, ctx)
        if a is None:
            return AuditResult(name=self.name, verdict=Verdict.SKIPPED,
                               headline="信号太少，无法评估重叠")
        det = [
            f"全部信号：{a['n_all']:.0f} 笔，毛 {a['gross_all']:+.4f}%，"
            f"净 {a['net_all']:+.4f}%",
            f"去重叠后：{a['n_uni']:.0f} 笔，毛 {a['gross_uni']:+.4f}%，"
            f"净 {a['net_uni']:+.4f}%",
        ]
        flipped = a["net_all"] > 0 >= a["net_uni"]
        # ⚠ 缩水率只在【毛期望本来为正】时才有意义。
        #   基数为负或接近零时，`1 - uni/all` 会算出没有意义的数
        #   （实测过一次：−0.0816% → +0.0323% 被报成「缩水 140%」，
        #    其实是变号变好了）。所以这里只在基数为正且不接近零时计算。
        base = a["gross_all"]
        shrink = (1 - a["gross_uni"] / base) if base > 1e-4 else 0.0
        if flipped:
            return AuditResult(
                name=self.name, verdict=Verdict.FAILED,
                headline="按重叠计数时是盈利的，去掉重叠就变亏 —— 原数字是灌水来的",
                detail=det + ["这正是 2026-09-02 那次差点误判『宽屏障能盈利』的原因。"],
                numbers=a)
        if shrink > _INFLATION_WARN:
            return AuditResult(
                name=self.name, verdict=Verdict.FAILED,
                headline=f"去掉重叠后毛期望缩水 {shrink*100:.0f}% —— 原数字被信号持续期加权了",
                detail=det, numbers=a)
        if base <= 1e-4:
            return AuditResult(
                name=self.name, verdict=Verdict.CLEAN,
                headline="重叠没有把结论撑起来（毛期望本来就不为正，无从灌水）",
                detail=det, numbers=a)
        return AuditResult(
            name=self.name, verdict=Verdict.CLEAN,
            headline=f"重叠影响有限：去重后毛期望变化 {(-shrink)*100:+.0f}%",
            detail=det, numbers=a)
