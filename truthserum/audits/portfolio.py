"""组合层模拟 —— 账户里最后到底剩多少钱

## 为什么单笔期望不作数

单笔 +0.11% 听起来能赚，但账户不是这么运作的：

  · 多个标的【共用一个账户】，不是每个标的一个独立资金池
  · 最多同时持 N 仓 —— 信号来了没仓位就得放弃
  · 仓位按【当前权益】算，会复利，也会被回撤压缩
  · 一笔占住仓位期间，同标的的后续信号全部作废

## 真实案例（2026-09-02）

同一个策略：

    单笔期望（含重叠）     +0.0945%/笔     → 「能赚」
    组合模拟（真实约束）   平均折收益 −8.76%，8 折里 1 折盈利，折合年化 −42%

**中间指标会骗人，账户余额不会。**

## 自检：抛硬币必须亏掉恰好一个成本

一个零 edge 的随机策略，每笔期望的理论值就是 −c（成本）。
如果模拟器给出别的数，说明成本没被正确扣、或者仓位/复利算错了。
这是一个可以精确验证的不变量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..audit import Audit, AuditResult, SelfCheckFailed, Verdict
from ..core import atr, fold_bounds

INIT_EQUITY = 10_000.0


def simulate(ctx, sig_by_sym: dict[str, np.ndarray], i0: int, i1: int,
             grid: pd.DatetimeIndex, mats: dict):
    """逐根 K 线推进的组合模拟。返回 (期末权益, 笔数, 胜数, 最大回撤%)"""
    C, H, L, A = mats["C"], mats["H"], mats["L"], mats["A"]
    syms = ctx.symbols
    ns = len(syms)
    Z = np.vstack([sig_by_sym[s] for s in syms])
    eq = INIT_EQUITY
    side = np.zeros(ns, np.int8)
    ent = np.zeros(ns); tp = np.zeros(ns); sl = np.zeros(ns)
    notional = np.zeros(ns); t_open = np.zeros(ns, np.int64)
    ntr = nwin = 0
    peak, mdd = eq, 0.0
    cost = ctx.costs.round_trip
    for g in range(i0, i1):
        # ── 先出场 ──
        for s in range(ns):
            if side[s] == 0 or not np.isfinite(C[s, g]):
                continue
            ex = np.nan
            if side[s] == 1:
                if L[s, g] <= sl[s]: ex = sl[s]
                elif H[s, g] >= tp[s]: ex = tp[s]
            else:
                if H[s, g] >= sl[s]: ex = sl[s]
                elif L[s, g] <= tp[s]: ex = tp[s]
            if not np.isfinite(ex) and (g - t_open[s]) >= ctx.horizon:
                ex = C[s, g]
            if np.isfinite(ex):
                r = side[s] * (ex - ent[s]) / ent[s] - cost
                eq += notional[s] * r
                ntr += 1; nwin += int(r > 0)
                side[s] = 0
                peak = max(peak, eq)
                if peak > 0:
                    mdd = min(mdd, (eq - peak) / peak)
        # ── 再入场（用出场后的权益）──
        nopen = int((side != 0).sum())
        if nopen >= ctx.max_positions:
            continue
        cand = [(abs(Z[s, g]), s) for s in range(ns)
                if side[s] == 0 and np.isfinite(Z[s, g]) and Z[s, g] != 0
                and np.isfinite(A[s, g]) and A[s, g] > 0 and np.isfinite(C[s, g])]
        cand.sort(reverse=True)
        for _, s in cand:
            if nopen >= ctx.max_positions:
                break
            c0, d = C[s, g], (1 if Z[s, g] > 0 else -1)
            b = ctx.barrier_mult * A[s, g]
            side[s] = d; ent[s] = c0
            tp[s] = c0 + d * b; sl[s] = c0 - d * b
            notional[s] = eq * ctx.pos_pct * ctx.leverage
            t_open[s] = g
            nopen += 1
    return eq, ntr, nwin, mdd * 100


def _build(ctx):
    grid = None
    for s in ctx.symbols:
        idx = ctx.bars[s].index
        grid = idx if grid is None else grid.union(idx)
    def mat(col, fn=None):
        rows = []
        for s in ctx.symbols:
            v = ctx.bars[s][col].reindex(grid).to_numpy(float)
            rows.append(fn(ctx.bars[s]) if fn else v)
        return np.vstack(rows)
    A = np.vstack([pd.Series(atr(ctx.bars[s])).set_axis(ctx.bars[s].index)
                   .reindex(grid).to_numpy(float) for s in ctx.symbols])
    return grid, {"C": mat("close"), "H": mat("high"), "L": mat("low"), "A": A}


def _sig_on_grid(ctx, strategy, grid):
    out = {}
    for s in ctx.symbols:
        z = strategy.signal(ctx.bars[s])
        out[s] = z.reindex(grid).fillna(0).to_numpy(float)
    return out


def _run_folds(ctx, strategy):
    grid, mats = _build(ctx)
    sig = _sig_on_grid(ctx, strategy, grid)
    rets, dds, trades = [], [], 0
    for i0, i1 in fold_bounds(grid, ctx.n_folds):
        eq, ntr, nw, mdd = simulate(ctx, sig, i0, i1, grid, mats)
        rets.append((eq / INIT_EQUITY - 1) * 100)
        dds.append(mdd); trades += ntr
    R = np.array(rets)
    sd = R.std(ddof=1) if R.size > 1 else 0.0
    t = R.mean() / (sd / np.sqrt(R.size)) if sd else 0.0
    return {"mean": float(R.mean()), "t": float(t),
            "profitable": int((R > 0).sum()), "folds": int(R.size),
            "compound": float((np.prod(1 + R / 100) - 1) * 100),
            "worst_dd": float(min(dds)) if dds else 0.0, "trades": trades}


class PortfolioAudit(Audit):
    name = "④ 组合层模拟（账户里最后剩多少钱）"
    catches = "用单笔期望冒充账户收益 —— 忽略共用资金池、仓位上限、复利与回撤"

    def _self_check(self, ctx) -> str:
        """抛硬币策略：每笔期望的理论值必须 ≈ −成本"""
        rng = np.random.default_rng(ctx.seed + 1)

        class _Coin:
            name = "coinflip"
            def signal(self, b):
                return pd.Series(rng.choice([-1.0, 1.0], len(b)), index=b.index)

        r = _run_folds(ctx, _Coin())
        if r["trades"] < 50:
            raise SelfCheckFailed(f"抛硬币只成交 {r['trades']} 笔，样本不足以验证")
        # 每笔平均收益率（相对名义）≈ 账户收益 / (笔数 × 名义占比)
        per_trade = (r["mean"] / 100) / max(r["trades"] / ctx.n_folds, 1) \
            / (ctx.pos_pct * ctx.leverage)
        cost = ctx.costs.round_trip
        if not (-cost * 3 < per_trade < cost * 0.5):
            raise SelfCheckFailed(
                f"抛硬币的每笔期望应 ≈ −{cost*100:.4f}%（纯成本），"
                f"实测 {per_trade*100:+.4f}% —— 成本或仓位模型算错了")
        return (f"抛硬币策略 {r['trades']} 笔，每笔 {per_trade*100:+.4f}% "
                f"≈ −成本 {cost*100:.4f}% —— 成本与仓位模型正确")

    def _run(self, ctx) -> AuditResult:
        r = _run_folds(ctx, ctx.strategy)
        det = [
            f"{r['folds']} 折走向前：平均折收益 {r['mean']:+.2f}%，"
            f"盈利折 {r['profitable']}/{r['folds']}，t = {r['t']:+.2f}",
            f"复合收益 {r['compound']:+.1f}%，最差回撤 {r['worst_dd']:+.1f}%，"
            f"共 {r['trades']} 笔",
            f"约束：共用账户、最多 {ctx.max_positions} 仓、"
            f"单笔名义 = 权益 × {ctx.pos_pct} × {ctx.leverage} 倍杠杆、"
            f"成本 {ctx.costs.describe()}",
        ]
        if r["trades"] < 100:
            det.append(f"⚠ 仅 {r['trades']} 笔，统计功效很低，正负都不足以下结论。")
        if r["mean"] <= 0:
            return AuditResult(
                name=self.name, verdict=Verdict.FAILED,
                headline=f"账户层面亏损：平均折收益 {r['mean']:+.2f}%",
                detail=det, numbers=r)
        if r["t"] < 2.0:
            return AuditResult(
                name=self.name, verdict=Verdict.FAILED,
                headline=f"账户层面为正但不显著（t = {r['t']:+.2f}，需 ≥ 2.0）",
                detail=det + ["折间波动太大，这个正收益无法与运气区分。"],
                numbers=r)
        return AuditResult(
            name=self.name, verdict=Verdict.CLEAN,
            headline=f"账户层面盈利且显著：平均折收益 {r['mean']:+.2f}%，t = {r['t']:+.2f}",
            detail=det, numbers=r)
