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
        self.last_log = log

        if self.verbose:
            print(f"  [{self.name}] 搜完 {len(self.space)} 组。"
                  f"最好 {best:+.4f}%/笔，中位 {log.median:+.4f}%")
            print(f"  [{self.name}] 胜出：{best_c.label}")
            print(f"  [{self.name}] ⚠ 我只知道「数字变好了」。"
                  f"这里面有多少是运气，我自己看不出来 —— 那是审计器的事。")

        return best_c.make(), log

    # ────────────────────────────────────────────────────────
    # 有反馈的循环：这才是「agent」，上面那个只是遍历
    # ────────────────────────────────────────────────────────
    def run_with_feedback(self, bars, rounds: int = 3, check_fn=None,
                          **check_kw):
        """跑闸门 → 读哪道没过 → 据此调整搜索方向 → 再来。

        和 `run()` 的区别很实在：`run()` 是一个 for 循环把网格遍历完，
        没有观察、没有适应，叫它 agent 是抬举它。这个方法才有闭环 ——
        它看得到审计结果，并且**根据具体是哪道闸门没过**改变下一轮怎么搜。

        调整规则是照人调参的习惯写的，每一条都写明了理由：

          ② 重叠没过 → 信号太"黏"，同一段行情被反复计数。
                        把阈值拉开，让触发更稀疏。
          ③ 本底没过 → 和随机分不开，换个 RSI 周期，去别的区域找。
          ④ 账户没过 → 交易太频繁被成本吃掉，进一步收紧阈值。

        ⚠ 关键在于：**每一轮的每一次试验都累加进同一份 SearchLog。**
          分轮搜索最容易产生的错觉是「我这一轮只试了 36 组」——
          可选择偏差是按【累计】试验次数算的，不是按最后一轮。
        """
        from ..runner import check as _default_check
        check_fn = check_fn or _default_check

        pre = {s: barrier_outcomes(df, self.barrier_mult, self.horizon)[:2]
               for s, df in bars.items()}
        periods = [7, 14, 21]
        pi, spread = 1, 0            # 当前用哪个周期、阈值拉开多少
        all_scores, best, best_c, trail = [], -np.inf, None, []
        all_sigs = []          # 供 ⑤ 号估有效独立试验数

        for rd in range(1, rounds + 1):
            his = [60 + spread + k for k in (0, 3, 6)]
            los = [40 - spread - k for k in (0, 3, 6)]
            grid = [Candidate(periods[pi], h, l) for h in his for l in los]

            if self.verbose:
                print(f"\n  [第 {rd} 轮] RSI({periods[pi]})，阈值 "
                      f"{his[0]}~{his[-1]} / {los[-1]}~{los[0]}，共 {len(grid)} 组")

            r_best, r_c = -np.inf, None
            for c in grid:
                fn = c.make()
                vals = []
                for s, df in bars.items():
                    z = np.asarray(fn(df), dtype=float).reshape(-1)
                    v = net_expectancy(df, z, self.barrier_mult, self.horizon,
                                       self.costs.round_trip, pre=pre[s])
                    if np.isfinite(v):
                        vals.append(v)
                sc = float(np.mean(vals)) if vals else float("nan")
                all_scores.append(sc)
                if np.isfinite(sc):
                    all_sigs.append({s2: np.asarray(fn(df2), dtype=np.int8).reshape(-1)
                                     for s2, df2 in bars.items()})
                if np.isfinite(sc) and sc > r_best:
                    r_best, r_c = sc, c
            if r_c is None:
                continue
            if r_best > best:
                best, best_c = r_best, r_c

            if self.verbose:
                print(f"           本轮最好 {r_best:+.4f}%/笔  {r_c.label}")
                print(f"           累计已试 {len(all_scores)} 组")
                print(f"           跑闸门看看…")

            rep = check_fn(bars, r_c.make(), name=r_c.label, **check_kw)
            failed = [x.name.split("（")[0].strip() for x in rep.results
                      if x.verdict.value == "failed"]
            trail.append((rd, len(all_scores), r_best, r_c.label, list(failed)))

            if self.verbose:
                print(f"           没过的：{'、'.join(failed) if failed else '（全过）'}")

            # ── 读结果，改方向 ──
            if not failed:
                if self.verbose:
                    print("           全过了，停。")
                break
            if any("重叠" in f for f in failed):
                spread += 4
                why = "信号太黏，同一段行情被反复计数 → 把阈值拉开，让触发更稀疏"
            elif any("本底" in f for f in failed):
                pi = (pi + 1) % len(periods)
                why = f"和随机分不开 → 换个周期，去 RSI({periods[pi]}) 那片找"
            else:
                spread += 2
                why = "账户还是亏 → 交易太频繁被成本吃掉，再收紧一点"
            if self.verbose and rd < rounds:
                print(f"           → {why}")

        if best_c is None:
            raise RuntimeError("搜索没有产出任何有效结果")

        log = SearchLog(
            n_trials=len(all_scores), scores=all_scores,
            best_score=best, best_label=best_c.label,
            space=f"{rounds} 轮反馈式搜索，累计 {len(all_scores)} 组",
            signals=all_sigs or None)
        self.last_log, self.trail = log, trail

        if self.verbose:
            print(f"\n  [{self.name}] {len(trail)} 轮跑完，累计试了 "
                  f"{len(all_scores)} 组，最好 {best:+.4f}%/笔")
            print(f"  [{self.name}] ⚠ 我每一轮都在「根据反馈改进」。"
                  f"但改进的是【指标】，不是【策略真的变好了】——"
                  f"这两件事我自己分不出来。")
        return best_c.make(), log
