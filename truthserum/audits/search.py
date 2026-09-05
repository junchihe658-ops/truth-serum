"""⑤ 搜索选择偏差 —— 这个成绩是不是「搜出来的」

## 为什么 ③ 号闸门盖不住这一层

③ 号把【单个】策略的信号打乱，比它自己的本底。
但 agent 搜两百组参数挑最好的那个时，选择偏差发生在「搜了两百次」这一层 ——
被挑中的那组相对它自己的打乱本底照样可能显著，③ 完全看不见。

「我试了两百组，这组最好」和「我只试了这一组，它就是好」，
成绩可能一模一样，可信度差着数量级。差别藏在没交出来的那 199 组里。

## 判据

拿【随机搜同样多次能得到的最好成绩】当本底：

    真实最好  vs  随机搜 N 次的最好成绩分布
    比不过 → 你的成绩可以用「搜得多」解释，不是策略好

## 自检必须验两件事，缺一件这道闸门就没有意义

  1. 纯噪声搜 N 次挑最好  → 必须【拦下】（否则它抓不到东西）
  2. 真有优势的策略        → 必须【放行】（否则它见谁拦谁）

只验第一件是不够的：一个无脑对所有东西报警的闸门，同样能通过第一件。
这两件是 2026-09-05 建这道闸门前专门做的可行性实验的判据，原样搬进自检。

## 已知偏差，写在这里不藏着

本底用的是 N 次【独立】随机打乱，而真实搜索的 N 组参数往往【高度相关】
（RSI 阈值差 5 的两组信号几乎一样）。有效独立试验数远小于 N，
所以本底偏严、这道闸门偏向于报警。搜索空间越是相关，越要保守解读它的结论。
"""
from __future__ import annotations

import numpy as np

from ..audit import Audit, AuditResult, SelfCheckFailed, Verdict
from ..core import barrier_outcomes, net_expectancy

#: 本底抽样轮数。每轮 = 从成绩池里有放回抽 N 次取最好
DRAWS = 40
#: 自检用的规模，比正式跑小，够判方向就行
SELF_TRIALS, SELF_DRAWS = 20, 30
BLOCK = 24          # 块打乱的块长（根）


class SearchBiasAudit(Audit):
    name = "⑤ 搜索选择偏差（这个成绩是不是搜出来的）"
    catches = ("「我试了两百组，这组最好」—— 选择偏差藏在没交出来的那 199 组里；"
               "③ 号只查单个策略，盖不住这一层")

    # ── 内部工具 ────────────────────────────────────────────
    def _pre(self, ctx):
        """屏障结果只依赖 K 线、不依赖信号 —— 算一次，几百次打分复用"""
        if not hasattr(self, "_cache"):
            self._cache = {s: barrier_outcomes(df, ctx.barrier_mult,
                                               ctx.horizon)[:2]
                           for s, df in ctx.bars.items()}
        return self._cache

    def _score(self, ctx, sig_by_sym) -> float:
        pre = self._pre(ctx)
        vals = [net_expectancy(ctx.bars[s], z, ctx.barrier_mult, ctx.horizon,
                               ctx.costs.round_trip, pre=pre[s])
                for s, z in sig_by_sym.items()]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def _shuffle(self, rng, z):
        n = len(z)
        blocks = [z[i:i + BLOCK] for i in range(0, n, BLOCK)]
        rng.shuffle(blocks)
        return np.concatenate(blocks)[:n]

    def _null_best(self, ctx, base_sig, n_trials, draws, rng, pool=None):
        """随机搜 n_trials 次取最好，重复 draws 轮 → 本底分布。

        ⚠ 这里【不能】走「先采样一个池子、再有放回重抽」的捷径。
          重抽出来的最大值永远不会超过池子里的最大值 —— 真实胜出者一旦
          高于池子最大值，p 就自动等于 0，闸门直接失去判别力。
          2026-09-05 第一版就栽在这儿，是自检把它拦下来的。

        所以老实抽：每一轮独立生成 n_trials 个随机信号，取最好。
        代价是慢（draws × n_trials 次打分），只能靠压小 draws 来平衡。
        """
        out = []
        for _ in range(draws):
            best = -np.inf
            for _ in range(n_trials):
                sh = {s: self._shuffle(rng, z) for s, z in base_sig.items()}
                v = self._score(ctx, sh)
                if np.isfinite(v) and v > best:
                    best = v
            out.append(best)
        return np.asarray(out)

    def _sig_of(self, ctx, strategy):
        return {s: np.asarray(strategy.signal(df), dtype=float).reshape(-1)
                for s, df in ctx.bars.items()}

    # ── 自检 ────────────────────────────────────────────────
    def _self_check(self, ctx) -> str:
        rng = np.random.default_rng(ctx.seed)
        sym0 = ctx.symbols[0]
        n = len(ctx.bars[sym0])

        # 1) 纯噪声搜 SELF_TRIALS 次挑最好 —— 必须被拦下
        noise_best, noise_sig = -np.inf, None
        for _ in range(SELF_TRIALS):
            cand = {s: rng.choice([-1.0, 0.0, 1.0], size=len(df),
                                  p=[0.25, 0.5, 0.25])
                    for s, df in ctx.bars.items()}
            v = self._score(ctx, cand)
            if np.isfinite(v) and v > noise_best:
                noise_best, noise_sig = v, cand
        if noise_sig is None:
            raise SelfCheckFailed("噪声搜索没产出任何有效成绩，样本可能不足")
        null = self._null_best(ctx, noise_sig, SELF_TRIALS, SELF_DRAWS, rng)
        p_noise = float(np.mean(null >= noise_best))
        if p_noise <= 0.05:
            raise SelfCheckFailed(
                f"纯噪声搜 {SELF_TRIALS} 次挑出的最好成绩 {noise_best:+.4f}%，"
                f"本底只有 {p_noise:.3f} 的概率超过它 —— 闸门没能识破选择偏差")

        # 2) 真有优势的策略 —— 必须被放行，否则这闸门见谁拦谁
        oracle = {}
        for s, df in ctx.bars.items():
            c = df["close"].to_numpy(float)
            fut = np.concatenate([c[1:], [np.nan]]) - c
            oracle[s] = np.nan_to_num(np.sign(fut))
        o_score = self._score(ctx, oracle)
        o_null = self._null_best(ctx, oracle, SELF_TRIALS, SELF_DRAWS, rng)
        p_oracle = float(np.mean(o_null >= o_score))
        if p_oracle > 0.05:
            raise SelfCheckFailed(
                f"人为植入的『直接看未来』策略 {o_score:+.4f}% 也被判成"
                f"「可以用搜索解释」（p={p_oracle:.3f}）—— 这道闸门见谁拦谁，没有判别力")

        return (f"双向验证通过：纯噪声搜 {SELF_TRIALS} 次的最好成绩被拦下"
                f"（p={p_noise:.3f}）；人为植入的『直接看未来』策略被放行"
                f"（p={p_oracle:.3f}）—— 既抓得到选择偏差，也不会见谁拦谁")

    # ── 正式检验 ────────────────────────────────────────────
    def _run(self, ctx) -> AuditResult:
        log = getattr(ctx, "search_log", None)
        if log is None or not getattr(log, "n_trials", 0):
            return AuditResult(
                name=self.name, verdict=Verdict.SKIPPED,
                headline="没有搜索日志，无从判断选择偏差",
                detail=["这个策略不是搜出来的，或者搜索过程没有被记录。",
                        "只交出胜出者、不交出试过多少组，选择偏差就查不了 ——"
                        "这不是「没问题」，是「没查」。"])

        rng = np.random.default_rng(ctx.seed)
        sig = self._sig_of(ctx, ctx.strategy)
        real = self._score(ctx, sig)
        if not np.isfinite(real):
            return AuditResult(
                name=self.name, verdict=Verdict.SKIPPED,
                headline="胜出策略的成交笔数不足，无法评估")

        null = self._null_best(ctx, sig, log.n_trials, DRAWS, rng)
        p = float(np.mean(null >= real))

        det = [f"搜了 {log.n_trials} 组，最好 {log.best_score:+.4f}%/笔"
               f"（{log.best_label}），中位 {log.median:+.4f}%",
               f"随机搜同样 {log.n_trials} 次的最好成绩："
               f"中位 {np.median(null):+.4f}%，"
               f"范围 [{null.min():+.4f}%, {null.max():+.4f}%]，"
               f"{DRAWS} 轮抽样",
               f"胜出成绩 {real:+.4f}%/笔，本底超过它的比例 p ≈ {p:.3f}"]
        if log.space:
            det.append(f"搜索空间：{log.space}")
        det.append("⚠ 本底用的是【独立】随机打乱，而搜索的参数组往往高度相关，"
                   "有效独立试验数小于名义次数 —— 这道闸门偏向于报警。")
        nums = {"n_trials": log.n_trials, "best": real,
                "null_median": float(np.median(null)), "p": p}

        if p > 0.05:
            return AuditResult(
                name=self.name, verdict=Verdict.FAILED,
                headline=f"这个成绩可以用「搜得多」解释：随机搜 {log.n_trials} 次"
                         f"也能做到（p ≈ {p:.3f}）",
                detail=det + ["把搜索次数算进去之后，胜出的那组并不比"
                              "「随机试同样多次里最好的那次」更好。"],
                numbers=nums)
        return AuditResult(
            name=self.name, verdict=Verdict.CLEAN,
            headline=f"扣掉搜索次数后仍然显著（p ≈ {p:.3f}）",
            detail=det, numbers=nums)
