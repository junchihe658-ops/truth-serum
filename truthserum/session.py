"""会话级搜索日志 —— 让审计器记住「你这个会话试过几个策略」

## 为什么这一层是整个项目的关键

Python 里那个 `TunerAgent` 只是个离线演示：调整规则是 if/elif 查表，
同样的输入永远走同样的路。叫它 agent 是抬举它。

真正的 agent 是**正在使用这个工具的那个模型**。它读懂闸门的反馈、
自己想出下一个策略、再试一次 —— 这是真推理，不是查表。

而它一旦这么做，就落进了 ⑤ 号闸门要抓的那件事里：
**「我试了很多个，这个最好」**。

所以这一层做的事很简单：把一个会话里所有的审计请求记下来，
让 ⑤ 号闸门能对着【累计次数】说话，而不是只看最后交上来的那一个。

一个专治「试很多次挑最好」的工具，抓住的第一个对象，
就是正在用它的那个 agent。

## 为什么按「配置」分组

同一个会话里可能先审 BTC 的 1h，再审 ETH 的 4h ——
这两组成绩根本不可比，混在一起算选择偏差是错的。
所以按 (标的, 周期, 屏障, 时长, 费率) 分组，只有完全同配置的尝试才累计。

## 什么会重置

`reset()` 会清空。清空是合理的（换课题了），但它会让选择偏差从视野里消失，
所以工具里必须把这句话说出来 —— 不能让人在不知情的情况下把证据抹掉。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import SearchLog, barrier_outcomes, net_expectancy


@dataclass
class Attempt:
    """一次审计请求。score 是去重叠后的净每笔期望（%），与 ②③⑤ 口径一致。"""
    n: int                    # 这是本配置下的第几次
    label: str                # 试的是什么
    score: float
    via: str                  # 走的哪个入口：自然语言 / 代码


@dataclass
class _Group:
    key: tuple
    attempts: list[Attempt] = field(default_factory=list)
    #: 目前最好那次的信号。⑤ 号闸门要拿它构造本底 —— 必须是【胜出者】的，
    #: 用当次提交的那个会让 p 值跨次不可比。
    best_sig: dict | None = None
    best_score: float = float("-inf")


class SessionSearch:
    """一个 MCP 会话内的搜索记录。进程活着它就活着，进程退出就没了 ——
    这正是「会话级」该有的语义，不需要落盘。
    """

    def __init__(self):
        self._groups: dict[tuple, _Group] = {}

    # ── 记录 ────────────────────────────────────────────────
    @staticmethod
    def key_of(symbols, interval, barrier_mult, horizon, fee) -> tuple:
        return (tuple(sorted(symbols)), interval,
                round(float(barrier_mult), 6), int(horizon), round(float(fee), 8))

    def score_of(self, bars, fn, barrier_mult, horizon, cost):
        """按 ②③⑤ 的统一口径给一次尝试打分。返回 (分数, 信号)。

        信号要一起返回，因为如果这次是新的最好成绩，⑤ 号闸门之后要拿它
        构造本底 —— 那时候策略函数已经不在手边了。
        """
        vals, sigs = [], {}
        for s, df in bars.items():
            pre = barrier_outcomes(df, barrier_mult, horizon)[:2]
            z = np.asarray(fn(df), dtype=float).reshape(-1)
            sigs[s] = z
            v = net_expectancy(df, z, barrier_mult, horizon, cost, pre=pre)
            if np.isfinite(v):
                vals.append(v)
        return (float(np.mean(vals)) if vals else float("nan")), sigs

    def record(self, key: tuple, label: str, score: float, via: str,
               sig: dict | None = None) -> Attempt:
        g = self._groups.setdefault(key, _Group(key))
        a = Attempt(n=len(g.attempts) + 1, label=label, score=score, via=via)
        g.attempts.append(a)
        if sig is not None and np.isfinite(score) and score > g.best_score:
            g.best_score, g.best_sig = score, sig
        return a

    # ── 取用 ────────────────────────────────────────────────
    def attempts(self, key: tuple) -> list[Attempt]:
        g = self._groups.get(key)
        return list(g.attempts) if g else []

    def as_search_log(self, key: tuple) -> SearchLog | None:
        """攒够 2 次才有意义：N=1 时「随机试 1 次的最好成绩」就是随机试 1 次，
        这个比较不含任何信息。这是统计上的下限，不是为了凑演出效果 ——
        到底报不报警完全由 p 值决定，不设任何「第几次才触发」的机关。
        """
        att = [a for a in self.attempts(key) if np.isfinite(a.score)]
        if len(att) < 2:
            return None
        best = max(att, key=lambda a: a.score)
        return SearchLog(
            n_trials=len(att),
            scores=[a.score for a in att],
            best_score=best.score,
            best_label=f"第 {best.n} 次：{best.label}",
            space=f"本会话在同一配置下累计提交的 {len(att)} 个策略",
            best_signal=self._groups[key].best_sig,
            n_submitted=len(self.attempts(key)))

    def other_groups(self, key: tuple) -> list[tuple]:
        """别的配置下有没有记录 —— 用来防止「查错配置」被误读成「没试过」"""
        return [k for k in self._groups if k != key and self._groups[k].attempts]

    def render(self, key: tuple) -> str:
        att = self.attempts(key)
        if not att:
            # ⚠ 「这个配置下没有记录」和「你什么都没试过」是两回事。
            #   只说前一句,一个 agent 很容易读成后一句,以为选择偏差不存在 ——
            #   而它可能只是把标的写少了一个。这正是本项目最该防的那类误导。
            other = self.other_groups(key)
            if not other:
                return "本会话还没有任何审计记录。"
            lines = ["本会话在【这个配置】下还没有审计记录。",
                     "",
                     "⚠ 但别的配置下有 —— 不要读成「什么都没试过」："]
            for k in other:
                syms, iv, bm, hz, _fee = k
                lines.append(f"  · {'、'.join(syms)} {iv} 屏障{bm} 持{hz}根："
                             f"{len(self._groups[k].attempts)} 次")
            lines.append("")
            lines.append("查询参数必须和审计时完全一致，否则查到的是另一组。")
            return "\n".join(lines)
        best = max((a for a in att if np.isfinite(a.score)),
                   key=lambda a: a.score, default=None)
        lines = [f"本会话在这个配置下已经试过 {len(att)} 个策略："]
        for a in att:
            mark = " ← 目前最好" if best and a.n == best.n else ""
            sc = f"{a.score:+.4f}%/笔" if np.isfinite(a.score) else "（无法评分）"
            lines.append(f"  {a.n:>2}. {sc}   {a.label[:52]}   [{a.via}]{mark}")
        scorable = [a for a in att if np.isfinite(a.score)]
        if len(scorable) < len(att):
            lines.append("")
            lines.append(f"其中 {len(att) - len(scorable)} 次因成交笔数不足无法评分，"
                         f"实际参与比较的是 {len(scorable)} 次 ——")
            lines.append("  ⑤ 号闸门按【能参与比较的】次数算，"
                         "因为 best-of-N 的 N 就该是能竞争的候选数。")

        other = self.other_groups(key)
        if other:
            # ⚠ 改一下手续费或屏障参数，配置键就变了，计数从头开始。
            #   那是一条真实存在的规避路径 —— 不写出来等于留着不说。
            n_other = sum(len(self._groups[k].attempts) for k in other)
            lines.append("")
            lines.append(f"⚠ 本会话在另外 {len(other)} 个配置下还试过 {n_other} 次。")
            lines.append("  改动标的、周期、屏障或费率都会让计数【从头开始】——"
                         "这是一条真实存在的规避路径，所以在这里说明。")

        if len(att) >= 2:
            lines.append("")
            lines.append("⚠ 试的次数本身就是信息。「我试了 N 个，这个最好」和"
                         "「我只试了这一个」，成绩一样、可信度差着数量级 ——")
            lines.append("  ⑤ 号闸门就是拿这个次数在算。")
        return "\n".join(lines)

    def reset(self) -> int:
        n = sum(len(g.attempts) for g in self._groups.values())
        self._groups.clear()
        return n


#: MCP server 进程内唯一的一份。会话结束、进程退出，它就没了。
SESSION = SessionSearch()
