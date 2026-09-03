"""审计框架 —— 每个审计都必须先证明自己能抓到已知的 bug

## 为什么这么设计

一个永远报「一切正常」的检测器，比没有检测器更危险 —— 它会让你放心地
在错误的结论上继续投入。

2026-09-02 的真实经历：作者写通用前瞻检测器时连错三版，每一版都会在
真实数据上报告「未发现前瞻」。三次都是被自检拦下的：

  · v1 按【标签时间】截断 —— 但泄露藏在行的【内容】里（4h bar 的标签
    在过去、内容伸向未来），按标签截根本删不掉它
  · v2 把 K 线窗口截在切点 —— 而标签需要 12 根前瞻，末尾 12 行会被
    dropna 丢掉；泄露只影响最后 4 行，正好落在被丢掉的区间里
  · v3 直接拿返回值的索引做 intersection —— 那是【行号】不是时间戳，
    两个 frame 行数一不同就整体错位，报出一堆假阳性

**所以这里把「自检」做成了结构性要求，不是可选项。**
`Audit.report()` 在 `_self_check()` 通过之前，拒绝返回任何结论。
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    CLEAN = "clean"          # 查了，没发现问题
    FAILED = "failed"        # 查出问题
    UNUSABLE = "unusable"    # 检测器自己不可信 —— 不给结论
    SKIPPED = "skipped"      # 前置条件不满足


@dataclass
class AuditResult:
    name: str
    verdict: Verdict
    headline: str
    detail: list[str] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    #: 自检证据：这个检测器在【已知有 bug】的输入上抓到了什么
    self_check_evidence: str = ""

    @property
    def trustworthy(self) -> bool:
        return self.verdict is not Verdict.UNUSABLE

    def render(self, width: int = 78) -> str:
        icon = {Verdict.CLEAN: "✅", Verdict.FAILED: "❌",
                Verdict.UNUSABLE: "⛔", Verdict.SKIPPED: "⏭"}[self.verdict]
        out = [f"{icon}  {self.name}", f"    {self.headline}"]
        if self.self_check_evidence:
            out.append(f"    ├ 自检: {self.self_check_evidence}")
        for d in self.detail:
            wrapped = textwrap.wrap(d, width - 6) or [""]
            for i, line in enumerate(wrapped):
                out.append(("    ├ " if i == 0 else "    │ ") + line)
        return "\n".join(out)


class SelfCheckFailed(RuntimeError):
    """检测器在已知有 bug 的输入上没抓到东西 —— 它自己坏了"""


class Audit:
    """所有审计的基类。

    子类实现两个方法：
      · `_self_check(ctx)` 构造一个【已知有 bug】的输入，返回抓到的证据字符串；
                           抓不到就 raise SelfCheckFailed
      · `_run(ctx)`        在真实输入上跑，返回 AuditResult

    调用方只能用 `report(ctx)` —— 它保证自检先于结论。
    """

    name: str = "unnamed"
    #: 这个审计声称能抓到什么。写清楚，评审和使用者都要能看懂。
    catches: str = ""

    def _self_check(self, ctx) -> str:
        raise NotImplementedError

    def _run(self, ctx) -> AuditResult:
        raise NotImplementedError

    def report(self, ctx) -> AuditResult:
        try:
            evidence = self._self_check(ctx)
        except SelfCheckFailed as e:
            return AuditResult(
                name=self.name, verdict=Verdict.UNUSABLE,
                headline="检测器自检未通过 —— 不给出任何结论",
                detail=[f"它本该抓到人为植入的 bug，但没抓到：{e}",
                        "在检测器自己被修好之前，它说『干净』是没有意义的。"])
        except Exception as e:          # 自检本身崩了，同样不可信
            return AuditResult(
                name=self.name, verdict=Verdict.UNUSABLE,
                headline=f"检测器自检过程异常：{type(e).__name__}: {e}",
                detail=["自检跑不起来，就无法证明这个检测器有效。"])

        res = self._run(ctx)
        res.self_check_evidence = evidence
        return res


@dataclass
class TruthReport:
    """一次完整体检的结果。

    ⚠ 只要有任何一项 UNUSABLE，整体就【不给结论】——
      因为无法区分「真的干净」和「检测器瞎了」。
    """
    strategy_name: str
    results: list[AuditResult]
    claimed: str = ""        # 策略自称的成绩，例如 "年化 +212%"

    @property
    def any_unusable(self) -> bool:
        return any(r.verdict is Verdict.UNUSABLE for r in self.results)

    @property
    def failures(self) -> list[AuditResult]:
        return [r for r in self.results if r.verdict is Verdict.FAILED]

    def verdict_line(self) -> str:
        if self.any_unusable:
            return ("⛔ 无法判定：有检测器自检未通过。"
                    "我们无法区分『真的干净』和『检测器瞎了』。")
        if self.failures:
            names = "、".join(r.name for r in self.failures)
            return f"❌ 这个数字不可信：{len(self.failures)} 项审计未通过（{names}）"
        return "✅ 四项审计全部通过 —— 经得起我们已知的所有自欺检验"

    def render(self) -> str:
        bar = "═" * 78
        out = [bar, f"  策略体检报告：{self.strategy_name}", bar]
        if self.claimed:
            out += [f"  策略自称：{self.claimed}", ""]
        for r in self.results:
            out += [r.render(), ""]
        out += [bar, "  " + self.verdict_line(), bar]
        return "\n".join(out)
