"""Truth Serum —— 一个不肯对你撒谎的回测审计器

    from truthserum import check
    report = check(bars, strategy)
    print(report.render())

也能直接从大白话开始：

    from truthserum import parse
    spec = parse("RSI 超过 70 做空、低于 30 做多、持 12 小时")
    print(spec.explain())        # 先把解读回读给人核对，再往下走
    report = check(bars, spec.to_strategy())

五道闸门，每一道都必须先证明自己能抓到人为植入的 bug，
否则拒绝输出任何结论。
"""
from .audit import Audit, AuditResult, TruthReport, Verdict
from .core import Context, Costs, FuncStrategy, Strategy
from .nl import CannotParse, Spec, parse
from .runner import check

__all__ = ["check", "parse", "Spec", "CannotParse",
           "Context", "Costs", "FuncStrategy", "Strategy",
           "Audit", "AuditResult", "TruthReport", "Verdict"]
__version__ = "0.2.0"
