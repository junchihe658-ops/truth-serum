"""Truth Serum —— 一个不肯对你撒谎的回测审计器

    from truthserum import check
    report = check(bars, strategy)
    print(report.render())

四道闸门，每一道都必须先证明自己能抓到人为植入的 bug，
否则拒绝输出任何结论。
"""
from .audit import Audit, AuditResult, TruthReport, Verdict
from .core import Context, Costs, FuncStrategy, Strategy
from .runner import check

__all__ = ["check", "Context", "Costs", "FuncStrategy", "Strategy",
           "Audit", "AuditResult", "TruthReport", "Verdict"]
__version__ = "0.1.0"
