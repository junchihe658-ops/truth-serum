"""Truth Serum 演示 —— 从一句话到「这个数字有多少是真的」

    py examples/demo.py

四幕：
  一  用户说一句大白话，Truth Serum 把解读回读给他确认
  二  用【大多数人的算法】在真实币安行情上算，得到一个漂亮数字
  三  五道闸门逐道过
  四  两个数字并排，逐条说清差在哪

⚠ 全片没有一个编造的数字。第二幕那个漂亮数字是 truthserum/naive.py 在同一份
  真实行情、同一段策略代码上真算出来的 —— 只不过用的是错的算法。
  一个讲「不要自欺」的工具，演示里摆个编的数字，一问就塌。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from truthserum import Verdict, check, parse
from truthserum.data import describe, load
from truthserum.naive import best_of, claim_line, naive_backtest
from truthserum.report_html import save_html

SAID = "RSI 超过 60 做多、低于 40 做空，持 12 小时"
SYMS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
W = 74


def act(n, title):
    print()
    print("━" * W)
    print(f"  第{n}幕   {title}")
    print("━" * W)


# ══════════════════════════════════════════════════════════════
act("一", "一句话")

print(f'\n  用户说：「{SAID}」\n')
spec = parse(SAID, "1h")
print("\n".join("  " + l for l in spec.explain().splitlines()))
print("\n  ── 生成的代码（没有模型参与，确定性解析器直接生成）──\n")
print("\n".join("  " + l for l in spec.to_code().rstrip().splitlines()))
fn = spec.to_strategy()

# ══════════════════════════════════════════════════════════════
act("二", "大多数人会得到的数字")

bars, prov = load(SYMS, "1h")
print(f"\n  {describe(prov).splitlines()[0]}")
for line in describe(prov).splitlines()[1:]:
    print(f"  {line}")

rows = naive_backtest(bars, fn, horizon=spec.horizon or 12)
print(f"\n  单仓顺序回测（大多数人自己写的那个循环）：\n")
print(f"  {'标的':<10}{'笔数':>6}{'胜率':>8}{'年化·未扣成本':>16}{'年化·已扣成本':>16}")
print("  " + "─" * (W - 4))
for sym in SYMS:
    g = next((r for r in rows if r.symbol == sym and r.cost_per_trade == 0), None)
    c = next((r for r in rows if r.symbol == sym and r.cost_per_trade > 0), None)
    if g and c:
        print(f"  {sym:<10}{g.trades:>6}{g.win_rate*100:>7.1f}%"
              f"{g.annualized*100:>15.1f}%{c.annualized*100:>15.1f}%")

claim = claim_line(rows)
b_free = best_of(rows, with_costs=False)
b_cost = best_of(rows, with_costs=True)
print(f"\n  他会拿去宣传的那个数字：{claim}")
print(f"\n  ⚠ 这个数字是【真的】—— 真实币安行情、真实策略代码算出来的。")
print(f"    只是算法漏了两件事，第四幕再说。")

# ══════════════════════════════════════════════════════════════
act("三", "五道闸门")

print()
rep = check(bars, fn, name=SAID, claimed=claim,
            horizon=spec.horizon or 12,
            barrier_mult=spec.barrier_mult or 1.5)
print(rep.render())

# ══════════════════════════════════════════════════════════════
act("四", "差在哪")

port = next((r for r in rep.results if r.name.startswith("④")), None)
same = next((r for r in rows                   # 宣传那个币、扣了成本的同一行
             if r.symbol == b_free.symbol and r.cost_per_trade > 0), None)
if same is None:                               # 正常不会发生；发生了就说出来
    raise SystemExit(f"内部不一致：{b_free.symbol} 有未扣成本的结果，"
                     f"却没有已扣成本的对照行")
rt = same.cost_per_trade

print(f"""
  宣传的数字    {claim}
  账户的数字    {port.headline if port else "（④ 号闸门未产出）"}

  差额由两件具体的事构成，都不是玄学：

  1. 漏掉手续费和滑点
     {b_free.symbol} 同一段行情、同一段代码，只差扣不扣成本：
       未扣成本  总收益 {b_free.total_return*100:+.1f}%   年化 {b_free.annualized*100:+.1f}%
       已扣成本  总收益 {same.total_return*100:+.1f}%   年化 {same.annualized*100:+.1f}%
     来回成本 {rt*100:.4f}% × {b_free.trades} 笔 ≈ 累计 {rt*b_free.trades*100:.0f}% 的本金，
     而未扣成本的毛利只有 {b_free.total_return*100:.1f}%。这一项就够把它抹平。

  2. 只报表现最好的那个标的
     同一个策略在 {len(SYMS)} 个币上跑，写进宣传的是最好的那个。
     其余几个见第二幕的表 —— 有的即使不扣成本也是亏的。
""")

# ⚠ 这一段必须从结果算出来，不许写死。
#   初版我硬写了「另外三道闸门这次没有全部触发」，而实际 ②③ 都失败了 ——
#   一个讲「别自欺」的演示里出现和数据对不上的旁白，比什么都难看。
failed = [r for r in rep.results if r.verdict is Verdict.FAILED]
clean = [r for r in rep.results if r.verdict is Verdict.CLEAN]
other = [r for r in rep.results if r.verdict not in (Verdict.FAILED, Verdict.CLEAN)]

print(f"  五道闸门这一趟的实际结果：{len(failed)} 项异常，{len(clean)} 项未见异常"
      + (f"，{len(other)} 项未检/无效" if other else "") + "\n")
for r in failed:
    print(f"    ✗ {r.name.split('（')[0].strip():<16}{r.headline}")
for r in clean:
    print(f"    ✓ {r.name.split('（')[0].strip():<16}{r.headline}")
for r in other:
    print(f"    · {r.name.split('（')[0].strip():<16}{r.headline}")
print("""
  「未见异常」这几个字有分量，是因为那道闸门在给结论之前，
  先在【人为植入的 bug】上证明过自己抓得到 —— 自检证据就印在报告里。
  一个永远报平安的检测器，比没有检测器更危险。
""")

out = save_html(rep, "reports/demo.html",
                data_note=describe(prov).replace("\n", " | "))
print(f"  HTML 报告 → {out}")
print("━" * W)
