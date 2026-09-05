"""agent 自己优化策略 → 数字变好看 → 被自己的审计器当场拆穿

    py examples/agent_demo.py

## 这一幕演的是什么

一个完全"诚实"的 agent：不偷看未来、不改数据、每一步都可复现。
它只干一件事 —— 把「每笔期望」这个指标优化到最高。

它成功了：数字确实变好了。
然后 ⑤ 号闸门告诉它：**随机搜同样多次也能做到**。

这不是 agent 作弊，是**单一指标优化的必然结果**。人类调参时干的是同一件事，
区别只在于人类通常不记录自己试过多少组 —— 于是选择偏差就消失在视野外了。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from truthserum import check
from truthserum.agents import TunerAgent
from truthserum.data import describe, load
from truthserum.indicators import rsi
from truthserum.naive import claim_line, naive_backtest
from truthserum.report_html import save_html

SYMS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
W = 74


def act(n, title):
    print()
    print("━" * W)
    print(f"  {n}   {title}")
    print("━" * W)


bars, prov = load(SYMS, "1h")
t0 = time.time()

# ══════════════════════════════════════════════════════════════
act("第一幕", "人给的起点：一个普通的 RSI 策略")

def baseline(b):
    r = rsi(b["close"])
    return pd.Series(np.where(r > 60, 1.0, np.where(r < 40, -1.0, 0.0)),
                     index=b.index)

base_claim = claim_line(naive_backtest(bars, baseline))
print(f"\n  RSI(14) >60 做多 / <40 做空")
print(f"  按大多数人的算法：{base_claim}")

# ══════════════════════════════════════════════════════════════
act("第二幕", "agent 接手，自己找更好的参数")

print()
agent = TunerAgent()
# 有反馈的循环 —— 跑闸门、读哪道没过、据此改搜索方向，再来一轮。
# 用 run() 也能得到结果，但那只是个 for 循环把网格遍历完，
# 没有观察也没有适应，叫它 agent 是抬举它。
tuned, log = agent.run_with_feedback(bars, rounds=3)

tuned_claim = claim_line(naive_backtest(bars, tuned))
print(f"\n  优化后按同一套算法：{tuned_claim}")

# ══════════════════════════════════════════════════════════════
act("第三幕", "六道闸门")

print()
rep = check(bars, tuned, name=f"agent 调参结果：{log.best_label}",
            claimed=tuned_claim, search_log=log)
print(rep.render())

# ══════════════════════════════════════════════════════════════
act("第四幕", "agent 错在哪")

g5 = next((r for r in rep.results if r.name.startswith("⑤")), None)
print(f"""
  agent 做的事，每一步都是"诚实"的：
    · 没偷看未来   —— ① 号闸门放行
    · 没改数据     —— ⓪ 号闸门放行
    · 每一步可复现 —— 同样的输入永远得到同样的输出

  它唯一做的事情是：**在累计 {log.n_trials} 组参数里挑了最好的那组。**
""")

print("  它每一轮是怎么「改进」的：\n")
for rd, cum, sc, label, failed in getattr(agent, "trail", []):
    print(f"    第 {rd} 轮  累计试了 {cum:>3} 组  最好 {sc:+.4f}%/笔  {label}")
    print(f"            没过：{'、'.join(failed) if failed else '（全过）'}")
print(f"""
  注意累计那一列。分轮搜索最容易骗人的地方就在这儿 ——
  agent 每一轮只觉得「我这轮试了 9 组」，但选择偏差是按【累计】次数算的。

  搜索结果的分布：
    最好   {log.best_score:+.4f}%/笔   （{log.best_label}）
    中位   {log.median:+.4f}%/笔
    也就是说，绝大多数参数组是亏的，它挑出了尾巴上那一个。

  ⑤ 号闸门的判读：
    {g5.headline if g5 else "（未产出）"}
""")

if g5 and g5.self_check_evidence:
    print(f"  这道闸门凭什么这么说 —— 它先证明过自己：")
    print(f"    {g5.self_check_evidence}\n")

print("""  ⚠ 注意这里的不对称：
    agent 看到的是「我把数字从 A 优化到了 B」。
    审计器看到的是「你试了两百次，B 是这两百次里最好的那次」。
    **同一件事，两种描述，结论完全相反。**

    人类调参时干的是同一件事 —— 区别只在于人类通常不记录试过多少组，
    于是这一层偏差直接消失在视野之外。""")

out = save_html(rep, "reports/agent.html", data_note=describe(prov))
print(f"\n  HTML 报告 → {out}")
print(f"  总耗时 {time.time()-t0:.0f} 秒")
print("━" * W)
