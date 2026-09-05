"""模拟一个 agent 连续尝试多个策略，看 ⑤ 号闸门什么时候翻脸

这是「评审自己的 Claude 就是那个 agent」那条路径的端到端验证。
真实场景里这五个策略是模型自己想出来的；这里写死只是为了可复现地测。

⚠ 什么时候报警完全由 p 值决定。这个脚本【不预设】它第几次翻脸 ——
  真跑出来是几次就是几次，跑完不报警也照实说。
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from truthserum import server as S
from truthserum.session import SESSION

SYMS = ["BTCUSDT", "ETHUSDT"]

#: 一个 agent 会怎么一步步"改进"：每次都是对上一次的合理调整
TRIES = [
    "RSI 超过 60 做多、低于 40 做空，持 12 小时",
    "RSI 超过 65 做多、低于 35 做空，持 12 小时",
    "RSI 超过 70 做多、低于 30 做空，持 12 小时",
    "RSI(21) 超过 70 做多、低于 30 做空，持 12 小时",
    "RSI(21) 超过 72 做多、低于 28 做空，持 12 小时",
    "EMA12 上穿 EMA48 做多，EMA12 下穿 EMA48 做空，持 12 小时",
]

SESSION.reset()
t0 = time.time()
print("=" * 76)
print("模拟：一个 agent 连续提交策略，每次都想让数字更好看")
print("=" * 76)

for i, desc in enumerate(TRIES, 1):
    print(f"\n【第 {i} 次】{desc}")
    out = S.audit_plain_language(desc, SYMS, confirmed=True)

    m = re.search(r"本会话在这个配置下已经试过 (\d+) 个策略", out)
    g5 = re.search(r"[⛔❌✅⏭]\s+⑤[^\n]*\n\s+([^\n]+)", out)
    v = re.search(r"(❌ 这个数字不可信[^\n]*|✅[^\n]*全部通过[^\n]*|⛔[^\n]*)", out)
    print(f"    会话累计 {m.group(1) if m else '?'} 次")
    print(f"    ⑤ 号：{g5.group(1).strip() if g5 else '（未产出）'}")
    print(f"    整体：{v.group(1).strip()[:60] if v else '?'}")

print()
print("=" * 76)
print("会话记录：")
key = SESSION.key_of(SYMS, "1h", 1.5, 12, 0.0005)
print(SESSION.render(key))
print("=" * 76)
print(f"总耗时 {time.time()-t0:.0f} 秒")
