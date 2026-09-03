"""验证自然语言层 —— 重点在【该拒绝的时候拒绝】

这一层的价值不在于能翻译多少句子，而在于翻译不了的时候不装懂。
所以拒绝用例比成功用例更重要，写在前面。

不联网、不读缓存，全部用合成 K 线，跑完两秒。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from truthserum.indicators import TOOLBOX, ema, rsi, sma
from truthserum.nl import CannotParse, parse

FAIL = []


def ck(name, cond, detail=""):
    if not cond:
        FAIL.append(name)
    print(f"  {'✅' if cond else '❌'} {name}" + (f"   {detail}" if detail else ""))


def refuses(name, text, must_mention="", interval="1h"):
    """必须拒绝，而且必须说清楚是哪一段没看懂"""
    try:
        spec = parse(text, interval)
    except CannotParse as e:
        ok = (must_mention in str(e)) if must_mention else True
        ck(name, ok, f"→ {str(e).splitlines()[0]}")
        return
    ck(name, False, f"没有拒绝，反而解析成了 {len(spec.rules)} 条规则")


def bars_synth(n=600, seed=7):
    """一段确定性的合成行情。用正弦 + 噪声，保证 RSI 会真的穿越 30/70。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    px = 100 * (1 + 0.06 * np.sin(t / 23) + 0.03 * np.sin(t / 7)
                + np.cumsum(rng.normal(0, 0.0012, n)))
    hi = px * (1 + rng.uniform(0, 0.004, n))
    lo = px * (1 - rng.uniform(0, 0.004, n))
    return pd.DataFrame(
        {"open": px, "high": hi, "low": lo, "close": px,
         "volume": rng.uniform(50, 150, n)},
        index=pd.date_range("2026-01-01", periods=n, freq="h"))


BARS = bars_synth()

print("=" * 74)
print("【1】该拒绝的必须拒绝 —— 这一层的立身之本")
print("=" * 74)

refuses("词汇表外的指标不装懂", "MACD 金叉就买入", "MACD")
refuses("没消费掉的字必须报出来",
        "RSI 超过 70 并且 月亮是圆的 做多", "月亮")
refuses("只有条件没有动作", "RSI 超过 70")
refuses("只有动作没有条件", "做多")
refuses("一段里两个动作", "RSI 超过 70 做多做空", "动作")
refuses("百分比量给了裸数字不敢猜", "24小时涨幅超过 3 做多", "%")
refuses("非百分比量写了 % 也不敢猜", "RSI 超过 70% 做空", "%")
refuses("比较词缺失", "RSI 70 做空", "比较词")
refuses("空输入", "   ")
refuses("换算下来不足一根", "涨幅超过 3% 做多，持 1 分钟", interval="1h",
        must_mention="0 根")

print()
print("=" * 74)
print("【2】能翻译的要翻译对")
print("=" * 74)

s = parse("RSI 超过 70 做空、低于 30 做多、持 12 小时")
ck("两条规则", len(s.rules) == 2)
ck("动作方向正确", [r.action for r in s.rules] == [-1.0, 1.0],
   str([r.action for r in s.rules]))
ck("时间屏障 12 根（1h 周期）", s.horizon == 12, f"horizon={s.horizon}")
ck("回读里印出了阈值", "RSI(14) > 70" in s.explain() and "RSI(14) < 30" in s.explain())

s4 = parse("RSI 超过 70 做空、低于 30 做多、持 12 小时", "4h")
ck("同一句话在 4h 上是 3 根", s4.horizon == 3, f"horizon={s4.horizon}")
ck("换算写进了回读", "12小时 = 3 根 4h K 线" in s4.explain())

s2 = parse("EMA12 上穿 EMA48 做多，EMA12 下穿 EMA48 做空，止盈止损 2 倍 ATR")
ck("止盈止损倍数", s2.barrier_mult == 2.0, f"barrier={s2.barrier_mult}")
ck("穿越的歧义被回读出来", any("穿越那一根" in w for w in s2.warnings))

s3 = parse("持仓 20 根，RSI 超过 70 做空")
ck("「持仓 20 根」认得出（曾经因正则顺序整条失配）", s3.horizon == 20,
   f"horizon={s3.horizon}")

sw = parse("RSI 超过 ７０ 做空")          # 全角数字
ck("全角数字能归一", len(sw.rules) == 1 and "70" in sw.explain())

sm = parse("24小时涨幅超过 3% 并且 收盘价高于 MA20 做多")
ck("一条规则两个条件", len(sm.rules[0].conds) == 2)

print()
print("=" * 74)
print("【3】生成的代码要跑得对")
print("=" * 74)

fn = parse("RSI 超过 70 做空、低于 30 做多").to_strategy()
out = fn(BARS)
r = rsi(BARS["close"], 14)
want = np.where((r > 70).to_numpy(), -1.0,
                np.where((r < 30).to_numpy(), 1.0, 0.0))
ck("信号与手算一致", np.array_equal(out.to_numpy(), want))
ck("长度等于 K 线数", len(out) == len(BARS))
ck("取值只有 -1/0/1", set(np.unique(out.to_numpy())) <= {-1.0, 0.0, 1.0})
ck("真的产生了非零信号（否则测了个寂寞）", (out != 0).sum() > 20,
   f"非零 {(out != 0).sum()} 根")

# 先命中的生效：RSI>70 时两条规则都满足，必须按写的顺序给 +1
fo = parse("RSI 超过 50 做多、RSI 超过 70 做空").to_strategy()(BARS)
hi = (r > 70).to_numpy()
ck("先写的规则优先（RSI>70 时仍是做多）",
   hi.any() and np.all(fo.to_numpy()[hi] == 1.0), f"命中 {hi.sum()} 根")

# 穿越 vs 水平：穿越只在越过那一根触发，必须严格更少
cross = parse("EMA12 上穿 EMA48 做多").to_strategy()(BARS)
level = parse("EMA12 高于 EMA48 做多").to_strategy()(BARS)
ck("穿越触发次数远少于水平比较",
   0 < (cross != 0).sum() < (level != 0).sum(),
   f"穿越 {(cross != 0).sum()} 根 vs 水平 {(level != 0).sum()} 根")

code = parse("24小时涨幅超过 3% 并且 收盘价高于 MA20 做多").to_code()
ck("生成的代码是合法 Python", compile(code, "<gen>", "exec") is not None)
undefined = [n for n in ("sklearn", "open(", "import ", "__") if n in code]
ck("生成的代码不含额外依赖或导入", not undefined, str(undefined))

print()
print("=" * 74)
print("【4】因果性：截断不变 —— ① 号闸门要查的那个性质")
print("=" * 74)

# 这不能替代 ① 号闸门（真闸门跑在真实行情上、还带自检），
# 但生成器要是连合成数据都过不了，就不必浪费时间去跑真闸门了。
for text in ("RSI 超过 70 做空、低于 30 做多",
             "EMA12 上穿 EMA48 做多",
             "24小时涨幅超过 3% 并且 收盘价高于 MA20 做多"):
    f = parse(text).to_strategy()
    full = f(BARS).to_numpy()
    ok = True
    for cut in (200, 350, 500):
        part = f(BARS.iloc[:cut]).to_numpy()
        if not np.array_equal(part, full[:cut]):
            ok = False
            bad = int(np.argmax(part != full[:cut]))
            ck(f"截断不变：{text[:20]}", False, f"cut={cut} 第 {bad} 根就变了")
            break
    if ok:
        ck(f"截断不变：{text[:24]}", True, "3 个切点全部逐根一致")

print()
print("=" * 74)
print(f"{'✅ 全部通过' if not FAIL else f'❌ {len(FAIL)} 项失败'}")
for f_ in FAIL:
    print("   ", f_)
print("=" * 74)
sys.exit(1 if FAIL else 0)
