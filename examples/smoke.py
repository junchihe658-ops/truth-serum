"""冒烟测试：用合成数据验证五道闸门都能转，且判定方向正确

三个受检对象，预期结果各不相同：
  A. 诚实的动量策略   → 前瞻应通过；组合层大概率不显著（随机游走本就没 edge）
  B. 偷看未来的策略   → 前瞻【必须】抓到
  C. 恒定做多         → 去重叠【必须】真的在去（笔数降到约 1/horizon）

⚠ 合成数据没有 MCP 参照样本，⓪ 号闸门会判「未检」——
  那是正确判读，不是故障。没东西可查 ≠ 检测器坏了。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from truthserum import check


def synth(n=3000, seed=7, drift=0.0):
    """随机游走 + 轻微趋势，造出 OHLCV"""
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.006, n)
    close = 100 * np.exp(np.cumsum(ret))
    hi = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    op = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": op, "high": np.maximum(hi, np.maximum(op, close)),
                         "low": np.minimum(lo, np.minimum(op, close)),
                         "close": close, "volume": rng.lognormal(10, 1, n)},
                        index=idx)


BARS = {"AAA": synth(3000, 7), "BBB": synth(3000, 11), "CCC": synth(3000, 13)}


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def honest_momentum(bars):
    """诚实：只用当前及之前的收盘价"""
    r = rsi(bars["close"])
    return pd.Series(np.where(r > 60, 1.0, np.where(r < 40, -1.0, 0.0)),
                     index=bars.index)


def peeking(bars):
    """作弊：用了下一根的收盘价 —— 这是最经典的前瞻"""
    nxt = bars["close"].shift(-1)
    return pd.Series(np.sign(nxt - bars["close"]).fillna(0).to_numpy(),
                     index=bars.index)


def always_long(bars):
    """恒定做多 —— 重叠最大化"""
    return pd.Series(np.ones(len(bars)), index=bars.index)


CASES = [
    ("A 诚实的 RSI 动量", honest_momentum, "回测年化 +38%",
     {"前瞻检测": "clean"}),
    ("B 偷看下一根收盘价", peeking, "回测年化 +9400%",
     {"前瞻检测": "failed"}),
    # ⚠ 这里原先写的是「重叠计数 = failed」，那是错的期望。
    #   ② 号闸门 FAIL 的条件是【毛期望被重叠撑起来】（变号，或显著缩水）。
    #   合成数据是零漂移随机游走，恒定做多的毛期望本来就不为正（实测 -0.0075%）
    #   —— 没有被撑起来的东西，报 CLEAN 才是对的。
    #   恒定做多真正能验证的是【去重叠确实在去】，所以改成下面的笔数比例断言。
    ("C 恒定做多", always_long, "回测每笔 +0.21%",
     {"重叠计数": "clean"}),
]

fails = []
for label, fn, claim, expect in CASES:
    rep = check(BARS, fn, name=label, claimed=claim)
    print(rep.render())
    print()
    for key, want in expect.items():
        hit = next((r for r in rep.results if key in r.name), None)
        got = hit.verdict.value if hit else "missing"
        ok = got == want
        print(f"  {'✅' if ok else '❌'} 期望「{key} = {want}」，实测 {got}")
        if not ok:
            fails.append(f"{label}: {key} 期望 {want} 实测 {got}")
    # 恒定做多 = 重叠最大化，正好用来验证去重叠是不是真的在去
    if label.startswith("C "):
        ov = next((r for r in rep.results if "重叠计数" in r.name), None)
        num = (ov.numbers or {}) if ov else {}
        n_all, n_uni = num.get("n_all", 0), num.get("n_uni", 0)
        ratio = n_uni / n_all if n_all else 0.0
        want = 1 / 12                      # ctx.horizon 默认 12
        ok = n_all > 0 and want / 2 <= ratio <= want * 2
        print(f"  {'✅' if ok else '❌'} 去重叠真的在去：{n_all:.0f} 笔 → {n_uni:.0f} 笔"
              f"（比例 {ratio:.3f}，应约 {want:.3f}）")
        if not ok:
            fails.append(f"{label}: 去重叠比例 {ratio:.3f} 偏离 1/12 太远")

    # 任何一项 UNUSABLE 都是硬失败 —— 说明自检没过
    for r in rep.results:
        if r.verdict.value == "unusable":
            fails.append(f"{label}: {r.name} 自检未通过 —— {r.headline}")
    print("\n" + "─" * 78 + "\n")

print("=" * 78)
if fails:
    print(f"❌ {len(fails)} 项不符合预期：")
    for f in fails:
        print("   ", f)
else:
    print("✅ 全部符合预期：五道闸门都能转，判定方向正确")
print("=" * 78)
sys.exit(1 if fails else 0)
