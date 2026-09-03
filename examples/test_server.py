"""验证 MCP server 的四个工具都能正常工作

分两层：
  1. 直接调工具函数 —— 验证业务逻辑
  2. 走真实 MCP 协议握手 —— 验证它确实能被别的客户端装上
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FAIL = []


def ck(name, cond, detail=""):
    if not cond:
        FAIL.append(name)
    print(f"  {'✅' if cond else '❌'} {name}" + (f"   {detail}" if detail else ""))


print("=" * 74)
print("【1】直接调工具函数")
print("=" * 74)
from truthserum import server as S

gates = S.list_gates()
ck("list_gates 返回五道闸门", all(g in gates for g in ("⓪", "①", "②", "③", "④")))
ck("说明了自检机制", "自检不过就返回" in gates)

GOOD = '''
def signal(bars):
    r = rsi(bars["close"])
    return pd.Series(np.where(r > 70, -1.0, np.where(r < 30, 1.0, 0.0)),
                     index=bars.index)
'''
BAD_SYNTAX = "def signal(bars) return 1"          # 真的语法错误
NO_SIGNAL = "x = 1"
# ⚠ 注意 `def signal(bars): this is not python` 【不是】语法错误 ——
#   `this is not python` 是合法的身份比较表达式，编译得过、调用时才 NameError。
#   这类"编译得过、一调用就炸"的代码单独测。
RUNTIME_BOOM = "def signal(bars):\n    return undefined_name(bars)"
WRONG_LEN = "def signal(bars):\n    return pd.Series([1.0, -1.0])"
PEEKING = '''
def signal(bars):
    nxt = bars["close"].shift(-1)
    return pd.Series(np.sign(nxt - bars["close"]).fillna(0).to_numpy(),
                     index=bars.index)
'''

r = S.audit_strategy(BAD_SYNTAX, ["ETHUSDT"])
ck("语法错误时给出清晰报错", "无法执行" in r, r[:60])
r = S.audit_strategy(NO_SIGNAL, ["ETHUSDT"])
ck("没有 signal 函数时报错", "没有找到" in r, r[:60])
r = S.audit_strategy(RUNTIME_BOOM, ["ETHUSDT"])
ck("编译得过但调用就炸 → 提前试调时抓到", "调用时出错" in r, r[:80])
ck("并提示可用的名字", "可用的名字" in r)
r = S.audit_strategy(WRONG_LEN, ["ETHUSDT"])
ck("返回长度不对 → 明确指出", "长度不对" in r, r[:80])

print("\n  跑一个真实策略（ETHUSDT，可能要 1~2 分钟）…")
r = S.audit_strategy(GOOD, ["ETHUSDT"], claimed="回测年化 +212%")
ck("正常策略产出完整报告", "策略体检报告" in r and "④" in r)
ck("报告里印了数据出处", "数据出处" in r)
ck("印了策略自称的成绩", "+212%" in r)
print("\n" + r[:1400])

print("\n  跑一个偷看未来的策略…")
r2 = S.audit_strategy(PEEKING, ["ETHUSDT"])
ck("① 号闸门抓到前瞻", "发现前瞻" in r2, r2[:200])

print()
print("=" * 74)
print("【2】自然语言入口")
print("=" * 74)

v = S.strategy_vocabulary()
ck("词汇表列出了指标与比较词", "RSI" in v and "上穿" in v and "跌破" in v)
ck("词汇表说明了超出范围怎么办", "audit_strategy" in v)

# 词汇表外 → 必须拒绝，而且要把词汇表一起给出来
r = S.audit_plain_language("MACD 金叉就满仓梭哈", ["ETHUSDT"])
ck("词汇表外的说法被拒绝", "没法可靠地翻译" in r and "MACD" in r, r[:70])
ck("拒绝时附上了词汇表", "词汇表" in r)
ck("拒绝时【没有】跑审计", "策略体检报告" not in r)

# dry_run：只回读解读和代码
r = S.audit_plain_language("RSI 超过 70 做空、低于 30 做多、持 12 小时",
                           ["ETHUSDT"], dry_run=True)
ck("dry_run 回读了解读", "我理解成这样" in r and "RSI(14) > 70" in r)
ck("dry_run 印出了生成的代码", "def signal(bars):" in r)
ck("dry_run 没有跑审计", "策略体检报告" not in r)
ck("省略主语的第二条被回读出来", "省略了主语" in r, )

# 有歧义时，未确认不许开跑 —— 这是这一层最重要的安全阀
r = S.audit_plain_language("EMA12 上穿 EMA48 做多", ["ETHUSDT"])
ck("有歧义且未确认时不跑审计", "策略体检报告" not in r and "confirmed=True" in r)
ck("并说明了歧义在哪", "穿越那一根" in r)

print("\n  自然语言 → 五道闸门，跑一次真的（ETHUSDT，可能要 1~2 分钟）…")
r = S.audit_plain_language("RSI 超过 70 做空、低于 30 做多、持 12 小时",
                           ["ETHUSDT"], claimed="回测年化 +180%",
                           confirmed=True)
ck("确认后跑出完整报告", "策略体检报告" in r and "④" in r, r[:60])
ck("报告里仍然带着解读回读", "我理解成这样" in r)
ck("印了策略自称的成绩", "+180%" in r)

print()
print("=" * 74)
print("【3】走真实 MCP 协议：它能不能被别的客户端装上")
print("=" * 74)


async def handshake():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "truthserum.server"],
        cwd=str(Path(__file__).resolve().parents[1]))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            tools = await sess.list_tools()
            return [t.name for t in tools.tools]


try:
    names = asyncio.run(asyncio.wait_for(handshake(), timeout=90))
    print(f"  服务器暴露的工具: {names}")
    for t in ("list_gates", "fetch_market_data", "audit_strategy",
              "save_mcp_reference", "strategy_vocabulary",
              "audit_plain_language"):
        ck(f"暴露了 {t}", t in names)
except Exception as e:
    ck("MCP 协议握手", False, f"{type(e).__name__}: {e}")

print()
print("=" * 74)
print(f"{'✅ 全部通过' if not FAIL else f'❌ {len(FAIL)} 项失败'}")
for f in FAIL:
    print("   ", f)
print("=" * 74)
sys.exit(1 if FAIL else 0)
