"""Truth Serum MCP Server —— 让任何 AI 客户端都能审计自己的策略

## 装法

    claude mcp add truth-serum -- python -m truthserum.server

装好之后，直接对你的 Claude 说：

    「用 truth-serum 审计这个策略：RSI 高于 70 做空、低于 30 做多」

它会生成策略代码、跑完全部闸门、告诉你这个回测数字有多少是真的。

## 信任模型（重要，先说清楚）

`audit_strategy` 会**在本机执行你传入的 Python 代码**。

这和「你自己写个脚本跑一下」是同一个信任级别 —— MCP server 跑在你自己
机器上、代码来自你自己的会话。但仍然请注意：**不要把来路不明的策略代码
丢进来跑**，就像你不会随便执行一个陌生人发来的脚本一样。

执行环境里预置了 pandas / numpy 和几个指标函数，没有额外做沙箱。
"""
from __future__ import annotations

import io
import json
import sys
import traceback
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from mcp.server.fastmcp import FastMCP

from .core import Costs, FuncStrategy
from .data import describe, load
from .indicators import TOOLBOX, TOOLBOX_DOC
from .nl import VOCAB_DOC, CannotParse
from .nl import parse as parse_nl
from .runner import AUDITS, check
from .session import SESSION

mcp = FastMCP("truth-serum")

#: 策略代码的工具箱住在 indicators.py —— 自然语言层和它的测试也要用，
#: 从这里拿会连带 import mcp（整个 MCP 框架）。单一出处，别再复制一份。
_SANDBOX = dict(TOOLBOX)

GATES_DOC = """Truth Serum 的闸门（从「最底层的前提」往上查）

⓪ 数据出处核验   —— 你的 K 线本身是不是真的
   拿官方 Binance MCP Server 的 K 线作参照，逐根比对本地行情。
   抓：镜像源偏差、现货/合约混用、模拟盘假影线。
   （需要先用 Binance MCP 抓一段参照样本；没有就跳过这道）

① 前瞻检测       —— 策略有没有偷看未来
   删掉切点之后的 K 线、重算信号，切点之前每一根必须一位不变。
   抓：时区错位、居中滚动窗口、bfill/nearest 填充、跨周期错位对齐。

② 重叠计数       —— 「平均每笔期望」是不是被信号持续期灌水
   同一段趋势里 20 根 K 线都满足入场条件，不是 20 笔独立观测。
   抓：把持续性当成重复证据，系统性高估每笔收益。

③ 零假设本底     —— 随机信号能不能也做出同样的成绩
   把信号块打乱若干次，看真实成绩落在本底分布的哪个位置。
   抓：「筛了很多个，最好的那个看起来不错」这种选择偏差。

④ 组合层模拟     —— 账户里最后到底剩多少钱
   共用资金池、仓位上限、按权益定仓、复利与回撤。
   抓：用单笔期望冒充账户收益。这是唯一有资格下结论的一道。

⑤ 搜索选择偏差 —— 这个成绩是不是「搜出来的」
   拿「随机搜同样多次能得到的最好成绩」当本底。
   抓：「我试了两百组，这组最好」—— 偏差藏在没交出来的那 199 组里。
   ③ 号只查单个策略，盖不住这一层。
   （需要策略附带搜索日志；没有就跳过这道）

⚠ 这是【证伪器，不是认证器】。
   挂了闸门是强信号 —— 抓到了具体的、可复现的失败机制。
   全部通过是弱信号 —— 只说明这几种已知死法没被检出，不等于能赚钱。

每一道闸门在给结论之前，都必须先在【人为植入的 bug】上证明自己抓得到。
自检不过就返回「不可判定」，而不是「一切正常」——
一个永远报平安的检测器，比没有检测器更危险。"""


@mcp.tool()
def list_gates() -> str:
    """列出 Truth Serum 的审计闸门，说明每一道抓什么、为什么需要它。

    在向用户解释这个工具做什么、或者决定要不要审计某个策略之前，先调用它。
    """
    return GATES_DOC


@mcp.tool()
def fetch_market_data(symbols: list[str], interval: str = "1h",
                      bars: int = 9000) -> str:
    """把币安行情拉到本地缓存，供后续审计使用。

    审计一律跑在缓存上，不在跑分析时实时请求交易所 —— 交易所偶发的限流和
    5xx 不该让一次审计半途而废。

    symbols  例如 ["BTCUSDT", "ETHUSDT"]
    interval 1m/5m/15m/1h/4h/1d
    bars     大致要多少根（会自动翻页）
    """
    try:
        _, prov = load(symbols, interval, pages=max(1, bars // 1500))
        return describe(prov)
    except Exception as e:
        return f"拉取失败：{type(e).__name__}: {e}"


def _load_probe_check(fn, symbols, interval, claimed,
                      barrier_mult, horizon, fee_per_side,
                      name: str = "待审策略", via: str = "代码") -> str:
    """加载行情 → 试调 → 跑五道闸门。两条审计入口共用这一份。

    抽出来是因为两条入口（手写代码 / 自然语言）必须走【完全相同】的流程。
    要是哪天只改了一边，两条路会对同一个策略给出不同结论 ——
    那是最难查的那种 bug，而且恰好是这个项目最不该犯的错。
    """
    try:
        bars, prov = load(symbols, interval)
    except Exception as e:
        return (f"拿不到行情：{type(e).__name__}: {e}\n"
                f"先调用 fetch_market_data 把数据拉到缓存。")

    # ⚠ 先用几十根 K 线试调一次再进审计。
    #   策略代码可能【编译得过但一调用就炸】（比如笔误成了未定义的名字），
    #   不先试调的话，这类错误会深埋在某道闸门里冒出来，报错难读。
    probe_sym = next(iter(bars))
    try:
        out = fn(bars[probe_sym].head(200))
        n = len(out) if hasattr(out, "__len__") else -1
        if n != 200:
            return (f"策略返回的长度不对：喂了 200 根 K 线，返回 {n} 个值。\n"
                    f"`signal(bars)` 必须返回与 bars 【等长】的序列。")
    except Exception:
        return (f"策略代码能编译，但调用时出错：\n"
                f"{traceback.format_exc(limit=3)}\n"
                f"可用的名字：{TOOLBOX_DOC}")

    # ── 把这次尝试记进会话日志 ───────────────────────────────
    #   真正的 agent 不是我们写的那个 for 循环，是【正在用这个工具的模型】。
    #   它读懂闸门反馈、自己想下一个策略、再试一次 —— 那就是「我试了很多个，
    #   这个最好」，正是 ⑤ 号闸门要抓的东西。
    #   所以每一次审计请求都要记下来，让 ⑤ 号能对着累计次数说话。
    key = SESSION.key_of(symbols, interval, barrier_mult, horizon, fee_per_side)
    try:
        score, sigs = SESSION.score_of(bars, fn, barrier_mult, horizon,
                                       Costs(fee_per_side=fee_per_side).round_trip)
    except Exception:
        score, sigs = float("nan"), None
    SESSION.record(key, name, score, via, sig=sigs)
    search_log = SESSION.as_search_log(key)

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):          # 别让策略里的 print 污染 MCP 协议
            rep = check(bars, FuncStrategy(name, fn),
                        name=name, claimed=claimed,
                        costs=Costs(fee_per_side=fee_per_side),
                        barrier_mult=barrier_mult, horizon=horizon,
                        search_log=search_log)
    except Exception:
        return f"审计过程出错：\n{traceback.format_exc(limit=5)}"

    out = [describe(prov), "", SESSION.render(key), "", rep.render()]
    noise = buf.getvalue().strip()
    if noise:
        out += ["", "（策略代码的输出）", noise[:2000]]
    return "\n".join(out)


@mcp.tool()
def audit_strategy(signal_code: str, symbols: list[str],
                   interval: str = "1h", claimed: str = "",
                   barrier_mult: float = 1.5, horizon: int = 12,
                   fee_per_side: float = 0.0005) -> str:
    """给一个交易策略做体检，返回「这个回测数字有多少是真的」。

    ⚠ 会在本机执行 signal_code。信任级别等同于你自己跑一个脚本 ——
      不要把来路不明的代码丢进来。

    signal_code
        一段 Python，必须定义 `def signal(bars):`，返回与 bars 等长的
        +1(做多) / −1(做空) / 0(观望) 序列。
        bars 是 DataFrame，index 为时间，列有 open/high/low/close/volume。
        预置可用：pd, np, rsi(c,n), ema(c,n), sma(c,n), atr(bars,n), bb_pct(c,n,k)

        ⚠ 合约：第 i 行的信号只能用第 i 行【及之前】的数据。
          违反了也没关系 —— ① 号闸门会抓到。

        例：
            def signal(bars):
                r = rsi(bars["close"])
                return pd.Series(np.where(r > 70, -1.0,
                                 np.where(r < 30, 1.0, 0.0)), index=bars.index)

    symbols       要审计的标的，例如 ["BTCUSDT", "ETHUSDT"]
    claimed       这个策略自称的成绩（如 "年化 +212%"），会印在报告顶部作对照
    barrier_mult  止盈止损各 mult × ATR
    horizon       时间屏障：多少根 K 线后强制平仓
    fee_per_side  单边手续费，默认 0.05%（币安永续 taker）
    """
    ns = dict(_SANDBOX)
    try:
        exec(signal_code, ns)
    except Exception:
        return f"策略代码无法执行：\n{traceback.format_exc(limit=3)}"
    fn = ns.get("signal")
    if not callable(fn):
        return "策略代码里没有找到 `def signal(bars):`"

    return _load_probe_check(fn, symbols, interval, claimed,
                             barrier_mult, horizon, fee_per_side, via="代码")


@mcp.tool()
def save_mcp_reference(symbol: str, klines_json: str,
                       interval: str = "1h") -> str:
    """把你通过【官方 Binance MCP Server】取到的 K 线存为可信参照，
    供 ⓪ 号闸门核验本地行情。

    典型用法（两个 MCP 配合干活）：
      1. 调 binance-mcp-server 的 klines 工具取一段 K 线
      2. 把返回的原始 JSON 数组原样传给这里
      3. 之后每次 audit_strategy 都会自动拿它核验缓存行情

    klines_json  币安 kline 接口返回的原始数组，形如
                 [[openTime,"open","high","low","close","volume",...], ...]
    """
    from pathlib import Path
    from .data import CACHE_DIR
    try:
        rows = json.loads(klines_json) if isinstance(klines_json, str) else klines_json
        if not rows or not isinstance(rows[0], list):
            return "格式不对：应当是 kline 数组的数组"
        # 最后一根很可能还没收盘，丢掉 —— 拿进行中的 bar 当参照会误报
        rows = rows[:-1] if len(rows) > 1 else rows
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = Path(CACHE_DIR) / f"mcp_ref_{symbol.replace('/', '')}_{interval}.json"
        p.write_text(json.dumps(rows), encoding="utf-8")
        first = pd.to_datetime(rows[0][0], unit="ms")
        last = pd.to_datetime(rows[-1][0], unit="ms")
        return (f"已存 {len(rows)} 根参照 → {p.name}\n"
                f"区间 {first} ~ {last}（已丢弃最后一根未收盘的）\n"
                f"之后 audit_strategy 会用它核验本地行情。")
    except Exception as e:
        return f"保存失败：{type(e).__name__}: {e}"


@mcp.tool()
def strategy_vocabulary() -> str:
    """列出「用大白话描述策略」时能用的全部说法。

    在调用 audit_plain_language 之前先看一眼。这一层是【确定性解析】，
    只认词汇表里的说法 —— 认不出会明确告诉你哪几个字没看懂，而不是
    猜一段代码给你。
    """
    return VOCAB_DOC


@mcp.tool()
def audit_plain_language(description: str, symbols: list[str],
                         interval: str = "1h", claimed: str = "",
                         confirmed: bool = False, dry_run: bool = False,
                         fee_per_side: float = 0.0005) -> str:
    """用一句大白话描述策略，自动翻译成代码并跑完全部闸门。

    例："RSI 超过 70 做空、低于 30 做多、持 12 小时"

    和 audit_strategy 的区别：**这里不执行任何外部代码**。策略代码由
    确定性解析器生成，只可能用到词汇表里的算子；词汇表外的说法会被
    明确拒绝，不会猜。完整词汇表见 strategy_vocabulary。

    ⚠ 返回的第一段永远是【解读回读】—— 把你的话按解析结果读回去。
      **请把这段原样转述给用户核对。** 解析器可能理解错，而理解错
      又没人发现的话，用户会拿着一份体检报告，以为审的是他想的那个策略。
      那正是这个工具存在的意义所要消灭的东西。

    description  策略的大白话描述
    symbols      要审计的标的，例如 ["BTCUSDT", "ETHUSDT"]
    interval     K 线周期。它决定「持 12 小时」换算成多少根 —— 传错了，
                 审的就是另一个策略
    claimed      这个策略自称的成绩，印在报告顶部作对照
    confirmed    解析中若存在歧义（例如「跌破」有两种理解），默认只回读、
                 不开跑；用户确认解读无误后带 confirmed=True 再调一次
    dry_run      只看解读和生成的代码，不跑审计
    """
    try:
        spec = parse_nl(description, interval)
    except CannotParse as e:
        return ("这句话我没法可靠地翻译成策略，所以不翻译 ——\n"
                "猜一个给你，比直接说不会更糟。\n\n"
                f"{e}\n\n"
                "两条路：\n"
                "  1. 换成词汇表里的说法（见下）\n"
                "  2. 直接写 `def signal(bars):` 代码，走 audit_strategy\n\n"
                f"{VOCAB_DOC}")

    code = spec.to_code()
    echo = [spec.explain(), "", "生成的代码：", "```python", code.rstrip(), "```"]

    if dry_run:
        return "\n".join(echo + ["", "（dry_run：没有跑审计）"])

    if spec.warnings and not confirmed:
        return "\n".join(echo + [
            "", "─" * 58,
            f"上面有 {len(spec.warnings)} 处是我替你做的选择，先请用户核对。",
            "确认无误后带 confirmed=True 再调一次，才会真的跑审计。",
            "这一步不能省 —— 解读错了而没人发现，报告审的就是另一个策略。"])

    out = _load_probe_check(spec.to_strategy(), symbols, interval, claimed,
                            spec.barrier_mult if spec.barrier_mult else 1.5,
                            spec.horizon if spec.horizon else 12,
                            fee_per_side, name=description.strip()[:40],
                            via="自然语言")
    return "\n".join(echo + ["", "=" * 58, "", out])


@mcp.tool()
def search_history(symbols: list[str], interval: str = "1h",
                   barrier_mult: float = 1.5, horizon: int = 12,
                   fee_per_side: float = 0.0005) -> str:
    """看看这个会话在某个配置下已经试过多少个策略、哪个最好。

    ⚠ 为什么这个工具存在：

    如果你（作为 agent）在反复尝试 —— 提一个策略、看哪道闸门没过、
    改一个再提 —— 那你正在做的事情叫【搜索】。搜索的结果里，有多少是
    策略真的好、有多少只是「试得够多」，这两件事你自己分不出来。

    ⑤ 号闸门就是拿这个次数在算。所以在你打算把某个成绩报出去之前，
    先看一眼你到底试了多少次。

    参数要和你审计时用的完全一致 —— 不同配置的成绩不可比，分开计数。
    """
    key = SESSION.key_of(symbols, interval, barrier_mult, horizon, fee_per_side)
    return SESSION.render(key)


@mcp.tool()
def reset_search_history() -> str:
    """清空本会话的搜索记录。

    ⚠ 什么时候该用：换了一个完全不同的课题，之前的尝试不该再计入。

    ⚠ 什么时候【不该】用：你试了很多次、成绩不好看，想让 ⑤ 号闸门
      别再提这件事。清空之后选择偏差就从视野里消失了 —— 而它并没有消失，
      只是你看不见了。这个工具存在的意义就是不让这种事发生。

    这条警告写在这里，是因为「能清空」本身就是一个可以用来自欺的接口。
    """
    n = SESSION.reset()
    return (f"已清空 {n} 条搜索记录。\n\n"
            f"提醒一句：清空不会让已经发生的搜索消失，只会让 ⑤ 号闸门看不见它。"
            f"如果你是因为「数字不好看」才清的，那正是这个工具想拦住的事。")


def main():
    mcp.run()


if __name__ == "__main__":
    main()
