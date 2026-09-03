"""Truth Serum MCP Server —— 让任何 AI 客户端都能审计自己的策略

## 装法

    claude mcp add truth-serum -- python -m truthserum.server

装好之后，直接对你的 Claude 说：

    「用 truth-serum 审计这个策略：RSI 高于 70 做空、低于 30 做多」

它会生成策略代码、跑完五道闸门、告诉你这个回测数字有多少是真的。

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
from .runner import AUDITS, check

mcp = FastMCP("truth-serum")

# ── 给策略代码用的工具箱 ──────────────────────────────────────
def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def ema(c: pd.Series, n: int) -> pd.Series:
    return c.ewm(span=n, adjust=False).mean()


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - pc).abs(),
                    (bars["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bb_pct(c: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    m, s = c.rolling(n).mean(), c.rolling(n).std()
    return (c - (m - k * s)) / ((m + k * s) - (m - k * s) + 1e-12)


_SANDBOX = {"pd": pd, "np": np, "pandas": pd, "numpy": np,
            "rsi": rsi, "ema": ema, "atr": atr, "bb_pct": bb_pct}

GATES_DOC = """Truth Serum 的五道闸门（从「最底层的前提」往上查）

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

每一道闸门在给结论之前，都必须先在【人为植入的 bug】上证明自己抓得到。
自检不过就返回「不可判定」，而不是「一切正常」——
一个永远报平安的检测器，比没有检测器更危险。"""


@mcp.tool()
def list_gates() -> str:
    """列出 Truth Serum 的五道审计闸门，说明每一道抓什么、为什么需要它。

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
        预置可用：pd, np, rsi(c,n), ema(c,n), atr(bars,n), bb_pct(c,n,k)

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
                f"可用的名字：pd, np, rsi(c,n), ema(c,n), atr(bars,n), bb_pct(c,n,k)")

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):          # 别让策略里的 print 污染 MCP 协议
            rep = check(bars, FuncStrategy("待审策略", fn),
                        name="待审策略", claimed=claimed,
                        costs=Costs(fee_per_side=fee_per_side),
                        barrier_mult=barrier_mult, horizon=horizon)
    except Exception:
        return f"审计过程出错：\n{traceback.format_exc(limit=5)}"

    out = [describe(prov), "", rep.render()]
    noise = buf.getvalue().strip()
    if noise:
        out += ["", "（策略代码的输出）", noise[:2000]]
    return "\n".join(out)


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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
