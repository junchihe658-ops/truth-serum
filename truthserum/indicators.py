"""策略代码能用的指标工具箱 —— 单一出处

原先这几个函数长在 `server.py` 里。抽出来是因为自然语言层和它的测试都要用，
而从 `server.py` 拿会连带 `import mcp`（整个 MCP 框架），测试不该为了算一个
RSI 去拉那么一大坨。

## 这里的每一个函数都必须是【因果】的

也就是第 i 个输出只能用到第 i 个及之前的输入。理由很直接：这些是喂给
策略代码的原料，原料本身要是偷看了未来，① 号闸门抓到的会是我们自己的 bug，
而用户会以为是他策略写错了。

具体就是：**禁止 center=True 的滚动窗口、禁止 shift(负数)、禁止 bfill**。
`.shift(1)` 是允许的（往回看一根）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    """相对强弱指标。第一根是 NaN（没有前一根就算不出涨跌）。"""
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def ema(c: pd.Series, n: int) -> pd.Series:
    return c.ewm(span=n, adjust=False).mean()


def sma(c: pd.Series, n: int) -> pd.Series:
    """简单均线。前 n-1 根是 NaN —— 不够长就是算不出，不许拿 bfill 补。"""
    return c.rolling(n).mean()


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - pc).abs(),
                    (bars["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bb_pct(c: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """价格在布林带里的相对位置：0 = 下轨，1 = 上轨"""
    m, s = c.rolling(n).mean(), c.rolling(n).std()
    return (c - (m - k * s)) / ((m + k * s) - (m - k * s) + 1e-12)


#: 执行策略代码时预置的名字。自然语言层生成的代码也只许用这里面的东西。
TOOLBOX = {
    "pd": pd, "np": np, "pandas": pd, "numpy": np,
    "rsi": rsi, "ema": ema, "sma": sma, "atr": atr, "bb_pct": bb_pct,
}

#: 给人看的一行说明，工具描述和报告里都要用到，别写两遍
TOOLBOX_DOC = "pd, np, rsi(c,n), ema(c,n), sma(c,n), atr(bars,n), bb_pct(c,n,k)"
