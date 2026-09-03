"""行情数据入口 + 本地缓存

## 两条入口，同一个出口

  · `BinanceMCP`   —— 通过 Binance MCP Server 拉（需要授权，赛题要求的路径）
  · `BinancePublic`—— 公开 REST，无需认证（开发期用，也是没授权时的兜底）

两者都吐出同一个格式：DatetimeIndex + open/high/low/close/volume。

## 为什么一定要缓存

演示要稳。2026-09-02 那天 OKX 连着返回 50001「服务暂不可用」——
录视频时碰上一次，整个 demo 就翻车。所以：**MCP/REST 只负责把数据取回来，
审计一律跑在本地缓存上。**

缓存是 parquet，带元数据（来源、抓取时间、K线区间），
报告里会印出来 —— 让人知道这份结论是基于哪一份数据。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(os.environ.get("TRUTHSERUM_CACHE", "./.cache"))
BINANCE_SPOT = "https://api.binance.com/api/v3/klines"
BINANCE_FUT = "https://fapi.binance.com/fapi/v1/klines"

_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


@dataclass
class Provenance:
    """这份数据是从哪来的 —— 报告里要印出来，结论才可追溯"""
    source: str
    symbol: str
    interval: str
    rows: int
    first: str
    last: str
    fetched_at: str


def _cache_path(source: str, symbol: str, interval: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{source}_{symbol.replace('/', '')}_{interval}.parquet"


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_COLS[:len(rows[0])])
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df.set_index("ts")[["open", "high", "low", "close", "volume"]]
    return out[~out.index.duplicated(keep="first")].sort_index()


class BinancePublic:
    """公开 K 线接口，无需认证。开发期与兜底用。"""

    name = "binance-public"

    def __init__(self, market: str = "futures", proxy: str | None = None,
                 timeout: int = 30):
        self.url = BINANCE_FUT if market == "futures" else BINANCE_SPOT
        p = proxy or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        self.proxies = {"http": p, "https": p} if p else None
        self.timeout = timeout

    def fetch(self, symbol: str, interval: str = "1h",
              limit: int = 1500, pages: int = 6) -> pd.DataFrame:
        sym = symbol.replace("/", "").upper()
        out, end = [], None
        for _ in range(pages):
            params = {"symbol": sym, "interval": interval, "limit": limit}
            if end is not None:
                params["endTime"] = end
            r = self._get(params)
            if not r:
                break
            out = r + out
            end = int(r[0][0]) - 1
            if len(r) < limit:
                break
        if not out:
            raise RuntimeError(f"{symbol} 没拉到任何 K 线")
        return _to_frame(out)

    def _get(self, params: dict, retries: int = 5):
        for i in range(retries):
            try:
                resp = requests.get(self.url, params=params,
                                    proxies=self.proxies, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if i == retries - 1:
                    raise
                # 交易所偶发 5xx/限流是常态，不该让整条管线挂掉
                time.sleep(2 * (i + 1))
        return []


class BinanceMCP:
    """通过 Binance MCP Server 取行情。

    ⚠ 授权必须由用户在自己的 AI 客户端里完成：
        claude mcp add binance-mcp-server --transport http https://agent.binance.com/mcp/agentic
      然后在 /mcp 菜单里选中它，走币安的同意页。
      官方文档明确要求【不要】把 endpoint 粘进 AI 对话让它代装。

    本类只负责把 MCP 工具返回的 K 线整成统一格式；调用方把 MCP 工具
    以 `call(tool_name, **kwargs) -> list` 的形式注入进来。
    """

    name = "binance-mcp"

    def __init__(self, call):
        self.call = call

    def fetch(self, symbol: str, interval: str = "1h", limit: int = 1500,
              pages: int = 6) -> pd.DataFrame:
        rows = self.call("klines", symbol=symbol.replace("/", "").upper(),
                         interval=interval, limit=limit)
        return _to_frame(rows)


def load(symbols: list[str], interval: str = "1h", source=None,
         refresh: bool = False, **fetch_kw) -> tuple[dict[str, pd.DataFrame],
                                                     list[Provenance]]:
    """取数据：优先读缓存，缓存没有才去拉。返回 (bars, 出处)"""
    src = source or BinancePublic()
    bars, prov = {}, []
    for s in symbols:
        p = _cache_path(src.name, s, interval)
        meta = p.with_suffix(".json")
        if p.exists() and not refresh:
            df = pd.read_parquet(p)
            pv = Provenance(**json.loads(meta.read_text(encoding="utf-8"))) \
                if meta.exists() else Provenance(src.name, s, interval, len(df),
                                                 str(df.index[0]), str(df.index[-1]),
                                                 "（缓存，无元数据）")
        else:
            df = src.fetch(s, interval, **fetch_kw)
            df.to_parquet(p)
            pv = Provenance(src.name, s, interval, len(df), str(df.index[0]),
                            str(df.index[-1]),
                            time.strftime("%Y-%m-%d %H:%M:%S"))
            meta.write_text(json.dumps(asdict(pv), ensure_ascii=False),
                            encoding="utf-8")
        bars[s] = df
        prov.append(pv)
    return bars, prov


def describe(prov: list[Provenance]) -> str:
    lines = ["数据出处："]
    for p in prov:
        lines.append(f"  {p.symbol:<10} {p.interval}  {p.rows} 根  "
                     f"{p.first[:16]} ~ {p.last[:16]}  "
                     f"[{p.source}, 抓取于 {p.fetched_at}]")
    return "\n".join(lines)
