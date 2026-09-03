"""数据出处核验 —— 你的 K 线本身是不是真的

## 为什么这也是一道闸门

前四道闸门查的是「你怎么用数据」。这一道查「数据本身对不对」。

一个讲『不要自欺』的工具，如果自己盲信某个数据接口，那是最讽刺的漏洞。
回测再严谨，喂进去的 K 线错了，结论就是错的 —— 而这类错误几乎不会报错，
只会安静地把数字变好看。

真实案例（作者 2026-08 的生产系统）：
  · 模拟盘交易所有自己的一套 K 线，同一分钟的最低价比生产端低 0.52%，
    多出一根真实市场不存在的下影线。止损是按交易所侧价格触发的 ——
    于是仓位被【不存在的价格】打掉，而回测永远看不到这件事。
  · 另一次：模型吃的是【现货】K 线，下单却在【永续】合约上，
    两者存在 0.05% 的稳定基差，屏障因此被系统性平移。

## 判据

拿官方 Binance MCP Server 返回的 K 线作为可信参照，
逐根比对本地缓存的 OHLC。任何一根对不上就报警。

MCP 是官方通道、走 OAuth 授权、不落地 API key —— 它比任何第三方镜像
都更适合当"真相的参照系"。

## 自检

把参照样本人为改一个数（改动小到 0.01%），检测器必须抓到。
抓不到说明容差设太松，它的"一致"没有意义。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..audit import Audit, AuditResult, SelfCheckFailed, Verdict

#: 相对容差。交易所返回的是定点小数，同一根 K 线应当【完全相等】；
#: 留 1e-9 只是为了吸收 float 往返误差，不是为了容忍真实差异。
RTOL = 1e-9


def _ref_path(cache_dir: Path, symbol: str, interval: str) -> Path:
    return cache_dir / f"mcp_ref_{symbol.replace('/', '')}_{interval}.json"


def load_reference(cache_dir, symbol: str, interval: str) -> pd.DataFrame | None:
    """读入 MCP 参照样本（原始 kline 数组）"""
    p = _ref_path(Path(cache_dir), symbol, interval)
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("ts")[["open", "high", "low", "close", "volume"]].sort_index()


def _compare(cached: pd.DataFrame, ref: pd.DataFrame):
    """返回 (比对根数, 不一致根数, 最大相对偏差, 示例说明)"""
    idx = cached.index.intersection(ref.index)
    if len(idx) == 0:
        return 0, 0, 0.0, "缓存与参照没有重叠的时间段"
    a, b = cached.loc[idx], ref.loc[idx]
    worst, bad_rows, example = 0.0, set(), ""
    for col in ("open", "high", "low", "close"):
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(x - y) / np.where(y != 0, np.abs(y), np.nan)
        bad = ~np.isclose(x, y, rtol=RTOL, atol=0)
        if bad.any():
            bad_rows |= set(np.where(bad)[0].tolist())
            i = int(np.nanargmax(np.where(bad, rel, 0)))
            if rel[i] > worst:
                worst = float(rel[i])
                example = (f"{idx[i]} 的 {col}：缓存 {x[i]} vs MCP {y[i]}"
                           f"（差 {rel[i]*100:.4f}%）")
    return len(idx), len(bad_rows), worst, example


class ProvenanceAudit(Audit):
    name = "⓪ 数据出处核验（K 线本身是不是真的）"
    catches = "喂进回测的行情与交易所官方数据不一致：镜像源偏差、现货/合约混用、模拟盘假影线"

    def __init__(self, cache_dir="./.cache", interval="1h"):
        self.cache_dir = Path(cache_dir)
        self.interval = interval

    def _pairs(self, ctx):
        out = []
        for sym, bars in ctx.bars.items():
            ref = load_reference(self.cache_dir, sym, self.interval)
            if ref is not None and len(ref):
                out.append((sym, bars, ref))
        return out

    def _self_check(self, ctx) -> str:
        pairs = self._pairs(ctx)
        if not pairs:
            raise SelfCheckFailed(
                f"没有找到任何 MCP 参照样本（{self.cache_dir}/mcp_ref_*.json）—— "
                f"无参照就无从核验")
        sym, bars, ref = pairs[0]
        # 人为把参照的一个收盘价改掉 0.01% —— 小到肉眼难辨，检测器必须抓到
        tampered = ref.copy()
        i = len(tampered) // 2
        tampered.iloc[i, tampered.columns.get_loc("close")] *= 1.0001
        n, bad, worst, _ = _compare(bars, tampered)
        if bad == 0:
            raise SelfCheckFailed(
                "把参照样本的一个收盘价改动 0.01% 后，检测器仍报告一致 —— 容差太松")
        return (f"在人为篡改 0.01% 的参照上抓到了（{sym}，"
                f"检出偏差 {worst*100:.4f}%）—— 核验精度足够")

    def _run(self, ctx) -> AuditResult:
        pairs = self._pairs(ctx)
        if not pairs:
            return AuditResult(
                name=self.name, verdict=Verdict.SKIPPED,
                headline="没有 MCP 参照样本，跳过核验",
                detail=["先用 Binance MCP Server 抓一段 K 线存为参照："
                        f"{self.cache_dir}/mcp_ref_<SYMBOL>_{self.interval}.json"])
        det, total, bad_total, worst_all, ex = [], 0, 0, 0.0, ""
        for sym, bars, ref in pairs:
            n, bad, worst, example = _compare(bars, ref)
            total += n; bad_total += bad
            if worst > worst_all:
                worst_all, ex = worst, example
            det.append(f"{sym}：比对 {n} 根，不一致 {bad} 根"
                       + (f"，最大偏差 {worst*100:.4f}%" if bad else ""))
        if bad_total:
            return AuditResult(
                name=self.name, verdict=Verdict.FAILED,
                headline=f"缓存行情与币安官方数据不一致：{bad_total}/{total} 根对不上",
                detail=det + [f"示例：{ex}",
                              "回测再严谨，喂进去的 K 线错了，结论就是错的。"],
                numbers={"compared": total, "mismatched": bad_total,
                         "worst_rel": worst_all})
        return AuditResult(
            name=self.name, verdict=Verdict.CLEAN,
            headline=f"行情与币安官方 MCP 逐根一致（{total} 根全对）",
            detail=det + ["参照来自 Binance MCP Server（官方通道、OAuth 授权、"
                          "不落地 API key）"],
            numbers={"compared": total, "mismatched": 0})
