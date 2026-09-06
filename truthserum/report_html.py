"""把体检报告渲染成单文件 HTML

## 设计取舍

  · **单文件、零外部依赖** —— 不引 CDN、不引字体。评审可能在没网的环境下打开，
    而且一个讲『不要盲信』的工具，自己去外部拉资源是说不过去的。
  · **自检证据放在显眼处**，不折叠。那是这个工具区别于其他回测器的地方 ——
    「我凭什么相信你说的『干净』」这个问题，必须在第一屏就被回答。
  · **策略自称的数字和结论并排放**。落差本身就是叙事。
  · **异常的闸门默认展开，正常的折叠**。评审只有两三分钟，先看该看的。

## 关于这一版的样式

中间试过一版浅色「排版文档」风格（奶油纸 + 深松绿报头 + 左窄栏网格）。
作者看过两版之后选了这一版深色的，所以换了回来。
浅色那版的三处内容改进被保留下来了，因为它们跟观感无关、只跟正确性有关：

  1. `_fmt()` 把原始读数排成人能看的样子 —— 否则页面上会出现 16 位小数
  2. 每道闸门印出 `catches`（这道闸门针对什么），让「未见异常」有具体所指
  3. 数据出处按行渲染成表，而不是糊成一坨

⚠ 数据出处必须**保留换行**传进来。压成一行的话，这里按行数去数标的，
  4 个标的会被读成 1 个 —— 报头印「1 个标的」而送检声明印的是另一个币的
  成绩。这个 bug 真的发生过。
"""
from __future__ import annotations

import html
import re
from datetime import datetime

from .audit import TruthReport, Verdict

_STYLE = """
:root{
  --bg:#0b0e14; --card:#141922; --line:#232a36; --ink:#e6e9ef; --dim:#8b94a7;
  --ok:#3fb950; --bad:#f85149; --warn:#d29922; --skip:#6e7681; --accent:#58a6ff;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
  "Hiragino Sans GB","Microsoft YaHei",sans-serif;}
.wrap{max-width:920px;margin:0 auto;padding:40px 24px 80px}
.brand{font-size:13px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--dim);margin-bottom:6px}
h1{font-size:26px;margin:0 0 4px;font-weight:650}
.sub{color:var(--dim);font-size:13px;margin-bottom:28px}

.claim{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 24px}
.claim>div{flex:1;min-width:240px;background:var(--card);
  border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.claim .k{font-size:12px;color:var(--dim);margin-bottom:6px}
.claim .v{font-size:19px;font-weight:600}
.claim .v.big{font-size:22px}

.verdict{border-radius:12px;padding:18px 22px;margin:0 0 30px;
  border:1px solid;font-size:17px;font-weight:600;line-height:1.5}
.verdict.pass{background:rgba(63,185,80,.10);border-color:rgba(63,185,80,.45);
  color:var(--ok)}
.verdict.fail{background:rgba(248,81,73,.10);border-color:rgba(248,81,73,.45);
  color:var(--bad)}
.verdict.unknown{background:rgba(210,153,34,.10);
  border-color:rgba(210,153,34,.45);color:var(--warn)}

.strip{display:flex;gap:8px;margin:0 0 28px;flex-wrap:wrap}
.chip{flex:1;min-width:150px;background:var(--card);border:1px solid var(--line);
  border-radius:10px;padding:12px 14px;text-align:center}
.chip .icon{font-size:22px;line-height:1}
.chip .nm{font-size:12px;color:var(--dim);margin-top:6px}

.gate{background:var(--card);border:1px solid var(--line);border-radius:12px;
  margin:0 0 14px;overflow:hidden}
.gate>summary{list-style:none;cursor:pointer;padding:18px 20px;
  display:flex;gap:14px;align-items:flex-start}
.gate>summary::-webkit-details-marker{display:none}
.gate .ic{font-size:20px;line-height:1.3}
.gate .body{flex:1;min-width:0}
.gate .ttl{font-weight:600;margin-bottom:3px}
.gate .head{color:var(--dim);font-size:14px}
.gate[open] .head{color:var(--ink)}
.gate .caret{color:var(--skip);font-size:12px;padding-top:4px}
.gate[open] .caret{transform:rotate(90deg)}

.inner{padding:0 20px 20px 54px}
.selfcheck{background:rgba(88,166,255,.07);border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0;padding:11px 14px;margin:0 0 14px;font-size:13.5px}
.selfcheck b{color:var(--accent);font-weight:600}
ul.det{margin:0;padding-left:18px;color:var(--dim);font-size:14px}
ul.det li{margin:5px 0}
.nums{margin-top:12px;font-size:12.5px;color:var(--skip);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  display:flex;flex-wrap:wrap;gap:4px 18px}
.nums b{font-weight:400;color:var(--dim)}
.catches{font-size:12.5px;color:var(--skip);margin-top:10px;font-style:italic}

/* 结构性免责 —— 这个工具是证伪器不是认证器，
   报告不能把「通过」印得和「未通过」一样重。 */
.disclaimer{margin:34px 0 0;padding:16px 20px;border-radius:10px;
  background:rgba(210,153,34,.08);border:1px solid rgba(210,153,34,.35);
  color:var(--ink);font-size:13.5px;line-height:1.75}
.disclaimer b{color:var(--warn)}

footer{margin-top:28px;padding-top:20px;border-top:1px solid var(--line);
  color:var(--skip);font-size:12.5px;line-height:1.8}
footer code{background:var(--card);padding:2px 6px;border-radius:4px}
footer b{color:var(--dim)}
.srcwrap{overflow-x:auto;margin:0 0 16px}
footer table{border-collapse:collapse}
footer table td{padding:1px 16px 1px 0;white-space:nowrap;vertical-align:top;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px}
@media(max-width:640px){
  .inner{padding-left:20px}
  footer table td{white-space:normal}
}
"""

_ICON = {Verdict.CLEAN: "✅", Verdict.FAILED: "❌",
         Verdict.UNUSABLE: "⛔", Verdict.SKIPPED: "⏭"}


def _esc(s) -> str:
    return html.escape(str(s))


def _fmt(v) -> str:
    """把原始读数排成人能看的样子 —— 不往页面上倒 16 位小数"""
    try:
        if isinstance(v, bool):
            return "是" if v else "否"
        if isinstance(v, int):
            return f"{v:,}"
        if isinstance(v, float):
            a = abs(v)
            if a >= 100:
                return f"{v:,.1f}"
            if a >= 1:
                return f"{v:,.2f}"
            return f"{v:.4f}"
    except Exception:            # 将来塞进别的类型时不许整页炸掉
        pass
    return str(v)


def _src_rows(note: str) -> list[list[str]]:
    """describe() 的每行按 2+ 空格切成列。

    只做切分、不做语义解析 —— 上游改了格式，最坏情况是列数变了，
    页脚照样能原样印出来，不会印出编造的东西。
    """
    rows = []
    for ln in str(note).splitlines():
        ln = ln.strip()
        if not ln or ln.endswith(("：", ":")):
            continue
        rows.append(re.split(r"\s{2,}", ln))
    return rows


def render_html(rep: TruthReport, *, data_note: str = "") -> str:
    if rep.any_unusable:
        cls, headline = "unknown", rep.verdict_line()
    elif rep.failures:
        cls, headline = "fail", rep.verdict_line()
    else:
        cls, headline = "pass", rep.verdict_line()

    strip = "".join(
        f'<div class="chip"><div class="icon">{_ICON[r.verdict]}</div>'
        f'<div class="nm">{_esc(r.name.split("（")[0].strip())}</div></div>'
        for r in rep.results)

    gates = []
    for r in rep.results:
        det = "".join(f"<li>{_esc(d)}</li>" for d in r.detail)
        sc = (f'<div class="selfcheck"><b>自检</b> — 这道闸门先在人为植入的 bug 上'
              f'证明了自己抓得到：<br>{_esc(r.self_check_evidence)}</div>'
              if r.self_check_evidence else "")
        nums = ""
        if r.numbers:
            items = "".join(f"<span><b>{_esc(k)}</b> {_esc(_fmt(v))}</span>"
                            for k, v in r.numbers.items())
            nums = f'<div class="nums">{items}</div>'
        catches = (f'<div class="catches">这道闸门抓：{_esc(r.catches)}</div>'
                   if r.catches else "")
        gates.append(f"""
<details class="gate"{" open" if r.verdict is not Verdict.CLEAN else ""}>
  <summary>
    <span class="ic">{_ICON[r.verdict]}</span>
    <span class="body"><div class="ttl">{_esc(r.name)}</div>
      <div class="head">{_esc(r.headline)}</div></span>
    <span class="caret">▶</span>
  </summary>
  <div class="inner">{sc}<ul class="det">{det}</ul>{nums}{catches}</div>
</details>""")

    # 上面这个框用【短判读】，完整那句留给下面的横幅 ——
    # 原版两处印同一个长句，叠在一起看着像重复贴了两遍。
    n_fail = len(rep.failures)
    if rep.any_unusable:
        short = "⛔ 不可判定"
    elif n_fail:
        short = f"❌ {n_fail} 项审计未通过"
    else:
        short = "✅ 全部通过"

    claim = ""
    if rep.claimed:
        claim = f"""
<div class="claim">
  <div><div class="k">策略自称</div><div class="v big">{_esc(rep.claimed)}</div></div>
  <div><div class="k">体检结论</div><div class="v">{_esc(short)}</div></div>
</div>"""

    src = ""
    rows = _src_rows(data_note)
    if rows:
        body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row)
                       + "</tr>" for row in rows)
        src = (f'<b>受检数据逐项出处</b><div class="srcwrap">'
               f'<table>{body}</table></div>')

    # 闸门数不能写死 —— 从五道加到六道那次，底部那句「五道全过」没跟着改，
    # 于是页面上同时印着六个图标和「五道全过」，自相矛盾。改成从结果里数。
    n_gates = len(rep.results)

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Truth Serum · {_esc(rep.strategy_name)}</title>
<style>{_STYLE}</style></head><body><div class="wrap">
  <div class="brand">Truth Serum</div>
  <h1>{_esc(rep.strategy_name)}</h1>
  <div class="sub">策略体检报告 · 生成于 {datetime.now():%Y-%m-%d %H:%M}</div>
  {claim}
  <div class="verdict {cls}">{_esc(headline)}</div>
  <div class="strip">{strip}</div>
  {''.join(gates)}
  <div class="disclaimer">
    <b>这是证伪器，不是认证器。</b><br>
    「挂了闸门」是<b>强</b>信号 —— 它抓到了具体的、可复现的失败机制。<br>
    「{n_gates} 道全过」是<b>弱</b>信号 —— 那只意味着<b>这 {n_gates} 种已知死法没被检出</b>，
    不等于这个策略能赚钱。<br>
    没被检出的死法还有很多：资金费、连续性缺口、跨所成本差异、
    以及任何我们还没想到的自欺方式。
  </div>
  <footer>
    {src}
    每一道闸门在给结论之前，都必须先在<b>人为植入的 bug</b> 上证明自己抓得到；
    自检不过就返回「不可判定」，而不是「一切正常」。<br>
    一个永远报平安的检测器，比没有检测器更危险。<br><br>
    <b>自检是必要不充分的</b>：它只证明了这道闸门抓得住<b>那一种</b>人为 bug，
    不代表抓得住同类的其它变体。<br><br>
    装成 MCP：<code>claude mcp add truth-serum -- python -m truthserum.server</code>
  </footer>
</div></body></html>"""


def save_html(rep: TruthReport, path: str, *, data_note: str = "") -> str:
    import pathlib
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(rep, data_note=data_note), encoding="utf-8")
    return str(p)
