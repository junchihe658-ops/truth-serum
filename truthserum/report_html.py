"""把体检报告渲染成单文件 HTML —— 一份排过版的文档，不是一摞色块

## 三次返工换来的三条结论

1. **深色仪表盘不行。** 霓虹绿红 + emoji 图标 + 圆角卡是生成式前端的默认长相。
2. **全浅色也不行。** 第一版纸底 #f4f1ea 和纸张 #fffefb 只差 2%，整页塌成白纸。
3. **通栏色块堆叠最不行 —— 那就是 PPT。** 每块 100% 宽、左右边距一样、上下摞起来，
   再配三个不同颜色的「左边条 + 底色」提示框，就是套模板的样子。

## 所以这一版靠结构，不靠配色

· **两栏网格：左窄栏放标签，右版心放内容。** 所有标签右对齐贴在同一条竖线上，
  发丝线横贯两栏。这条对齐轴是「排过版」与「堆盒子」的分界，比任何配色都管用。

· **版心限宽 ~62 字符。** 行宽跑满的文档不可能显得讲究。

· **一个底色都不留。** 层级只用字号、字重、留白和发丝线做。
  唯一的大色块是顶部深松绿报头，用来压住第一屏、防止整页发白。

· **三层围合，但都不靠色块。** 只有对齐、没有围合的版面会显得内容「浮」着 ——
  ① 纸张有实边（1px 描边 + 暖阴影，浮在明显更深的纸底上）；
  ② 左栏与版心之间一条通贯全文的发丝竖线，让网格从「对齐」变成「围合」；
  ③ 报头底部 2px 黄铜线收口。

· **编号排版化：⓪①② → 00 01 02。** CJK 圈码字符各系统渲染不一，本身就像
  模板项目符号；两位数字 + 字距才是排版。

## 一以贯之的取舍

· **单文件、零外部依赖** —— 不引 CDN、不引字体。评审可能没网，而且一个讲
  『不要盲信』的工具自己去外部拉资源说不过去。代价是没有独家字形，中文标题
  落在宋体，识别度只能靠版式。这是清醒的取舍，不是偷懒。

· **不用 emoji 当状态图标。** ✅❌ 在三个系统上是三种画风。这里用内联 SVG
  几何符号 + 中文判读词，形状与文字双编码 —— 黑白打印、色盲读者一样读得出。

· **第一屏只放三样**：叫什么、自称多少、判读是什么。落差本身就是叙事。

· **不折叠。** 评审只花两三分钟，要点击才展开的东西等于没写。

· **能打印。** @media print 把报头翻成浅色，省墨，且仍然是一份文档。
"""
from __future__ import annotations

import html
import re
from datetime import datetime

from .audit import TruthReport, Verdict

_STYLE = """
:root{
  --ground:#cec1a2;                     /* ground the sheet sits on */
  --paper:#fbf8f1;                      /* 奶油纸：纸张本身 */
  --head:#15322d; --head-ink:#eee8da;   /* 深松绿报头：唯一的大色块 */
  --ink:#1c1a14; --ink2:#575346; --ink3:#8e8776;
  --hair:#e0d9c6; --hair2:#efe9db;      /* 发丝线 */
  --brass:#94702f; --brass-l:#d0a765;   /* 黄铜：标签与对齐轴 */
  --pass:#3c6b39; --fail:#a0361d; --void:#87610f; --skip:#8e8776;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;color:var(--ink);background:var(--ground);
  font:15px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
  "Hiragino Sans GB","Microsoft YaHei",sans-serif;
  font-variant-numeric:tabular-nums;
  -webkit-font-smoothing:antialiased}
.serif{font-family:Georgia,"Times New Roman","Songti SC","SimSun",serif}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,Menlo,monospace}

/* ── 纸张与网格：纸有实边，左栏与版心之间一条通贯全文的发丝竖线 ── */
/* 纸宽 = 左padding 46 + 左栏 172 + 版心留白 33 + 版心 ~515 + 右padding
   —— 定成 1000 会在右边空出近 190px 的死角，左右不对称。 */
.page{max-width:840px;margin:46px auto 76px;background:var(--paper);
  border:1px solid #d5cbb2;
  /* 阴影必须廉价：模糊半径大的阴影，重绘代价随元素面积走，
     而这张纸有三千多像素高 —— 实测会让滚动掉帧、截图超时。
     边界主要靠上面那道 1px 描边，阴影只负责一点点浮起感。 */
  box-shadow:0 1px 2px rgba(58,46,22,.07), 0 6px 14px -8px rgba(58,46,22,.22)}
.grid{display:grid;grid-template-columns:172px minmax(0,1fr);
  column-gap:0;padding:0 46px}
.rail{text-align:right;font-size:10px;letter-spacing:.24em;color:var(--brass);
  padding:26px 26px 0 0;line-height:1.7;border-right:1px solid var(--hair)}
.main{padding:24px 0 30px 32px;max-width:62ch}
.rail.hr,.main.hr{border-top:1px solid var(--hair)}
.rail.hr2,.main.hr2{border-top:2px solid var(--ink)}
.doc{padding-bottom:64px}

/* ── 报头：唯一的大色块，用同一套网格，让对齐轴从这里就开始 ────── */
.mast{background:var(--head);color:var(--head-ink);padding:32px 0 36px;
  background-image:linear-gradient(158deg,#1b3d36 0%,#15322d 52%,#0f2621 100%);
  border-bottom:2px solid var(--brass-l)}
.mast .rail{color:var(--brass-l);padding-top:11px;
  border-right-color:rgba(238,232,218,.20)}
.mast .main{padding:0;max-width:none}
/* 编号跟着品牌走同一条轴 —— 浮到标题行里会和折行的标题撞上 */
.mast .id{display:block;margin-top:8px;font-size:10.5px;letter-spacing:.1em;
  color:rgba(238,232,218,.38)}
.mast h1{font-size:clamp(27px,3.7vw,40px);line-height:1.16;margin:0;
  font-weight:400;color:#fdfaf3;letter-spacing:-.018em}
.mast .spec{margin-top:20px;padding-top:14px;font-size:11.5px;line-height:1.75;
  color:rgba(238,232,218,.48);border-top:1px solid rgba(238,232,218,.14)}

/* ── 送检声明 / 判读：落差叙事，中间不插东西 ───────────────── */
.claim .v{font-size:21px;line-height:1.5;color:var(--ink2)}
.verdict .big{font-size:31px;line-height:1.28;letter-spacing:-.012em}
.verdict .sub{font-size:13.5px;color:var(--ink2);margin-top:13px;line-height:1.65}
.pass{color:var(--pass)} .fail{color:var(--fail)}
.void{color:var(--void)} .skip{color:var(--skip)}

/* ── 分项一览 ─────────────────────────────────────────────── */
table.res{width:100%;border-collapse:collapse}
table.res td{padding:9px 0;border-bottom:1px solid var(--hair2);font-size:14.5px}
table.res tr:last-child td{border-bottom:0}
table.res td.no{width:42px;font-size:11px;letter-spacing:.14em;color:var(--ink3);
  font-family:ui-monospace,SFMono-Regular,Consolas,Menlo,monospace}
table.res td.r{text-align:right;white-space:nowrap;font-size:13.5px}
.m{width:12px;height:12px;fill:none;stroke:currentColor;stroke-width:2.2;
  stroke-linecap:round;stroke-linejoin:round;vertical-align:-1px;margin-right:7px}

/* ── 闸门明细：编号与判读进左栏，正文守版心 ────────────────── */
.rail .gno{display:block;font-size:19px;letter-spacing:.06em;color:var(--ink3);
  font-family:ui-monospace,SFMono-Regular,Consolas,Menlo,monospace;
  margin-bottom:9px;line-height:1}
.rail .gjd{display:block;font-size:10px;letter-spacing:.18em}
.gate .ttl{font-size:19px;line-height:1.4}
.gate .note{margin:3px 0 0;font-size:12.5px;color:var(--ink3);line-height:1.55}
.gate .lead{margin:15px 0 0;font-size:15.5px;line-height:1.65}
.sc{border-left:2px solid var(--brass-l);padding-left:17px;margin:19px 0 0;
  font-size:13.5px;color:var(--ink2);line-height:1.7;max-width:56ch}
.sc .lb{display:block;font-size:10px;letter-spacing:.2em;color:var(--brass);
  margin-bottom:5px}
ul.det{margin:17px 0 0;padding:0;list-style:none;font-size:13.5px;color:var(--ink2)}
ul.det li{position:relative;padding-left:17px;margin:6px 0;line-height:1.7}
ul.det li:before{content:"\\2014";position:absolute;left:0;color:var(--ink3)}
.nums{display:grid;grid-template-columns:auto 1fr;gap:3px 16px;
  margin:18px 0 0;font-size:11.5px;
  font-family:ui-monospace,SFMono-Regular,Consolas,Menlo,monospace}
.nums dt{color:var(--ink3)} .nums dd{margin:0;color:var(--ink2)}
.catches{margin:17px 0 0;font-size:12px;color:var(--ink3);line-height:1.65;
  padding-top:11px;border-top:1px solid var(--hair2)}

/* ── 页脚说明（只作用在最后一行，不许波及其它版心） ──────────── */
.foot{font-size:12px;line-height:1.95;color:var(--ink3);max-width:70ch}
.foot b{color:var(--ink2);font-weight:600}
.foot .cmd{color:var(--ink2);word-break:break-all}
.srcwrap{overflow-x:auto;margin:0 0 20px}
.foot table{border-collapse:collapse}
.foot table td{padding:1px 16px 1px 0;font-size:11px;white-space:nowrap;
  vertical-align:top;color:var(--ink3);
  font-family:ui-monospace,SFMono-Regular,Consolas,Menlo,monospace}

@media(max-width:760px){
  .page{margin:0;border:0;box-shadow:none;max-width:none}
  .grid{grid-template-columns:1fr;padding:0 22px}
  .rail{text-align:left;padding:20px 0 0;border-right:0}
  .main{padding:8px 0 26px;max-width:none}
  .main.hr,.main.hr2{border-top:0}
  .mast{padding:24px 0 28px}
  .mast .rail{padding-top:0}
  .mast h1{font-size:28px;margin-top:16px}
  .mast .id{float:none;display:block;margin-top:4px}
  .verdict .big{font-size:25px}
  .rail .gno{display:inline;margin-right:12px}
  .rail .gjd{display:inline}
  .sc,ul.det,.foot{max-width:none}
  .foot table td{white-space:normal}
}
@media print{
  body{background:#fff}
  .page{margin:0;border:0;box-shadow:none;max-width:none}
  .mast{background:#fff;background-image:none;color:var(--ink);
    padding:0 0 26px;border-bottom:2px solid var(--ink)}
  .mast h1{color:var(--ink)}
  .mast .rail{color:var(--brass);border-right-color:var(--hair)}
  .mast .id,.mast .spec{color:var(--ink3)}
  .mast .spec{border-top-color:var(--hair)}
  .main.gate{break-inside:avoid}
}
"""

#: 判读记号：内联 SVG（形状） + 中文词（文字），双编码。
_MARK = {
    Verdict.CLEAN: ('<path d="M2.6 8.4 6.3 12.1 13.4 4.3"/>', "未见异常", "pass"),
    Verdict.FAILED: ('<path d="M3.6 3.6 12.4 12.4M12.4 3.6 3.6 12.4"/>', "异常", "fail"),
    Verdict.UNUSABLE: ('<circle cx="8" cy="8" r="5.7"/><path d="M4 12 12 4"/>',
                       "无效", "void"),
    Verdict.SKIPPED: ('<path d="M3.4 8h9.2"/>', "未检", "skip"),
}

#: 圈码字符各系统渲染不一，看着像模板项目符号 —— 换成排版数字
_CIRCLED = "⓪①②③④⑤⑥⑦⑧⑨"


def _esc(s) -> str:
    return html.escape(str(s))


def _mark(v: Verdict) -> str:
    path, word, cls = _MARK[v]
    return (f'<span class="{cls}"><svg class="m" viewBox="0 0 16 16" '
            f'aria-hidden="true">{path}</svg>{word}</span>')


def _gate_no(num: str) -> str:
    """⓪ → 00。认不出的原样返回，绝不编。"""
    i = _CIRCLED.find(num)
    return f"{i:02d}" if i >= 0 else num


def _split_name(n: str) -> tuple[str, str, str]:
    """「⓪ 数据出处核验（K 线本身是不是真的）」→ ('⓪', '数据出处核验', 'K 线…')"""
    n = str(n).strip()
    num, _, rest = n.partition(" ")
    if not rest:                       # 没有编号前缀
        num, rest = "", n
    short, _, note = rest.partition("（")
    return num, short.strip(), note.rstrip("）").strip()


def _fmt(v) -> str:
    """把原始读数排成人能看的样子 —— 不再往页面上倒 16 位小数"""
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
    except Exception:                  # 将来塞进别的类型时不许整页炸掉
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


def _spec_line(rows: list[list[str]], now: datetime) -> str:
    """报头那一行摘要。凑不出就只说凑得出的部分 —— 不猜、不编。"""
    bits: list[str] = []
    if rows:
        bits.append(f"{len(rows)} 个标的")
    try:
        tfs = {r[1] for r in rows}
        bars = {r[2] for r in rows}
        if len(tfs) == 1:
            bits.append(next(iter(tfs)))
        if len(bars) == 1:
            bits.append(f"{next(iter(bars))}/标的")
        lo, _, hi = rows[0][3].partition("~")
        bits.append(f"{lo.strip()[:10]} → {hi.strip()[:10]}")
        bits.append(rows[0][4].strip("[]").split(",")[0].strip())
    except Exception:
        pass
    bits.append(f"出具于 {now:%Y-%m-%d %H:%M}")
    return " · ".join(bits)


def _verdict_parts(rep: TruthReport) -> tuple[str, str, str]:
    """复用 verdict_line() 的措辞（免得两处逻辑漂移），但剥掉 emoji 前缀"""
    line = re.sub(r"^[✅❌⛔⏭]\s*", "", rep.verdict_line())
    big, sep, sub = line.partition("（")
    sub = sub.rstrip("）") if sep else ""
    if not sub and "。" in big:        # 整句太长会撑爆大标题，按第一个句号断开
        big, _, sub = big.partition("。")
        sub = sub.strip()
    if rep.any_unusable:
        cls = "void"
    elif rep.failures:
        cls = "fail"
        sub = ("未通过：" + sub.replace("、", " · ")) if sub else ""
    else:
        cls = "pass"
    return cls, big.rstrip("：:"), sub


def render_html(rep: TruthReport, *, data_note: str = "") -> str:
    now = datetime.now()
    cls, big, sub = _verdict_parts(rep)
    srcs = _src_rows(data_note)
    cells: list[str] = []

    def row(label: str, body: str, *, main_cls: str = "", rule: str = "hr"):
        """一条网格行：左栏标签 + 右栏版心，共用一条横贯发丝线"""
        cells.append(f'<div class="rail {rule}">{label}</div>')
        cells.append(f'<div class="main {rule} {main_cls}">{body}</div>')

    if rep.claimed:
        row("送检声明", f'<div class="v serif">{_esc(rep.claimed)}</div>',
            main_cls="claim")

    sub_h = f'<div class="sub">{_esc(sub)}</div>' if sub else ""
    row("判读", f'<div class="big serif {cls}">{_esc(big)}</div>{sub_h}',
        main_cls="verdict")

    res = "".join(
        f'<tr><td class="no">{_esc(_gate_no(_split_name(r.name)[0]))}</td>'
        f'<td>{_esc(_split_name(r.name)[1])}</td>'
        f'<td class="r">{_mark(r.verdict)}</td></tr>'
        for r in rep.results)
    row("分项一览", f'<table class="res"><tbody>{res}</tbody></table>')

    for i, r in enumerate(rep.results):
        num, short, note = _split_name(r.name)
        _, word, vcls = _MARK[r.verdict]
        rail = (f'<span class="gno">{_esc(_gate_no(num))}</span>'
                f'<span class="gjd {vcls}">{_esc(word)}</span>')
        parts = [f'<div class="ttl">{_esc(short)}</div>']
        if note:
            parts.append(f'<div class="note">{_esc(note)}</div>')
        parts.append(f'<p class="lead">{_esc(r.headline)}</p>')
        if r.self_check_evidence:
            parts.append(
                '<div class="sc"><span class="lb">自检 · 先在人为植入的 BUG 上'
                f'证明自己抓得到</span>{_esc(r.self_check_evidence)}</div>')
        if r.detail:
            lis = "".join(f"<li>{_esc(d)}</li>" for d in r.detail)
            parts.append(f'<ul class="det">{lis}</ul>')
        if r.numbers:
            kv = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(_fmt(v))}</dd>"
                         for k, v in r.numbers.items())
            parts.append(f'<dl class="nums">{kv}</dl>')
        if r.catches:
            parts.append(f'<div class="catches">本闸门针对：{_esc(r.catches)}</div>')
        row(rail, "".join(parts), main_cls="gate", rule="hr2" if i == 0 else "hr")

    src_tbl = ""
    if srcs:
        body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r_)
                       + "</tr>" for r_ in srcs)
        src_tbl = f'<div class="srcwrap"><table>{body}</table></div>'
    row("说明",
        f'{src_tbl}每一道闸门在给出结论之前，都必须先在<b>人为植入的 bug</b> 上'
        '证明自己抓得到；自检不过就返回「不可判定」，而不是「一切正常」。<br>'
        '一个永远报平安的检测器，比没有检测器更危险。<br><br>'
        '装成 MCP：<span class="cmd mono">claude mcp add truth-serum '
        '-- python -m truthserum.server</span>',
        main_cls="foot", rule="hr2")

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>策略体检报告 · {_esc(rep.strategy_name)}</title>
<style>{_STYLE}</style></head><body><div class="page">
<header class="mast"><div class="grid">
  <div class="rail">TRUTH SERUM<span class="id mono">{now:TS-%Y%m%d-%H%M}</span></div>
  <div class="main">
    <h1 class="serif">{_esc(rep.strategy_name)}</h1>
    <div class="spec">{_esc(_spec_line(srcs, now))}</div>
  </div>
</div></header>
<main class="doc grid">{''.join(cells)}</main>
</div></body></html>"""


def save_html(rep: TruthReport, path: str, *, data_note: str = "") -> str:
    import pathlib
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(rep, data_note=data_note), encoding="utf-8")
    return str(p)
