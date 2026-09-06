"""自然语言 → 策略代码 —— 一个宁可拒绝也不肯猜的翻译器

    parse("RSI 超过 70 做空、低于 30 做多、持 12 小时")
      → Spec，能 .explain() 给人看，能 .to_code() 变成 signal(bars)

## 为什么不调 LLM

这一层是**确定性的**：一套明确的词汇表 + 一个只认这套词汇的解析器。
没有 API key、没有额外依赖、同样的话永远得到同样的代码。

代价必须说清楚：**它只懂词汇表里的东西**。说「MACD 金叉就买」它不会装懂，
会告诉你 MACD 不在词汇表里。这不是缺陷，是这一层唯一的立身之本 ——
一个把你的话猜成代码的翻译器，比不翻译更危险：你会拿着一份体检报告，
以为审的是你想的那个策略。

真要超出词汇表，正常路径是让你的 Claude 直接写 `signal(bars)` 代码，
走 `audit_strategy`。那条路上代码是模型写的、你能看见、你自己负责。

## 三条硬规矩

1. **没消费掉的字必须报出来。** 解析完还剩字没认出来，就是解析失败，
   不许当噪音丢掉 —— 丢掉的那半句很可能正是策略的关键。

2. **歧义不许自己选。** 「跌破 30」到底是「下穿那一根」还是「只要低于 30」，
   这两种意思做出来的策略完全不同。解析器按一种理解走，但必须把这个
   理解**回读给你确认**，并告诉你另一种怎么说。

3. **生成的代码天生因果。** 词汇表里每个算子都只用当根及之前的数据
   （`.shift(1)` 允许，`shift(-1)`、`center=True`、`bfill` 一个都没有）。
   ——但生成的代码**照样要过 ① 号闸门**。我们不相信自己的构造，
   这正是这个项目的全部主张。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class CannotParse(ValueError):
    """解析失败。带上到底哪一段没看懂 —— 只说「解析失败」等于没说。"""

    def __init__(self, fragment: str, why: str, hint: str = ""):
        self.fragment, self.why, self.hint = fragment, why, hint
        msg = f"没看懂「{fragment}」：{why}"
        if hint:
            msg += f"\n{hint}"
        super().__init__(msg)


# ────────────────────────────────────────────────────────────
# 周期换算：「12 小时」在 1h 上是 12 根，在 4h 上是 3 根
# ────────────────────────────────────────────────────────────
_INTERVAL_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                 "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
                 "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080}

_UNIT_MIN = {"根": None, "条": None, "个": None, "k线": None, "K线": None,
             "分钟": 1, "分": 1, "小时": 60, "时": 60,
             "天": 1440, "日": 1440}


def _to_bars(n: int, unit: str, interval: str) -> tuple[int, str]:
    """把「N 个单位」换算成多少根 K 线，返回 (根数, 给人看的说明)。

    单位是「根/条/个」就直接是根数；是时间单位就得除以周期长度 ——
    这一步不做的话，「持 12 小时」在 4h 周期上会被当成 12 根 = 48 小时，
    静默地把策略改成另一个策略。
    """
    unit = (unit or "根").strip()
    per = _UNIT_MIN.get(unit, "?")
    if per == "?":
        raise CannotParse(unit, "不认识这个单位",
                          "认识的：根 / 条 / 个 / 分钟 / 小时 / 天")
    if per is None:                       # 本来就是根数
        return n, f"{n} 根"
    im = _INTERVAL_MIN.get(interval)
    if im is None:
        raise CannotParse(interval, f"不认识的 K 线周期，无法把「{n}{unit}」换成根数",
                          f"认识的周期：{', '.join(_INTERVAL_MIN)}")
    total = n * per
    bars = total // im
    if bars < 1:
        raise CannotParse(f"{n}{unit}", f"比一根 {interval} K 线还短，换算下来是 0 根")
    note = f"{n}{unit} = {bars} 根 {interval} K 线"
    if total % im:
        note += f"（除不尽，已向下取整；{n}{unit} 实为 {total/im:.2f} 根）"
    return bars, note


# ────────────────────────────────────────────────────────────
# 词汇表
# ────────────────────────────────────────────────────────────
@dataclass
class Expr:
    """条件左右两边的一个量"""
    code: str                  # 生成代码里的表达式
    label: str                 # 回读给人看的写法
    var: str = ""              # 需要在前言里赋值时的变量名
    setup: str = ""            # 前言那行代码
    is_pct: bool = False       # 是不是百分比量（涨跌幅）
    note: str = ""             # 换算说明，进 warnings


# 顺序有讲究：长的、具体的写在前面，否则 "超过" 会被 "超" 先吃掉一半
_OPS: list[tuple[str, str]] = [
    (r"上穿|向上突破|上破|突破", "cross_up"),
    (r"下穿|向下突破|跌破|击穿", "cross_down"),
    (r"大于等于|不低于|不小于|>=|≥", "ge"),
    (r"小于等于|不高于|不大于|<=|≤", "le"),
    (r"超过|大于|高于|超|>|＞", "gt"),
    (r"低于|小于|不足|<|＜", "lt"),
]
_OP_LABEL = {"gt": ">", "lt": "<", "ge": "≥", "le": "≤",
             "cross_up": "上穿", "cross_down": "下穿"}

_ACTIONS: list[tuple[str, float, str]] = [
    (r"做多|买入|开多|多头|long", 1.0, "做多"),
    (r"做空|卖出|开空|空头|short", -1.0, "做空"),
    (r"观望|空仓|平仓|不动|flat", 0.0, "观望"),
]

_JOIN = r"并且|而且|同时|且|和|与|and"
_SPLIT = r"[、，,；;。\n]+"

#: 允许出现在条件里、但不带信息的字。多余的字必须报错，这些除外。
_NOISE_HEAD = ("当", "若", "如果", "一旦", "只要", "假如")
_NOISE_TAIL = ("的时候", "时候", "时", "就", "则", "那么", "便", "的")


def _strip_noise(s: str) -> str:
    s = s.strip()
    changed = True
    while changed:
        changed = False
        for w in _NOISE_HEAD:
            if s.startswith(w):
                s, changed = s[len(w):].strip(), True
        for w in _NOISE_TAIL:
            if s.endswith(w):
                s, changed = s[:-len(w)].strip(), True
    return s


def _norm(s: str) -> str:
    """全角转半角，免得 ＞７０ 这种输入白白解析失败"""
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF10 <= o <= 0xFF19 or o in (0xFF05, 0xFF08, 0xFF09, 0xFF0E):
            out.append(chr(o - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def _match_expr(s: str, pos: int, interval: str) -> tuple[Expr, int] | None:
    """在 s[pos:] 开头尝试认出一个量。认不出返回 None —— 由调用方报错。

    ⚠ 必须先吃掉前导空白：比较词后面几乎必然跟着空格
      （"RSI 超过 EMA12"），不跳的话右边那个量永远认不出来。
    """
    rest = s[pos:]
    lead = len(rest) - len(rest.lstrip())
    rest, pos = rest.lstrip(), pos + lead

    # 周期必须紧贴（RSI14）或带括号（RSI(14)）。写成「RSI 70」时那个 70
    # 多半是阈值不是周期 —— 隔着空格就不当周期，让它去比较词那步报错。
    m = re.match(r"(?i)rsi(?:\s*\(\s*(\d+)\s*\)|(\d+))?", rest)
    if m:
        n = int(m.group(1) or m.group(2) or 14)
        return Expr(f"rsi(c, {n})", f"RSI({n})", f"v_rsi{n}",
                    f"v_rsi{n} = rsi(c, {n})"), pos + m.end()

    m = re.match(r"(?i)ema\s*\(?\s*(\d+)\s*\)?", rest)
    if m:
        n = int(m.group(1))
        return Expr(f"ema(c, {n})", f"EMA({n})", f"v_ema{n}",
                    f"v_ema{n} = ema(c, {n})"), pos + m.end()

    # MA20 / SMA20 / 20日均线 / 20根均线
    m = re.match(r"(?i)(?:sma|ma)\s*\(?\s*(\d+)\s*\)?", rest)
    unit = None
    if not m:
        m = re.match(r"(\d+)\s*(根|条|日|天|小时)?\s*(?:均线|移动平均|平均线)", rest)
        if m:
            unit = m.group(2)
    if m:
        n = int(m.group(1))
        note = ""
        if unit in ("日", "天", "小时"):
            note = (f"「{n}{unit}均线」按 {n} 根 K 线算（当前周期 {interval}）。"
                    f"若你要的是 {n} 个自然{unit}，请直接写根数。")
        return Expr(f"sma(c, {n})", f"MA({n})", f"v_sma{n}",
                    f"v_sma{n} = sma(c, {n})", note=note), pos + m.end()

    m = re.match(r"(?i)atr(?:\s*\(\s*(\d+)\s*\)|(\d+))?", rest)
    if m:
        n = int(m.group(1) or m.group(2) or 14)
        return Expr(f"atr(bars, {n})", f"ATR({n})", f"v_atr{n}",
                    f"v_atr{n} = atr(bars, {n})"), pos + m.end()

    m = re.match(r"(?i)(?:布林(?:带)?(?:位置)?|bb_?pct|bb)"
                 r"(?:\s*\(\s*(\d+)\s*\)|(\d+))?", rest)
    if m:
        n = int(m.group(1) or m.group(2) or 20)
        return Expr(f"bb_pct(c, {n})", f"布林位置({n})", f"v_bb{n}",
                    f"v_bb{n} = bb_pct(c, {n})"), pos + m.end()

    # N 小时涨幅 / N 根涨跌幅 / N 小时跌幅
    #
    # ⚠ 「跌幅」必须取反，不能和「涨幅」当成同一个量。
    #   初版把 涨幅/跌幅/涨跌幅 一律映射成 pct_change，于是
    #   「24小时跌幅超过 3% 做空」被解析成「涨跌幅 > 3% → 做空」——
    #   那是【上涨】3%，方向整个反了，而且不报错，静默地做了错误解释。
    #   一个宁可拒绝也不猜的解析器，绝不能有这种静默误读。
    m = re.match(r"(\d+)\s*(根|条|个|分钟|分|小时|时|天|日)?\s*"
                 r"(涨幅|跌幅|涨跌幅|收益率|变动)", rest)
    if m:
        n, unit, word = int(m.group(1)), m.group(2), m.group(3)
        bars, note = _to_bars(n, unit or "根", interval)
        u = unit or "根"
        if word == "跌幅":
            return Expr(f"(-c.pct_change({bars}))", f"{n}{u}跌幅",
                        f"v_drop{bars}", f"v_drop{bars} = -c.pct_change({bars})",
                        is_pct=True,
                        note=(note if unit not in (None, "根", "条", "个") else "")
                        ), pos + m.end()
        return Expr(f"c.pct_change({bars})", f"{n}{u}涨跌幅",
                    f"v_ret{bars}", f"v_ret{bars} = c.pct_change({bars})",
                    is_pct=True,
                    note=(note if unit not in (None, "根", "条", "个") else "")
                    ), pos + m.end()

    # 没写时间跨度的「涨幅 / 跌幅」—— 报错要说到点子上。
    # 原先只会说「开头这个量不在词汇表里」，可它明明在词汇表里，
    # 缺的只是跨度。这种报错等于把人往错方向指。
    m = re.match(r"(涨幅|跌幅|涨跌幅|收益率|变动)", rest)
    if m:
        raise CannotParse(
            m.group(1), f"「{m.group(1)}」前面要写时间跨度，不然不知道是多久的",
            f"例如「24小时{m.group(1)}」「12根{m.group(1)}」。"
            f"上一条写了跨度也不会自动带过来 —— 跨度不一样，说的就是两件事。")

    m = re.match(r"收盘价|收盘|价格|close", rest)
    if m:
        return Expr("c", "收盘价"), pos + m.end()

    m = re.match(r"成交量|volume", rest)
    if m:
        return Expr('bars["volume"]', "成交量", "v_vol",
                    'v_vol = bars["volume"]'), pos + m.end()

    return None


def _match_op(s: str, pos: int) -> tuple[str, int] | None:
    for pat, name in _OPS:
        m = re.match(rf"\s*(?:{pat})", s[pos:])
        if m:
            return name, pos + m.end()
    return None


def _match_number(s: str, pos: int, lhs: Expr) -> tuple[float, str, int] | None:
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*(%)?", s[pos:])
    if not m:
        return None
    raw, pct = float(m.group(1)), bool(m.group(2))
    if lhs.is_pct and not pct:
        raise CannotParse(
            m.group(0).strip(),
            f"「{lhs.label}」是百分比量，而这里只写了个 {m.group(1)}，"
            f"我不敢替你决定它是 {m.group(1)}% 还是 {m.group(1)} 倍",
            f"请写明白，例如「{lhs.label}超过 {m.group(1)}%」")
    if pct and not lhs.is_pct:
        raise CannotParse(
            m.group(0).strip(),
            f"「{lhs.label}」不是百分比量，写成 {m.group(1)}% 我不敢当成 {m.group(1)}",
            f"如果就是 {m.group(1)}，把 % 去掉")
    val = raw / 100 if pct else raw
    label = f"{m.group(1)}%" if pct else m.group(1)
    return val, label, pos + m.end()


# ────────────────────────────────────────────────────────────
# 结构
# ────────────────────────────────────────────────────────────
@dataclass
class Cond:
    lhs: Expr
    op: str
    rhs: Expr | float
    rhs_label: str

    @property
    def label(self) -> str:
        return f"{self.lhs.label} {_OP_LABEL[self.op]} {self.rhs_label}"

    def code(self) -> str:
        L = self.lhs.var or self.lhs.code
        if isinstance(self.rhs, Expr):
            R = self.rhs.var or self.rhs.code
        else:
            R = repr(self.rhs)
        if self.op in ("gt", "lt", "ge", "le"):
            sym = {"gt": ">", "lt": "<", "ge": ">=", "le": "<="}[self.op]
            return f"({L} {sym} {R})"
        # 穿越：这一根在那边、上一根不在 —— 只用到上一根，仍然是因果的
        prev_r = f"{R}.shift(1)" if isinstance(self.rhs, Expr) else R
        if self.op == "cross_up":
            return f"(({L} > {R}) & ({L}.shift(1) <= {prev_r}))"
        return f"(({L} < {R}) & ({L}.shift(1) >= {prev_r}))"


@dataclass
class Rule:
    conds: list[Cond]
    action: float
    action_label: str
    raw: str

    @property
    def label(self) -> str:
        return " 且 ".join(c.label for c in self.conds)


@dataclass
class Spec:
    rules: list[Rule]
    source: str
    interval: str
    horizon: int | None = None
    barrier_mult: float | None = None
    warnings: list[str] = field(default_factory=list)

    # ── 回读：这是整个模块最重要的方法 ──────────────────────
    def explain(self) -> str:
        """把解析结果用人话读回去。**必须在跑审计之前给人看。**

        解析器可能理解错。理解错而用户不知道，他会拿着一份体检报告，
        以为审的是他想的那个策略 —— 那正是这个项目要消灭的东西。
        """
        out = [f"原话：{self.source}", "",
               "我理解成这样（请核对）："]
        for i, r in enumerate(self.rules, 1):
            out.append(f"  {i}. {r.label:<34} → {r.action_label}")
        out.append("")
        out.append("  规则按顺序匹配，先命中的生效；一条都不命中则观望。")
        if self.horizon:
            out.append(f"  时间屏障：持有 {self.horizon} 根 K 线后强制平仓。")
        if self.barrier_mult:
            out.append(f"  止盈止损：各 {self.barrier_mult} × ATR。")
        if self.warnings:
            out += ["", "⚠ 这几处我替你做了选择，看看对不对："]
            out += [f"  · {w}" for w in self.warnings]
        return "\n".join(out)

    # ── 生成代码 ────────────────────────────────────────────
    def to_code(self) -> str:
        setups, seen = [], set()
        for r in self.rules:
            for cnd in r.conds:
                for e in (cnd.lhs, cnd.rhs):
                    if isinstance(e, Expr) and e.setup and e.var not in seen:
                        seen.add(e.var)
                        setups.append("    " + e.setup)
        head = [
            "def signal(bars):",
            '    """由 Truth Serum 从自然语言生成，未经人手修改。',
            "",
            f"    原话：{self.source}",
            "",
            "    规则按顺序匹配，先命中的生效。",
            '    """',
            '    c = bars["close"]',
        ]
        body = ["",
                "    sig  = np.zeros(len(bars), dtype=float)",
                "    done = np.zeros(len(bars), dtype=bool)"]
        for i, r in enumerate(self.rules, 1):
            expr = " & ".join(c.code() for c in r.conds)
            body += ["",
                     f"    # {i}. {r.label} → {r.action_label}",
                     f"    m = ({expr}).to_numpy()",
                     f"    sig, done = np.where(~done & m, {r.action}, sig), (done | m)"]
        body += ["", "    return pd.Series(sig, index=bars.index)"]
        return "\n".join(head + setups + body) + "\n"

    def to_strategy(self):
        """把生成的代码变成可调用的 signal(bars)。

        这里 exec 的是**我们自己生成的**代码 —— 只可能用到词汇表里的算子，
        和 exec 一段模型写的代码不是一个风险级别。
        """
        from .indicators import TOOLBOX
        ns = dict(TOOLBOX)
        exec(self.to_code(), ns)          # noqa: S102 —— 见上面注释
        return ns["signal"]


# ────────────────────────────────────────────────────────────
# 解析
# ────────────────────────────────────────────────────────────
def _extract_globals(text: str, interval: str) -> tuple[str, int | None,
                                                        float | None, list[str]]:
    horizon = barrier = None
    notes: list[str] = []

    # 长的写前面：`持有?` 会先吃掉「持」，让「持仓 20 根」整条匹配失败
    m = re.search(r"(?:持仓|持有|持|拿)\s*(\d+)\s*(根|条|个|分钟|分|小时|时|天|日)?"
                  r"\s*(?:k线|K线)?", text)
    if m:
        horizon, note = _to_bars(int(m.group(1)), m.group(2) or "根", interval)
        if m.group(2) not in (None, "根", "条", "个"):
            notes.append(f"持仓时长：{note}")
        text = text[:m.start()] + text[m.end():]

    # 前缀必需：写成可选的话，条件里的「高于 2 倍 ATR」会被当成全局设定吃掉，
    # 留下一个残缺的条件去报一个莫名其妙的错
    m = re.search(r"(?:止盈止损|止损止盈|止盈|止损|屏障)\s*(?:各)?\s*"
                  r"(\d+(?:\.\d+)?)\s*倍?\s*(?i:atr)", text)
    if m:
        barrier = float(m.group(1))
        text = text[:m.start()] + text[m.end():]

    return text, horizon, barrier, notes


def _parse_cond(chunk: str, interval: str,
                prev_lhs: Expr | None = None) -> tuple[Cond, list[str]]:
    s = _strip_noise(_norm(chunk))
    if not s:
        raise CannotParse(chunk, "条件是空的")

    ellipsis = ""
    got = _match_expr(s, 0, interval)
    if got:
        lhs, i = got
    elif prev_lhs is not None and _match_op(s, 0):
        # 省略主语：「RSI 超过 70 做空、低于 30 做多」第二条省掉了 RSI。
        # 中文里这是最自然的写法，但【补主语就是推断】——
        # 所以补，然后必须回读出来让人核对。
        lhs, i = prev_lhs, 0
        ellipsis = (f"「{s}」省略了主语，我按上一条的「{prev_lhs.label}」理解。"
                    f"若指的是别的量，请写全。")
    else:
        raise CannotParse(s, "开头这个量不在词汇表里", VOCAB_HINT)

    got_op = _match_op(s, i)
    if got_op is None:
        raise CannotParse(s[i:] or s, f"「{lhs.label}」后面没看到比较词",
                          "认识的比较词：超过/大于/高于、低于/小于、"
                          "上穿/突破、下穿/跌破、不低于、不高于")
    op, i = got_op

    notes = [n for n in (lhs.note, ellipsis) if n]
    rhs_expr = _match_expr(s, i, interval)
    if rhs_expr:
        rhs, i = rhs_expr[0], rhs_expr[1]
        rhs_label = rhs.label
        if rhs.note:
            notes.append(rhs.note)
    else:
        num = _match_number(s, i, lhs)
        if not num:
            raise CannotParse(s[i:] or s,
                              f"「{lhs.label} {_OP_LABEL[op]}」后面没看到数值或指标",
                              VOCAB_HINT)
        rhs, rhs_label, i = num[0], num[1], num[2]

    tail = _strip_noise(s[i:])
    if tail:
        raise CannotParse(tail, "这几个字我没认出来，不敢当噪音丢掉",
                          "整句里每个字都要有着落，否则生成的代码就不是你说的那个策略")

    if op in ("cross_up", "cross_down"):
        d = "上穿" if op == "cross_up" else "下穿"
        other = "高于" if op == "cross_up" else "低于"
        notes.append(
            f"「{d}」我理解成【只在穿越那一根触发】"
            f"（上一根在另一侧、这一根越过来）。"
            f"若你要的是「只要{other}就一直算数」，请改说「{other}」。")
    return Cond(lhs, op, rhs, rhs_label), notes


def parse(description: str, interval: str = "1h") -> Spec:
    """把一句话变成 Spec。看不懂就抛 CannotParse，绝不猜。

    interval 会影响「12 小时」这类说法换算成多少根 K 线 —— 传错了，
    生成的策略就不是你说的那个。
    """
    if not description or not description.strip():
        raise CannotParse("(空)", "什么都没说")
    src = description.strip()
    text, horizon, barrier, warns = _extract_globals(_norm(src), interval)

    rules: list[Rule] = []
    last_lhs: Expr | None = None        # 最近一次认出的主语，供省略时回指
    for chunk in re.split(_SPLIT, text):
        chunk = chunk.strip()
        if not chunk:
            continue

        hits = [(m.start(), m.end(), val, lab)
                for pat, val, lab in _ACTIONS
                for m in re.finditer(pat, chunk)]
        if not hits:
            raise CannotParse(chunk, "这一段里没说要做多、做空还是观望",
                              "每条规则都要有动作，例如「RSI 超过 70 做空」")
        if len({h[2] for h in hits}) > 1:
            raise CannotParse(chunk,
                              f"这一段里同时出现了 {len({h[3] for h in hits})} 个动作"
                              f"（{'、'.join(sorted({h[3] for h in hits}))}）",
                              "一条规则只能有一个动作，用顿号分成两条")
        st, en, action, alabel = hits[0]
        cond_text = (chunk[:st] + " " + chunk[en:]).strip()
        if not _strip_noise(cond_text):
            raise CannotParse(chunk, f"只说了「{alabel}」，没说在什么条件下",
                              "例如「RSI 超过 70 做空」")

        conds = []
        for part in re.split(_JOIN, cond_text):
            if not part.strip():
                continue
            cnd, notes = _parse_cond(part, interval, last_lhs)
            last_lhs = cnd.lhs          # 供后面省略主语的条件回指
            conds.append(cnd)
            warns.extend(notes)
        rules.append(Rule(conds, action, alabel, chunk))

    if not rules:
        raise CannotParse(src, "一条规则都没解析出来")

    # 去重但保持顺序 —— 同一条歧义提示说三遍没有意义
    seen, uniq = set(), []
    for w in warns:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return Spec(rules=rules, source=src, interval=interval,
                horizon=horizon, barrier_mult=barrier, warnings=uniq)


# ────────────────────────────────────────────────────────────
# 词汇表文档 —— 工具描述、报错提示、给人看的说明，都用这一份
# ────────────────────────────────────────────────────────────
VOCAB_DOC = """Truth Serum 自然语言策略 —— 词汇表

这一层是【确定性解析】，不是让模型猜。只认下面这些说法，
认不出的它会明确告诉你哪几个字没看懂，而不是编一段代码给你。

指标（比较式的左边或右边）
  RSI / RSI14 / RSI(14)          相对强弱，默认 14
  EMA12 / EMA(12)                指数均线，周期必须写
  MA20 / SMA20 / 20根均线        简单均线
  ATR / ATR14                    真实波幅
  布林位置 / bb20                 价格在布林带里的位置，0=下轨 1=上轨
  收盘价 / 价格 / close
  成交量 / volume
  24小时涨幅 / 12根涨跌幅          过去 N 根的涨跌幅（百分比量，右边要写 %）
  24小时跌幅                      同上但取反 ——「跌幅超过 3%」= 跌了超过 3%
                                  ⚠ 时间跨度必须每条都写，上一条的不会带过来

比较词
  超过 / 大于 / 高于 / >          这一根就满足即可
  低于 / 小于 / <
  不低于 / ≥        不高于 / ≤
  上穿 / 突破                     只在穿越那一根触发（上一根在另一侧）
  下穿 / 跌破

动作
  做多 / 买入 / 开多              做空 / 卖出 / 开空              观望 / 空仓

连接
  多个条件：并且 / 且 / 和         多条规则：顿号、逗号、分号、换行
  规则按写的顺序匹配，先命中的生效；都不命中则观望

全局设定（写在任意位置）
  持 12 小时 / 持仓 20 根         时间屏障（会按当前 K 线周期换算成根数）
  止盈止损 2 倍 ATR               屏障宽度

例子
  RSI 超过 70 做空、低于 30 做多、持 12 小时
  EMA12 上穿 EMA48 做多，EMA12 下穿 EMA48 做空，止盈止损 2 倍 ATR
  24小时涨幅超过 3% 并且 收盘价高于 MA20 做多，持 6 小时

超出词汇表怎么办
  让你的 Claude 直接写 `def signal(bars):` 代码，走 audit_strategy。
  那条路上代码是模型写的、你看得见、你自己判断 ——
  而不是这一层假装看懂了你的话。"""

VOCAB_HINT = "完整词汇表见 strategy_vocabulary 工具（或 truthserum.nl.VOCAB_DOC）"
