# Truth Serum

**English** | [中文](README.zh-CN.md)

**A backtest auditor that refuses to lie to you.**

> Binance Agent OS Mini Hackathon · Track A

Every trading-agent demo shows you a beautiful equity curve.
This project does the opposite: **it tells you how much of that curve is real.**

```
You say:   "Long when RSI > 60, short when RSI < 40, hold 12 hours"

The number you'd advertise    SOLUSDT, +62.2% annualized (541 trades, 53.4% win rate)
The number in your account    −24.98% mean per fold, −90.4% compounded
```

Both numbers are real. Same Binance data, same strategy code.
The gap is not mysterious — it decomposes into two things you can compute.
See section 4.

---

## 30 seconds

```bash
pip install -r requirements.txt
python examples/demo.py
```

Four acts: one plain-English sentence → the pretty number most people would get →
five gates → exactly where the gap comes from.

**Not a single fabricated number in the whole thing.** The pretty number is
produced by a deliberately-wrong backtester (`truthserum/naive.py`) running on
the same real Binance data as the audit.

---

## Install it into your own Claude

```bash
pip install -e .
claude mcp add truth-serum -- python -m truthserum.server
```

> **Don't skip `pip install -e .`.** Without it, `python -m truthserum.server`
> only resolves when your current directory happens to be this repo. And an MCP
> server that fails to start **usually doesn't report an error** — it just
> silently doesn't appear in the tool list. That's the worst kind of failure.
>
> On Windows, if `python` points at the Microsoft Store stub, use the absolute
> path to your Python. **Start a new session afterwards** — MCP tools are only
> loaded at session start.

Then just say:

> Audit this strategy with truth-serum: short when RSI > 70, long when RSI < 30, hold 12 hours

Six tools are exposed:

| Tool | What it does |
|---|---|
| `list_gates` | What each of the five gates catches |
| `strategy_vocabulary` | Which phrasings the plain-language layer understands |
| `audit_plain_language` | **one sentence → code → five gates** |
| `audit_strategy` | Feed a `signal(bars)` function directly |
| `fetch_market_data` | Pull Binance klines into the local cache |
| `save_mcp_reference` | Accept klines from the **official Binance MCP** as a trusted reference |

That last one is **two MCP servers cooperating**: your Claude calls
`binance-mcp-server` for klines, hands the raw JSON to Truth Serum, which stores
it as a reference and from then on verifies the local market data bar by bar.

A tool that preaches "don't trust blindly" had better not trust its own data feed.

---

## The plain-language layer refuses rather than guesses

**No LLM call. No external code execution.** A fixed vocabulary plus a
deterministic parser; the generated code can only use operators from that
vocabulary.

The first thing it does is **read its interpretation back to you**:

```
Here's how I understood it (please check):
  1. 24h return > 3% AND close > MA(20)   → long
  2. close crosses below MA(20)           → flat

⚠ I made these choices for you — verify them:
  · "24 hours" = 24 bars on a 1h timeframe
  · "below MA20" had no subject; I carried over "close" from the previous rule
  · "crosses below" = triggers only on the crossing bar.
    If you meant "whenever it is below", say "below" instead.
```

If the parse contains an ambiguity and you haven't confirmed it, **it will not
run**. A misread that nobody notices produces a report that audits a *different*
strategy than the one you meant — precisely the failure mode this tool exists to
eliminate.

Anything outside the vocabulary is rejected, naming the exact words it did not
understand:

```
> buy on MACD golden cross
Could not parse "MACD golden cross": that quantity is not in the vocabulary
```

**A translator that guesses your words into code is worse than no translator.**
For anything beyond the vocabulary, the intended path is to have the model write
`signal(bars)` directly and pass it to `audit_strategy` — there the code is
visible and you judge it yourself.

---

## Five gates

Checked from the most basic premise upward. If a premise is wrong, every number
above it is meaningless.

| Gate | What it catches |
|---|---|
| **⓪ Data provenance** | Whether your klines are real at all (bar-by-bar vs the official MCP) |
| **① Lookahead** | Future data changing past signals |
| **② Overlap counting** | "Average expectancy per trade" inflated by how long a signal persists |
| **③ Null baseline** | "I screened many, and the best one looks good" |
| **④ Portfolio simulation** | What's actually left in the account — the only gate entitled to a verdict |

### Three real cases (`python examples/make_report.py`)

Run on 4 symbols, 9000 hourly bars each:

```
⓪①②③④
✅❌✅✅✅   Peeks at tomorrow's close      claims SOLUSDT +3682.5% annualized
✅✅✅❌❌   Three-factor combo, tuned in-sample   claims BTCUSDT +51.0%
✅✅❌❌❌   An honest RSI momentum strategy       claims SOLUSDT +62.2%
```

**The first row is the point of the whole project.** For that blatantly cheating
strategy, **four of the five gates say it's fine**:

- ③ Null baseline: **significantly better than random (p ≈ 0.012)**
- ④ Portfolio: **profitable and significant — +71.75% mean per fold, t = +3.95**

Because **cheating genuinely does make money.** Only ① catches it.

That is why lookahead has to be checked first, and why one gate is never enough.

**The third row is the most sobering.** A completely reasonable, non-cheating
textbook strategy still fails ②③④: expectancy shrinks 78% once overlap is
removed, it is indistinguishable from random (p ≈ 0.358), and the account ends
at −24.98%.

---

## Where the gap comes from: two computable sins

`truthserum/naive.py` faithfully implements **the backtester most people write
themselves**. Its numbers are real; its algorithm is wrong.

| | SOLUSDT — same data, same strategy code |
|---|---|
| Costs ignored | total **+64.4%**, annualized **+62.2%** |
| Costs applied | total **−41.5%**, annualized **−40.6%** |

1. **Fees and slippage ignored.** 0.1908% round trip × 541 trades ≈ **103% of
   capital** in cumulative cost, against a gross profit of only 64.4%. That one
   item alone wipes it out.
2. **Only the best symbol is reported.** Same strategy across 4 symbols — BNB is
   **−26.5%** even *before* costs.

---

## The core design: every detector must first prove it isn't blind

> **A detector that always reports "all clear" is more dangerous than no
> detector at all.** It lets you keep investing in a wrong conclusion with
> confidence.

So `Audit.report()` **refuses to return any conclusion** until its self-check
passes:

```
⛔  ① Lookahead
    Self-check failed — no conclusion will be given
    ├ It should have caught a deliberately injected bug, and did not
    ├ Until the detector itself is fixed, its "clean" verdict means nothing.
```

How each gate proves itself:

| Gate | Self-check |
|---|---|
| ⓪ | Tamper with the reference by 0.01% — must be detected |
| ① | Wrap the strategy in a "peek 1 bar ahead" shell — must be caught |
| ② | Feed a constant signal (maximum overlap) — dedup must cut trades to ≈ 1/horizon |
| ③ | Inject a strategy that literally reads the future — it must beat its own baseline |
| ④ | A coin-flip strategy's expectancy must ≈ −cost (an exactly verifiable invariant) |

**This is not a design-doc platitude. While building this project, gate ① was
wrong three times in a row, and every wrong version reported "no lookahead
found" on real data. All three were stopped by the self-check:**

1. Truncating by *label* time — but the leak lived in the row's *content*, so
   truncating by label couldn't remove it
2. Cutting the kline window at the split point — but the last 12 rows get dropped
   by `dropna`, and the leak fell exactly inside them
3. Intersecting the returned index directly — that's a row number, not a
   timestamp, so everything shifted

Every gate prints its own self-check evidence in the report.
**"No anomaly found" carries weight only because that gate first proved, on a
deliberately injected bug, that it can find one.**

---

## Why this exists

On 2026-09-01, the author's automated trading system reported, in walk-forward
validation: **44 of 44 folds profitable, SOL compounded +2286%, 68% average win
rate.**

That same afternoon: the kline features and the derivatives data were on two
timestamp bases 8 hours apart, which silently cancelled an anti-lookahead buffer
in the code — **the backtest could read 4 hours into the future.**

| | Leaking | Fixed |
|---|---|---|
| Mean AUC over 44 folds | **0.6765** | **0.5017** |
| Profitable folds | 44/44 | 5/44 |
| SOL compounded | +2286% | −20.2% |

**AUC 0.5017 is indistinguishable from a coin flip.** Two months of tuning,
ensembling and cost calibration — all void.

Worse: it was the second time. Two weeks earlier, a 297-hour lookahead had been
found the same way. Both were dug out by hand, each followed by one more
targeted test, and then you wait for the next one.

**Truth Serum is the thing that means you don't have to wait for the next one.**

---

## Data provenance

Market data comes from **Binance's public REST endpoints**, cached locally as
parquet before auditing — an audit shouldn't die halfway because of exchange
rate limiting or a 5xx.

**The official Binance MCP Server's role is verification, not ingestion.**
Gate ⓪ compares its klines against the local cache bar by bar, catching mirror
drift, spot/perp mix-ups, and fake wicks from testnet feeds. Every report prints
its data provenance (source, range, fetch time), so conclusions are traceable.

Current reference coverage: **4/4 symbols, 91 bars**, all matching exactly.

**This is spot verification, not full verification** — and the report says so
out loud: 91 bars out of 36,000 = 0.25%. A matching sample does not prove a
matching whole.

Also, partial coverage **must not display "no anomaly"**: ticking the box after
checking 1 of 4 symbols would let one symbol's clean record vouch for the other
three. That case is reported as *not checked*. Same when no reference exists at
all — **nothing to check ≠ the detector is broken.**

---

## Repository layout

```
truthserum/
├── audit.py          Self-check framework: Audit base / Verdict / TruthReport
├── core.py           Strategy interface / cost model / barrier outcomes / folds
├── data.py           Public REST + MCP reference, one parquet cache
├── indicators.py     Toolbox for strategy code (single source, all causal)
├── nl.py             Plain language → code (deterministic; refuses, never guesses)
├── naive.py          The DELIBERATELY WRONG backtester — the thing being debunked
├── report_html.py    Single-file HTML report, zero external dependencies
├── runner.py         check() entry point
├── server.py         MCP server (six tools)
└── audits/
    ├── provenance.py ⓪ Data provenance
    ├── lookahead.py  ① Lookahead
    ├── overlap.py    ② Overlap counting
    ├── nulltest.py   ③ Null baseline
    └── portfolio.py  ④ Portfolio simulation
examples/
├── demo.py           The four-act demo ← start here
├── make_report.py    Generate the three HTML reports
├── smoke.py          Synthetic-data smoke test, no network
├── real_binance.py   The three cases on real data (terminal version)
├── provenance_demo.py Gate ⓪: two MCP servers cross-checking
├── test_nl.py        Plain-language layer, 32 assertions (10 are "must refuse")
└── test_server.py    MCP tools + real protocol handshake, 31 assertions
```

Verified on Python 3.11.5 / pandas 3.0.3 / numpy 2.4.4 / pyarrow 24.0.0 / mcp 1.27.1.

## Run it yourself

```bash
python examples/smoke.py        # synthetic data, seconds, offline
python examples/test_nl.py      # plain-language layer, seconds, offline
python examples/demo.py         # the full four acts (first run fetches data)
python examples/make_report.py  # generate the three HTML reports
python examples/test_server.py  # MCP tools + protocol handshake
```

## License

MIT
