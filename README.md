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
The gap decomposes into two things you can compute — see *Where the gap comes from*.

---

## The agent it catches is *your* agent

The most important thing this tool does is not audit a strategy. It is this:

**Truth Serum remembers how many strategies you have tried in this session.**

Ask your Claude to find a strategy that passes the gates. It proposes one, a gate
fails, it reads *why*, and proposes another. That is real reasoning — and it is
also, precisely, **"I tried many and this one won."**

```
Attempt 1   ⑤ not checked  (one trial — "best of 1" carries no information)
Attempt 2   p ≈ 0.175
Attempt 3   p ≈ 0.275
Attempt 4   p ≈ 0.550
Attempt 5   p ≈ 0.100      ← a genuinely better result appeared, so p drops
Attempt 6   p ≈ 0.100
```

Every audit report now opens with your session history, and gate ⑤ recomputes
against the **cumulative** number of attempts. Two tools expose it:
`search_history` and `reset_search_history`.

`reset_search_history`'s own documentation says the quiet part out loud: clearing
the log does not make the search go away, it only makes gate ⑤ blind to it. The
ability to clear is itself an interface you can deceive yourself through, so the
warning lives in the tool description.

**There is no scripted reveal.** ⑤ starts computing from the second attempt
(below that, "the best of one random trial" is one random trial — the comparison
carries no information), and whether it flags is decided entirely by the p-value.
No "fires on attempt 5" switch exists, and none will be added.

A tool built to catch "I tried many and picked the best" catches, first of all,
the agent that is using it.

---

## An offline agent, for a reproducible demo

```bash
python examples/agent_demo.py     # ~60 seconds
```

`TunerAgent` is given one goal — maximize per-trade expectancy — and a family of
RSI strategies. It runs the gates each round, reads **which gate failed**, and
changes where it searches next. Round 2 actually fixes gate ②; round 3 breaks it
again. Every trial across every round accumulates into one `SearchLog`.

Its adaptation rules are a hand-written table, not reasoning — which is exactly
why the section above matters more. This one exists so the behaviour is
reproducible offline, without an LLM in the loop.

The agent is completely honest: it never peeks at the future, never touches the
data, and every step is reproducible. It only does one thing — **it keeps the
best of everything it tried.**

```
Round 1    9 trials   best −0.1297% per trade   ②③④ fail  → widen thresholds
Round 2   18 trials   best −0.1094% per trade   ② PASSES, ③④ fail → change period
Round 3   27 trials   best −0.1002% per trade   ②③④ fail again

⑤ Search selection bias  →  ❌  p ≈ 0.800
   random search over the same 27 draws does at least as well, 80% of the time
```

The metric really does improve every round, and in round 2 the agent genuinely
fixes a gate. Then ⑤ points out that the improvement is explainable by having
tried 27 times.

Note the cumulative column. Round by round the agent only ever feels like it
"tried 9 this time" — but selection bias is counted on the **total**.

This is not cheating. It is what single-metric optimization does — to agents and
to humans alike. The difference is that a human tuning parameters usually
doesn't record how many combinations they tried, so this entire layer of bias
disappears from view.

That is why `SearchLog` is mandatory: an agent that hands you only its winner,
and not the losers, has already hidden the most common form of self-deception.

---

## 30 seconds

```bash
pip install -r requirements.txt
python examples/demo.py
```

Four acts: one plain-English sentence → the pretty number most people would get →
the gates → exactly where the gap comes from.

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

Eight tools are exposed:

| Tool | What it does |
|---|---|
| `list_gates` | What each gate catches |
| `strategy_vocabulary` | Which phrasings the plain-language layer understands |
| `audit_plain_language` | **one sentence → code → all gates** |
| `audit_strategy` | Feed a `signal(bars)` function directly |
| `fetch_market_data` | Pull Binance klines into the local cache |
| `save_mcp_reference` | Accept klines from the **official Binance MCP** as a trusted reference |
| `search_history` | How many strategies you have tried this session, and which won |
| `reset_search_history` | Clear it — with a warning that clearing hides the bias, not removes it |

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
strategy than the one you meant.

Anything outside the vocabulary is rejected, naming the exact words it did not
understand:

```
> buy on MACD golden cross
Could not parse "MACD golden cross": that quantity is not in the vocabulary
```

**A translator that guesses your words into code is worse than no translator.**

---

## The gates

Checked from the most basic premise upward. If a premise is wrong, every number
above it is meaningless.

| Gate | What it catches |
|---|---|
| **⓪ Data provenance** | Whether your klines are real at all (bar-by-bar vs the official MCP) |
| **① Lookahead** | Future data changing past signals |
| **② Overlap counting** | "Average expectancy per trade" inflated by how long a signal persists |
| **③ Null baseline** | "I screened many, and the best one looks good" — for *one* strategy |
| **④ Portfolio simulation** | What's actually left in the account — the only gate entitled to a verdict |
| **⑤ Search selection bias** | "I tried 200 combinations and this one won" — the bias lives in the 199 you didn't show |

③ and ⑤ are not the same check. ③ shuffles the signals of a *single* strategy.
⑤ asks whether the *winner of a search* beats what random search of the same size
would produce. An agent that tries a handful of strategies in one session passes ③ each time
and fails ⑤.

⑤ requires a `SearchLog`. Without one it reports **not checked** — never "fine".
Not checked and no problem are different statements.

### Three real cases (`python examples/make_report.py`)

Run on 4 symbols, 9000 hourly bars each:

```
⓪①②③④⑤
✅❌✅✅✅⏭   Peeks at tomorrow's close      claims SOLUSDT +3682.5% annualized
✅✅✅❌❌⏭   Three-factor combo, tuned in-sample   claims BTCUSDT +51.0%
✅✅❌❌❌⏭   An honest RSI momentum strategy       claims SOLUSDT +62.2%
```

**The first row is the point of the whole project.** For that blatantly cheating
strategy, **every gate except lookahead lets it through**:

- ③ Null baseline: **significantly better than random (p ≈ 0.012)**
- ④ Portfolio: **profitable and significant — +71.75% mean per fold, t = +3.95**

Because **cheating genuinely does make money.** Only ① catches it.

That is why lookahead has to be checked first, and why one gate is never enough.

**The third row is the most sobering.** A completely reasonable, non-cheating
textbook strategy still fails ②③④.

---

## Where the gap comes from: two computable sins

`truthserum/naive.py` faithfully implements **the backtester most people write
themselves**. Its numbers are real; its algorithm is wrong.

| | SOLUSDT — same data, same strategy code |
|---|---|
| Costs ignored | total **+64.4%**, annualized **+62.2%** |
| Costs applied | total **−41.5%**, annualized **−40.6%** |

1. **Fees and slippage ignored.** 0.1908% round trip × 541 trades ≈ **103% of
   capital** in cumulative cost, against a gross profit of only 64.4%.
2. **Only the best symbol is reported.** Same strategy across 4 symbols — BNB is
   **−26.5%** even *before* costs.

---

## The core design: every detector must first prove it isn't blind

> **A detector that always reports "all clear" is more dangerous than no
> detector at all.**

`Audit.report()` **refuses to return any conclusion** until its self-check passes:

```
⛔  ⑤ Search selection bias
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
| ④ | A coin-flip strategy's expectancy must ≈ −cost |
| ⑤ | **Both directions**: a pure-noise search must be caught, *and* a genuinely edged strategy must pass |

⑤'s self-check is deliberately two-sided. Catching the noise search alone proves
nothing — a gate that flags *everything* would also pass that test.

**This is not a design-doc platitude.**

Gate ① was wrong three times in a row while being written, and every wrong
version reported "no lookahead found" on real data. All three were stopped by
the self-check.

Gate ⑤'s first implementation was stopped the same way, on 2026-09-05. To make
the null cheap, it sampled a pool of random scores once and then resampled
best-of-N from that pool — but a resampled maximum can never exceed the pool's
maximum, so any real winner above the pool max got p = 0 automatically. The
self-check reported "a pure-noise search was not caught," and the gate returned
**cannot determine** instead of a confident wrong answer.

---

## Known limitations — what this tool cannot do

**This is a falsifier, not a certifier.**
Failing a gate is *strong* evidence: a specific, reproducible failure mechanism
was found. Passing every gate is *weak* evidence: it only means these known
failure modes were not detected. It does **not** mean the strategy makes money.

Specific weaknesses, stated rather than hidden:

- **A self-check is necessary but not sufficient.** It proves a gate catches
  *that one* planted bug. Gate ① proves it catches "peek 1 bar ahead"; it does
  not prove it catches timezone misalignment or cross-timeframe misalignment.
- **⓪ is a spot check.** Current reference coverage is 91 bars against 36,000 —
  about 0.25%. A matching sample does not prove a matching whole. Partial symbol
  coverage is reported as *not checked*, never as "clean".
- **④'s cost self-check is loose.** It accepts a coin-flip expectancy anywhere in
  `(−3×cost, +0.5×cost)`. A cost model wrong by 3× would still pass.
- **③'s self-check uses fewer shuffles than the real run** (5 vs 80). Enough for
  the huge effect of an oracle strategy, weaker than the real test.
- **⑤'s correlation bias is fixed, not listed.** It used to compute best-of-N
  from the nominal trial count, while a real search's parameter sets are highly
  correlated — which systematically overstates selection bias.
  `effective_trials()` now estimates the **effective number of independent
  trials** from the correlation structure of the candidate signals. Measured: 27
  same-family threshold variants come to 1.8 effective trials.
  **The cost is that ⑤ flags much less often** — but if you genuinely made only
  two independent attempts, the selection bias really is small. The remaining
  approximation: signal correlation stands in for score correlation.
- **No funding fee.** `Costs` covers fees and slippage only. A 12-hour perp
  position crosses one or two 8-hour funding settlements.
- **The cost model was measured on OKX; the data is from Binance.** Fees match
  closely; slippage is not portable across venues.
- **The cache never expires** and does not announce that it served stale data.
- **One year of data, one market regime.** Cannot span a bull/bear transition.

And the most honest caveat of all: **for the example strategy, the decisive
number is elementary arithmetic.** De-overlapped gross expectancy is +0.0102%
against a round-trip cost of 0.1908% — the edge is one nineteenth of the cost.
You can check that with a calculator. What the gates add is the *explanation* of
why the original number looked good: overlap counting had counted the same
opportunity 7.4 times over.

---

## Why this exists

Look-ahead leaks don't announce themselves. Two data sources on timestamp bases
a few hours apart, a centred rolling window, a stray `bfill` — any one of them
can quietly let a backtest read the future. What you see is not an error
message; it is a beautiful equity curve.

Finding one by hand takes days. Finding one does not stop the next one: you
patch it, add a test for that exact case, and wait.

Truth Serum turns each known failure mode into a gate that runs every time —
and each gate proves it can catch a planted bug before it is allowed to report
a verdict.

---

## Data provenance

Market data comes from **Binance's public REST endpoints**, cached locally as
parquet before auditing — an audit shouldn't die halfway because of exchange
rate limiting or a 5xx.

**The official Binance MCP Server's role is verification, not ingestion.**
Gate ⓪ compares its klines against the local cache bar by bar, catching mirror
drift, spot/perp mix-ups, and fake wicks from testnet feeds. Every report prints
its data provenance (source, range, fetch time), so conclusions are traceable.

---

## Repository layout

```
truthserum/
├── audit.py          Self-check framework: Audit base / Verdict / TruthReport
├── core.py           Strategy interface / costs / barrier outcomes / folds / SearchLog
├── data.py           Public REST + MCP reference, one parquet cache
├── indicators.py     Toolbox for strategy code (single source, all causal)
├── nl.py             Plain language → code (deterministic; refuses, never guesses)
├── naive.py          The DELIBERATELY WRONG backtester — the thing being debunked
├── report_html.py    Single-file HTML report, zero external dependencies
├── runner.py         check() entry point
├── server.py         MCP server (eight tools)
├── session.py        Session search log — the agent it catches is your agent
├── agents/
│   └── tuner.py      The searching agent — and the auditor's best target
└── audits/
    ├── provenance.py ⓪ Data provenance
    ├── lookahead.py  ① Lookahead
    ├── overlap.py    ② Overlap counting
    ├── nulltest.py   ③ Null baseline
    ├── portfolio.py  ④ Portfolio simulation
    └── search.py     ⑤ Search selection bias
examples/
├── demo.py           The four-act demo ← start here
├── agent_demo.py     The agent optimizing itself into bias
├── make_report.py    Generate the three HTML reports
├── smoke.py          Synthetic-data smoke test, no network
├── real_binance.py   The three cases on real data (terminal version)
├── provenance_demo.py Gate ⓪: two MCP servers cross-checking
├── test_nl.py        Plain-language layer, 32 assertions (10 are "must refuse")
├── test_invariants.py Cross-gate arithmetic invariants
└── test_server.py    MCP tools + real protocol handshake, 31 assertions
```

Verified on Python 3.11.5 / pandas 3.0.3 / numpy 2.4.4 / pyarrow 24.0.0 / mcp 1.27.1.

## Run it yourself

```bash
python examples/smoke.py           # synthetic data, seconds, offline
python examples/test_nl.py         # plain-language layer, seconds, offline
python examples/test_invariants.py # cross-gate invariants, seconds, offline
python examples/demo.py            # the full four acts
python examples/agent_demo.py      # the agent, and gate ⑤ catching it
python examples/make_report.py     # generate the three HTML reports
python examples/test_server.py     # MCP tools + protocol handshake
```

## License

MIT
