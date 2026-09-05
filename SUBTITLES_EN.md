# 视频英文字幕稿 / English subtitle script

> ⚠ **这份稿子已过时（2026-09-06）。**
> 写它的时候只有五道闸门、也还没有 agent。现在是六道，而且多了两幕：
> 离线 agent 的反馈循环、以及【你自己的 Claude 连续试策略被 ⑤ 号抓住】。
> 留在仓库里是为了记录当时的取舍（为什么零依赖、为什么不用 emoji 当图标），
> **念白部分不要照着用**。


> **怎么用**：剪映 →「智能字幕」自动识别中文并生成时间轴 → 逐条把中文替换成
> 下面对应的英文。术语（闸门 / 前瞻 / 重叠计数 / 零假设本底）机翻一定会错，
> 这份稿子已经统一过。

**术语对照**（全片必须一致）

| 中文 | 英文 |
|---|---|
| 闸门 / 关卡 | gate |
| 前瞻 | lookahead |
| 重叠计数 | overlap counting |
| 零假设本底 | null baseline |
| 组合层模拟 | portfolio simulation |
| 自检 | self-check |
| 证伪器 | falsifier |
| 每笔期望 | per-trade expectancy |
| 去重叠 | de-overlapped |

---

## 第 1 段 · 终端

> 我给你看两个数字。

Let me show you two numbers.

> 都是同一个策略、同一份币安数据跑出来的。一个是赚六十二个点，一个是亏二十五个点。

Same strategy, same Binance data. One says plus 62 percent. The other says minus 25 percent.

> 两个都没造假。

Neither one is fake.

> 那个赚六十二的，怎么来的呢。

So where does the plus 62 come from?

> 很简单——手续费没算，然后四个币里面挑了最好的那个。

Simple. Fees were never subtracted, and it picks the best of four symbols.

> 你看币安币这一行，亏二十六，这还是没扣手续费的。

Look at the BNB row: minus 26 percent — and that's *before* fees.

> 手续费一笔千分之一点九，跑了五百四十一笔，加起来比本金还多。而它不扣成本也才赚六十四个点。

Round-trip cost is 0.19 percent, times 541 trades — that's more than the entire account. And the gross profit was only 64 percent.

> 就这一项，全没了。

That one item wipes it out.

---

## 第 2 段 · Claude

> 你用大白话说个策略，它翻成代码。

You describe a strategy in plain language, and it turns into code.

> 但它翻完不直接跑，先把理解念回来给你听。

But it doesn't run it right away. First it reads its interpretation back to you.

> 你看这句，「低于四十」我没说是谁低于四十，它自己补了 RSI——但它告诉你它补了。

Here — I said "below 40" without saying *what* is below 40. It filled in RSI, and it tells you that it did.

> 有拿不准的地方，你不点头它就不跑。

Wherever it's unsure, it will not run until you confirm.

> 听着有点多此一举。可它要是理解错了、你又没发现，那份报告审的就不是你想的那个策略了。

Sounds like overkill. But if it misreads you and nobody notices, the report is auditing a different strategy than the one you meant.

> 词汇表里没有的，它直接说不会。

Anything outside its vocabulary, it simply refuses.

> 我觉得这点挺重要的。猜一个给你，比说不会糟糕多了。

I think that matters. Guessing is far worse than saying "I don't know."

---

## 第 3 段 · 报告

> 这是完整报告，五道关卡。

This is the full report. Five gates.

> 查数据那道，看你的 K 线本身是不是真的，拿币安官方数据抽样一根一根对。

The data gate checks whether your candles are real — sampled bar by bar against the official Binance feed.

> 查前瞻那道，看有没有偷看未来。

The lookahead gate checks whether the strategy peeked into the future.

> 查重叠那道，看每笔赚的钱是不是被重复算了——这个策略去掉重复之后，缩水了七成八。

The overlap gate checks whether per-trade profit was double counted. De-overlapped, this strategy shrinks by 78 percent.

> 查随机本底那道，把信号打乱重跑跟运气比。p 值零点三五八，比不出来。

The null baseline gate shuffles the signals and compares against luck. P value 0.358 — indistinguishable.

> 查账户那道，亏二十五个点。

The account gate: minus 25 percent.

> 但最说明问题的是这一份。

But this one is the real point.

> 这个策略是我故意让它偷看明天收盘价的，明摆着作弊。

I built this strategy to peek at tomorrow's close. Blatant cheating.

> 结果呢，除了查前瞻那道，其余四道全放行。

And every gate except lookahead let it through.

> 而且不是勉强放行——查随机本底那道说它明显比随机强；查账户那道说它赚钱，t 值三点九五。

Not marginally, either. The null baseline gate says it clearly beats random. The account gate says it's profitable, t equals 3.95.

> 为什么？因为作弊真的能赚钱啊。

Why? Because cheating really does make money.

> 只有查前瞻那道抓到了它。

Only the lookahead gate caught it.

> 所以这东西是证伪器，不是认证器。挂了是强信号，全过是弱信号——只说明这五种已知死法没被抓到，不等于能赚钱。

So this is a falsifier, not a certifier. Failing a gate is strong evidence. Passing all five is weak evidence — it only means these five known failure modes weren't detected. It does not mean the strategy makes money.

> 那它说"没问题"，我凭什么信它？

And when it says "no anomaly" — why should you believe it?

> 每一道关卡在下结论之前，得先在一个我故意埋进去的 bug 上证明自己能抓到。自检不过，它就说"我判断不了"，而不是"一切正常"。

Before any gate returns a verdict, it has to prove — on a bug I deliberately planted — that it can catch one. If that self-check fails, it reports "cannot determine," not "all clear."

> 这不是我吹的。就写这个前瞻检测，我连写错了三版，每一版拿真实数据跑都说"没发现问题"。三次都是被自检拦下来的。

This isn't a boast. Writing the lookahead gate, I got it wrong three times in a row. Every wrong version reported "no lookahead found" on real data. All three were stopped by the self-check.

> 一个永远报平安的检测器，还不如没有。

A detector that always reports "all clear" is worse than no detector at all.

---

## 第 4 段 · 安装

> 两行命令，装到你自己的 Claude 里。

Two commands, and it's installed in your own Claude.

---

## 第 5 段 · 收尾

> 代码都开源。

The code is open source.

> 每个交易机器人都会给你看一条漂亮的曲线。这个东西告诉你，那条曲线里有几分是真的。

Every trading bot shows you a beautiful equity curve. This one tells you how much of that curve is real.
