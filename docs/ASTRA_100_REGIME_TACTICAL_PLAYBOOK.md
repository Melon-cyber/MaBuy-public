# MaBuy 100-Regime Tactical Alpha & Crisis Response Playbook

**Document:** `docs/ASTRA_100_REGIME_TACTICAL_PLAYBOOK.md`  
**System:** MaBuy V5.2 Titan  
**Author:** Astra, Principal Quantitative Strategist  
**Universe:** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`  
**Decision Timeframes:** `1h`, `4h`  
**Execution Mode:** Maker-first, VIP-0  
**Status:** Research and execution-control specification; all thresholds require asset-specific validation before production deployment.

---

## 1. Purpose and Operating Doctrine

This playbook is a systematic catalog of 100 crypto-market regimes. It is designed to classify market conditions before trade generation and to constrain MaBuy V5.2 Titan behavior under liquidation, macro, volatility, microstructure, and systemic-risk conditions.

The regime engine is not a directional forecasting model by itself. It is a **risk-selection and execution-governance layer**. Its primary responsibilities are:

1. Identify conditions in which historical signals are structurally unreliable.
2. Prevent naive systems from entering during toxic liquidity or forced-flow events.
3. Select among six approved Titan actions:
   - **Veto**
   - **Scale Down 50%**
   - **Take Profit Aggressive**
   - **Widen Stop**
   - **Reverse Position**
   - **Cash Standby**
4. Translate observed market structure into an explicit risk-reward and payoff expectation.
5. Adapt trade construction to maker-first execution, where non-execution and adverse selection are both material risks.

---

## 2. Signal Definitions

All measurements should be computed independently for BTCUSDT, ETHUSDT, and SOLUSDT, with cross-asset confirmation where available.

### 2.1 Signature Notation

Each regime includes:

`H / F / O / A / V / B`

| Symbol | Definition | Interpretation |
|---|---|---|
| **H** | 1h/4h Hurst exponent | `<0.40` anti-persistent; `0.40–0.55` range-like; `>0.55` persistent trend; `>0.75` extreme persistence |
| **F** | Perpetual funding rate, signed bps per 8h | Positive = longs pay shorts; negative = shorts pay longs |
| **O** | Normalized order-flow imbalance | `-1` = aggressive seller dominance; `+1` = aggressive buyer dominance |
| **A** | 24h ATR as percentage of price | Volatility state, asset-normalized |
| **V** | Rolling volume z-score | Relative participation versus a 30-day intraday baseline |
| **B** | Perpetual spot basis in bps | Positive = perp premium; negative = perp discount |

### 2.2 Risk Unit and Payoff Mathematics

Let:

- `R` = initial account risk allocated to the trade after fees and expected maker slippage.
- `p` = probability of reaching the intended profit target.
- `G` = gross gain in units of `R`.
- `L` = gross loss in units of `R`.

The expected payoff is:

\[
E[R] = pG - (1-p)L
\]

The nominal reward-to-risk ratio is:

\[
RR = \frac{G}{L}
\]

For **Scale Down 50%**, the listed expectancy is the expectancy of the underlying setup at full risk; actual account-level expectancy is approximately one-half of the displayed value, subject to nonlinear liquidation and fee effects.

For **Veto** and **Cash Standby**, no new directional risk is authorized. The expected payoff of the proposed new trade is treated as non-positive or statistically unvalidated. The principal benefit is **loss avoidance**, not trade profit.

### 2.3 Execution Constraints

Maker-first VIP-0 execution requires:

- No market order solely to avoid missing a signal.
- Passive orders must be placed outside the estimated adverse-selection zone.
- Cancel or reprice when OFI reverses materially, spread widens, or the quote becomes stale.
- A missed trade is preferable to a filled trade with negative post-fill expectancy.
- During crisis regimes, maker status is not sufficient protection against gap-through execution.
- Stop logic must account for trigger slippage, mark-price behavior, and thin-book liquidation.
- BTC is the primary liquidity reference; ETH and SOL require additional liquidity and correlation haircuts.

---

# Cluster 1 — Liquidation Cascades, Deleveraging Spikes and Squeezes

## Regimes 01–20

| ID and Regime | Macro/Microstructural Signatures | Primary Failure Mode of Naive Systems | MaBuy V5.2 Titan Tactical Action | Mathematical Risk-Reward and Payoff Expectation |
|---|---|---|---|---|
| **#01: Cascade Liquidation Long Squeeze** | **H:** 0.70–0.90; **F:** +25 to +80 bps; **O:** -0.85 to -1.00; **A:** >8%; **V:** +4 to +10; **B:** -80 to -250 bps after forced selling | Treats oversold momentum as a normal dip and repeatedly buys while liquidation flow is still active. | **Veto** all new longs; existing longs are reduced only under pre-approved risk limits. | New-trade `RR = 0:1`; `E[R] ≤ 0`. Avoided loss can exceed `3R–10R` during gap-through conditions. |
| **#02: Vertical Short Squeeze** | **H:** 0.75–0.95; **F:** -25 to -90 bps; **O:** +0.85 to +1.00; **A:** >8%; **V:** +4 to +12; **B:** +100 to +350 bps | Fades an accelerating rally because price appears overbought, then gets stopped repeatedly above obvious highs. | **Take Profit Aggressive** on profitable shorts; do not initiate countertrend shorts. | Existing-short liquidation control: `RR ≈ 1.2:1`; `p=.62`, `G=1.2R`, `L=.8R`; `E[R]=+.46R` if managed aggressively. |
| **#03: Cross-Asset Deleveraging Shock** | **H:** 0.75–0.95 across BTC/ETH/SOL; **F:** converges rapidly toward zero; **O:** <-0.70; **A:** >7% BTC and >12% SOL; **V:** >+5; **B:** broad negative basis | Trades each asset independently and misses that correlation has converged to one under forced liquidation. | **Cash Standby** across the universe until liquidation intensity and basis stabilize. | `RR = 0:1`; `E[R] = 0` for new risk. Risk avoided is correlated portfolio loss rather than single-position loss. |
| **#04: Funding Unwind Flush** | **H:** >0.65; **F:** falls from >+50 to <0 bps/8h; **O:** <-0.70; **A:** rising 30%+ day-over-day; **V:** +3 to +8; **B:** sharply negative | Assumes funding normalization is bullish and buys before leveraged longs have fully exited. | **Scale Down 50%** only after a confirmed reversal in OFI and liquidation volume. | Conditional rebound setup: `RR=2.0:1`; `p=.42`, `G=2R`, `L=1R`; `E[R]=+.26R`, or approximately `+.13R` after half-sizing. |
| **#05: Spot-Led Panic Sale** | **H:** 0.65–0.85; **F:** neutral to negative; **O:** <-0.80 on spot and perp; **A:** >6%; **V:** +4 to +9; **B:** negative but less extreme than perp-led events | Reads the absence of positive funding as evidence that downside is exhausted, ignoring real spot liquidation. | **Veto** long entries; no reversal until spot OFI improves for at least two confirmation bars. | `RR=0:1`; `E[R]≤0`. Spot-led selling can persist for `2–5` bars after the first apparent washout. |
| **#06: Perpetual-Led Flush** | **H:** 0.70–0.90; **F:** positive before event, then sharply negative; **O:** perp <-0.90 while spot is <-0.40; **A:** >7%; **V:** +4; **B:** -150 to -500 bps | Treats a perpetual dislocation as fair value and sells spot into a temporary forced-flow discount. | **Reverse Position** only with spot confirmation and maker fill below the liquidation pocket. | Mean-reversion setup: `RR=2.5:1`; `p=.40`, `G=2.5R`, `L=1R`; `E[R]=+.40R`. |
| **#07: High-Beta SOL Liquidation Cascade** | **H:** >0.75; **F:** +40 to +120 bps before flush; **O:** <-0.95; **A:** >15%; **V:** +6 to +15; **B:** <-300 bps | Applies BTC-derived reversal levels to SOL, underestimating depth loss and liquidation convexity. | **Veto** new SOL positions; BTC/ETH trades receive a separate correlation haircut. | `RR=0:1`; `E[R]≤0`. SOL gap risk can produce `2R–6R` slippage-adjusted losses. |
| **#08: BTC Dominance Flight-to-Liquidity** | **H:** BTC >0.60, alts <0.40; **F:** BTC mildly positive, ETH/SOL negative; **O:** BTC positive or neutral, alts <-0.70; **A:** BTC 4–7%, SOL >10%; **V:** alt volume >+3; **B:** alt negative | Treats BTC strength as a broad crypto buy signal and accumulates underperforming altcoins during a liquidity migration. | **Scale Down 50%** on ETH/SOL; BTC setups may proceed only with independent confirmation. | Alt rebound setup: `RR=1.8:1`; `p=.48`, `G=1.8R`, `L=1R`; `E[R]=+.344R`, or `+.172R` at half-size. |
| **#09: First-Wave Liquidation Rebound** | **H:** falls from >0.80 to 0.45–0.60; **F:** deeply negative; **O:** turns from <-0.90 to >-0.20; **A:** still >8%; **V:** +3 to +6; **B:** negative but narrowing | Enters immediately on the first green candle without verifying that forced flow has stopped. | **Scale Down 50%** until two consecutive bars show improving OFI and narrowing basis. | Rebound: `RR=2.2:1`; `p=.44`, `G=2.2R`, `L=1R`; `E[R]=+.408R`, half-sized `+.204R`. |
| **#10: Dead-Cat Relief Rally** | **H:** 0.60–0.80; **F:** negative but recovering; **O:** +0.60 initially, then weakening; **A:** >7%; **V:** +2 to +5; **B:** remains negative | Interprets a short-covering bounce as a durable trend reversal and buys the top of the relief move. | **Take Profit Aggressive** on countertrend longs; do not widen stops on stale long signals. | Exit expectancy: `RR=1.1:1`; `p=.64`, `G=1.1R`, `L=.7R`; `E[R]=+.406R`. |
| **#11: Long Squeeze Continuation Leg** | **H:** >0.85; **F:** remains >+20 bps despite decline; **O:** <-0.90; **A:** rising; **V:** +5; **B:** negative and widening | Assumes the first liquidation burst is the entire move and averages down into a second forced-selling leg. | **Veto** all dip-buying systems. | `RR=0:1`; `E[R]≤0`; continuation risk is materially higher while funding remains positive. |
| **#12: Short Squeeze Exhaustion** | **H:** >0.80; **F:** <-40 bps; **O:** +0.90, then diverges negatively; **A:** >9%; **V:** +5 to +12; **B:** >+250 bps | Chases the final vertical candle and enters after shorts have already paid the available upside. | **Take Profit Aggressive** on longs; **Veto** fresh breakout entries unless price consolidates. | Profit-protection setup: `RR=1.3:1`; `p=.68`, `G=1.3R`, `L=.8R`; `E[R]=+.628R`. |
| **#13: Negative Basis Dislocation** | **H:** 0.35–0.55; **F:** <-50 bps; **O:** perp <-0.70, spot neutral; **A:** 5–10%; **V:** +3; **B:** <-300 bps | Sells the discounted perpetual as though the discount represents informed spot selling. | **Reverse Position** through a conservative spot/perp mean-reversion structure, subject to venue risk. | `RR=2.8:1`; `p=.38`, `G=2.8R`, `L=1R`; `E[R]=+.444R`. |
| **#14: Positive Basis Blowout** | **H:** >0.70; **F:** >+60 bps; **O:** +0.80; **A:** >8%; **V:** +4; **B:** >+300 bps | Buys a premium market late because price momentum appears strong. | **Take Profit Aggressive** on long exposure; **Veto** new longs. | `RR=1.4:1`; `p=.66`, `G=1.4R`, `L=.8R`; `E[R]=+.652R` for profit-protection exits. |
| **#15: Margin-Waterfall Session Open** | **H:** >0.75; **F:** extreme in direction of prior positioning; **O:** <-0.80 or >+0.80; **A:** >6%; **V:** +5; **B:** jumps >150 bps at regional open | Places resting orders at prior-session levels without accounting for opening liquidation and spread expansion. | **Cash Standby** for the first two 1h bars after the dislocation. | `RR=0:1`; `E[R]=0` for new trades; avoids high probability of passive-fill adverse selection. |
| **#16: Liquidation-Engine Gap Through Support** | **H:** >0.80; **F:** positive before event; **O:** <-0.95; **A:** >10%; **V:** >+7; **B:** negative by >400 bps | Believes the displayed order book contains reliable support during forced liquidation. | **Veto**; remove resting bids inside the liquidation corridor. | `RR=0:1`; `E[R]≤0`; visible-book support has low reliability under engine-driven execution. |
| **#17: Options-Gamma Liquidation Event** | **H:** transitions from <0.45 to >0.70; **F:** moderate; **O:** accelerates with price; **A:** >8%; **V:** +4; **B:** basis moves with spot rather than funding | Uses ordinary trend stops in a market where dealer hedging creates convex acceleration. | **Widen Stop** only while reducing position size to preserve fixed account risk. | `RR=2.0:1`; `p=.43`, `G=2R`, `L=1R`; `E[R]=+.29R`; stop distance increases but not total dollar risk. |
| **#18: Stablecoin-Collateral Deleveraging** | **H:** >0.70; **F:** negative; **O:** <-0.80; **A:** >9%; **V:** +5; **B:** broad negative basis with stablecoin pairs dislocated | Treats all quoted prices as equally reliable despite collateral impairment and conversion stress. | **Cash Standby** across BTC, ETH, and SOL until stablecoin spreads normalize. | `RR=0:1`; `E[R]=0`; model risk, not directional risk, dominates. |
| **#19: Crowded Basis-Arbitrage Unwind** | **H:** 0.60–0.85; **F:** moves rapidly from stable positive to negative; **O:** mixed but price declines; **A:** 6–11%; **V:** +4; **B:** collapses >200 bps | Assumes basis trades are market-neutral and ignores synchronized collateral calls and legging risk. | **Veto** new arbitrage and directional trades; existing positions are de-risked by venue. | `RR=0:1`; `E[R]≤0`; nominally hedged books can incur multi-leg execution loss. |
| **#20: Post-Cascade Trend Transition** | **H:** rises from 0.40–0.50 to >0.60; **F:** normalized; **O:** sustained sign for 3+ bars; **A:** declining from peak; **V:** +1 to +3; **B:** narrowing toward neutral | Remains in mean-reversion mode after forced flow has ended and misses the new directional trend. | **Widen Stop** on confirmed trend entries, with reduced notional and trailing profit protection. | `RR=2.6:1`; `p=.46`, `G=2.6R`, `L=1R`; `E[R]=+.656R`. |

---

# Cluster 2 — Macroeconomic, Geopolitical and Central-Bank Shocks

## Regimes 21–40

| ID and Regime | Macro/Microstructural Signatures | Primary Failure Mode of Naive Systems | MaBuy V5.2 Titan Tactical Action | Mathematical Risk-Reward and Payoff Expectation |
|---|---|---|---|---|
| **#21: Pre-FOMC Positioning Compression** | **H:** 0.40–0.52; **F:** ±10 bps; **O:** near zero; **A:** 2–4%; **V:** <-1; **B:** flat to +40 bps | Trades low-volatility technical patterns immediately before a binary macro event. | **Cash Standby** from the final scheduled pre-event window until post-release spreads normalize. | `RR=0:1`; `E[R]=0`; avoided gap risk is the primary alpha. |
| **#22: Hawkish FOMC Rate-Hike Shock** | **H:** >0.70; **F:** moves lower or negative; **O:** <-0.80; **A:** >7%; **V:** +4; **B:** negative | Buys the first dip or assumes the hike was already priced without measuring guidance and dollar/yield reaction. | **Veto** long entries; profitable shorts take profit at predefined liquidity levels. | New long `RR=0:1`; `E[R]≤0`. Macro repricing can extend across several 4h bars. |
| **#23: Dovish FOMC Liquidity Rally** | **H:** >0.65; **F:** rises toward positive; **O:** +0.80; **A:** >6%; **V:** +4; **B:** +100 to +250 bps | Fades the rally because price is overbought or enters shorts into dealer and short-covering flows. | **Take Profit Aggressive** on existing shorts; longs require no immediate reversal unless basis remains orderly. | Long continuation: `RR=1.8:1`; `p=.56`, `G=1.8R`, `L=1R`; `E[R]=+.568R`. |
| **#24: CPI Hot-Surprise Shock** | **H:** >0.70; **F:** declines; **O:** <-0.75; **A:** >7%; **V:** +4 to +8; **B:** negative | Uses pre-CPI positioning or stale historical correlation instead of the actual rate-path repricing. | **Veto** new longs; existing longs are managed with aggressive profit-taking or risk reduction. | `RR=0:1` for new longs; `E[R]≤0`. |
| **#25: CPI Cool-Surprise Melt-Up** | **H:** >0.65; **F:** positive; **O:** +0.75 to +0.95; **A:** >6%; **V:** +4; **B:** +100 bps or more | Shorts the first spike, expecting mean reversion, while liquidity providers widen spreads. | **Take Profit Aggressive** on shorts; do not reverse at the first print. | Managed long continuation: `RR=1.6:1`; `p=.58`, `G=1.6R`, `L=.9R`; `E[R]=+.446R`. |
| **#26: Payrolls/Employment Growth Shock** | **H:** 0.65–0.85; **F:** directionally reprices; **O:** ±0.70; **A:** 5–8%; **V:** +3 to +6; **B:** follows rates | Trades the headline number while ignoring revisions, wages, unemployment, and rates reaction. | **Scale Down 50%** until the second impulse confirms direction. | `RR=2.0:1`; `p=.43`, `G=2R`, `L=1R`; `E[R]=+.29R`, half-sized `+.145R`. |
| **#27: Treasury-Yield Spike** | **H:** >0.65; **F:** usually negative; **O:** <-0.65; **A:** 5–9%; **V:** +3; **B:** negative or flat | Buys crypto on isolated technical oversold readings while real-rate pressure is increasing. | **Veto** long exposure until yields stabilize or crypto decouples with confirmed spot demand. | `RR=0:1`; `E[R]≤0`. |
| **#28: Dollar-Index Liquidity Shock** | **H:** >0.65; **F:** negative; **O:** <-0.70; **A:** >6%; **V:** +3 to +5; **B:** negative | Assumes crypto-specific catalysts can overcome a broad dollar squeeze. | **Scale Down 50%** on directional shorts; **Veto** leveraged longs. | Short continuation: `RR=1.9:1`; `p=.51`, `G=1.9R`, `L=1R`; `E[R]=+.479R`, half-sized `+.240R`. |
| **#29: Central-Bank Emergency Liquidity Injection** | **H:** jumps >0.60; **F:** positive; **O:** +0.80; **A:** >7%; **V:** +5; **B:** positive | Fades a policy backstop as a temporary bounce and remains short during reflexive liquidity expansion. | **Reverse Position** only after confirmation that spot, not only perps, is buying. | `RR=2.3:1`; `p=.47`, `G=2.3R`, `L=1R`; `E[R]=+.621R`. |
| **#30: Geopolitical War-Escalation Shock** | **H:** >0.70; **F:** negative; **O:** <-0.80; **A:** >8%; **V:** +5 to +10; **B:** negative | Treats the initial selloff as a conventional technical pullback and ignores headline gap risk. | **Cash Standby** until headline volatility and cross-asset correlations decline. | `RR=0:1`; `E[R]=0`; event-gap loss avoidance dominates expected directional edge. |
| **#31: Geopolitical De-Escalation Relief Rally** | **H:** >0.60; **F:** normalizes; **O:** +0.75; **A:** >6% but falling; **V:** +4; **B:** basis recovers | Shorts a relief rally because the asset remains below its pre-event level. | **Take Profit Aggressive** on shorts; permit only small long exposure after retest. | `RR=1.7:1`; `p=.55`, `G=1.7R`, `L=1R`; `E[R]=+.485R`. |
| **#32: Spot Bitcoin ETF Approval Anticipation** | **H:** 0.60–0.75; **F:** positive; **O:** +0.60; **A:** 4–7%; **V:** +2 to +4; **B:** +100–250 bps | Buys without distinguishing anticipation premium from sustainable net inflows. | **Scale Down 50%**; take partial profit into approval headlines. | `RR=1.8:1`; `p=.48`, `G=1.8R`, `L=1R`; `E[R]=+.344R`, half-sized `+.172R`. |
| **#33: ETF Approval Sell-the-News Reversal** | **H:** falls from >0.70 to <0.50; **F:** still positive; **O:** turns from +0.80 to <-0.65; **A:** >7%; **V:** +4; **B:** basis compresses | Remains long because the catalyst is objectively bullish and ignores positioning exhaustion. | **Take Profit Aggressive**; **Reverse Position** only after a failed reclaim of the announcement level. | Reversal: `RR=2.0:1`; `p=.44`, `G=2R`, `L=1R`; `E[R]=+.32R`. |
| **#34: ETF Rejection or Delay Shock** | **H:** >0.70; **F:** falls; **O:** <-0.80; **A:** >8%; **V:** +5; **B:** negative | Buys the rejection dip based on long-term thesis while short-term liquidation is unresolved. | **Veto** long entries; existing short exposure takes profits into liquidation pockets. | `RR=0:1` for new longs; `E[R]≤0`. |
| **#35: Regulatory Enforcement Shock** | **H:** >0.70; **F:** negative; **O:** <-0.75; **A:** BTC 5–8%, SOL/ETH higher; **V:** +4; **B:** negative | Assumes the event is token-specific while exchange, custody, and liquidity risks spread across the universe. | **Cash Standby** for affected venues and assets; no cross-venue arbitrage until legal scope is clear. | `RR=0:1`; `E[R]=0`; operational risk dominates price edge. |
| **#36: Favorable Regulatory Clarity Rally** | **H:** >0.60; **F:** positive; **O:** +0.70; **A:** 5–8%; **V:** +3 to +6; **B:** positive but orderly | Fades the move because it appears headline-driven and misses a structural risk-premium repricing. | **Widen Stop** on confirmed long entries while reducing notional to retain fixed dollar risk. | `RR=2.4:1`; `p=.45`, `G=2.4R`, `L=1R`; `E[R]=+.53R`. |
| **#37: China/Asia Liquidity Policy Shock** | **H:** >0.60; **F:** directional; **O:** ±0.65; **A:** 5–9%; **V:** +3; **B:** follows regional risk | Applies US-session assumptions to an Asia-led move and enters during thin transition liquidity. | **Scale Down 50%** until Europe/US sessions confirm. | `RR=1.9:1`; `p=.47`, `G=1.9R`, `L=1R`; `E[R]=+.363R`, half-sized `+.182R`. |
| **#38: Sovereign Debt or Default Rumor** | **H:** >0.75; **F:** negative; **O:** <-0.80; **A:** >8%; **V:** +5; **B:** negative and widening | Treats rumor-driven price action as a tradable range before source verification. | **Cash Standby** until the information state is resolved. | `RR=0:1`; `E[R]=0`; rumor gaps and venue liquidity are unpriceable. |
| **#39: Election or Fiscal-Policy Volatility** | **H:** 0.55–0.80; **F:** ±30 bps; **O:** ±0.70; **A:** 5–10%; **V:** +3 to +6; **B:** unstable | Overfits historical election seasonality and trades a binary policy path with excessive leverage. | **Scale Down 50%**; use wider structural invalidation rather than tight candle stops. | `RR=2.1:1`; `p=.42`, `G=2.1R`, `L=1R`; `E[R]=+.302R`, half-sized `+.151R`. |
| **#40: Post-Shock Macro Trend Repricing** | **H:** >0.60; **F:** stable in new direction; **O:** sustained >±0.60; **A:** declining from event peak; **V:** +1 to +3; **B:** stabilizes at new level | Returns to pre-event mean-reversion rules even though the macro valuation regime has changed. | **Widen Stop** on directionally confirmed positions; trail only after a new 4h structure forms. | `RR=2.7:1`; `p=.44`, `G=2.7R`, `L=1R`; `E[R]=+.496R`. |

---

# Cluster 3 — Low-Volatility Compression, Chop and Volatility Breakouts

## Regimes 41–60

| ID and Regime | Macro/Microstructural Signatures | Primary Failure Mode of Naive Systems | MaBuy V5.2 Titan Tactical Action | Mathematical Risk-Reward and Payoff Expectation |
|---|---|---|---|---|
| **#41: Volatility Compression Box** | **H:** 0.45–0.55; **F:** -10 to +10 bps; **O:** -0.20 to +0.20; **A:** <3%; **V:** <-1; **B:** 0–30 bps | Assumes the next small breakout will trend and repeatedly pays fees inside the box. | **Cash Standby** until range expansion is confirmed. | `RR=0:1`; `E[R]≤0` after fees and maker non-execution costs. |
| **#42: Low-Volume Mean-Reversion Chop** | **H:** <0.42; **F:** neutral; **O:** oscillatory ±0.40; **A:** 2–4%; **V:** <-0.5; **B:** flat | Uses trend-following entries in an anti-persistent environment and gets whipsawed. | **Scale Down 50%** on range-edge reversals only. | `RR=1.5:1`; `p=.56`, `G=1.5R`, `L=1R`; `E[R]=+.40R`, half-sized `+.20R`. |
| **#43: False Upside Compression Breakout** | **H:** rises briefly >0.60 then falls; **F:** +20–50 bps; **O:** +0.75 then negative; **A:** >5%; **V:** +2 then fades; **B:** widens positive then compresses | Chases the breakout after the initial impulse and is trapped by a failed auction. | **Reverse Position** after a confirmed close back inside the prior range. | `RR=2.2:1`; `p=.45`, `G=2.2R`, `L=1R`; `E[R]=+.44R`. |
| **#44: False Downside Compression Breakdown** | **H:** briefly >0.60 then <0.45; **F:** -20 to -50 bps; **O:** <-0.75 then positive; **A:** >5%; **V:** +2 then fades; **B:** negative then recovers | Shorts the breakdown after passive bids have absorbed the initial sell impulse. | **Reverse Position** after reclaim of the range boundary with positive spot OFI. | `RR=2.1:1`; `p=.46`, `G=2.1R`, `L=1R`; `E[R]=+.426R`. |
| **#45: Volatility Expansion from Squeeze** | **H:** moves from 0.45 to >0.65; **F:** initially neutral; **O:** >±0.70; **A:** doubles from base; **V:** +3; **B:** expands in breakout direction | Continues selling premium-like range reversals after the market has transitioned to directional expansion. | **Widen Stop** on confirmed breakout entries, with reduced size. | `RR=2.8:1`; `p=.43`, `G=2.8R`, `L=1R`; `E[R]=+.634R`. |
| **#46: Bollinger/Keltner Squeeze with Directional Bias** | **H:** 0.55–0.70; **F:** mildly directional; **O:** ±0.55; **A:** <3.5% then rising; **V:** <-0.5 before expansion; **B:** modest directional premium | Treats compression as directionless despite persistent positioning and asymmetric order flow. | **Scale Down 50%** until breakout volume exceeds `z=2`. | `RR=2.4:1`; `p=.44`, `G=2.4R`, `L=1R`; `E[R]=+.496R`, half-sized `+.248R`. |
| **#47: Range-Edge Liquidity Harvest** | **H:** 0.35–0.48; **F:** neutral; **O:** reverses at edges; **A:** 2–4%; **V:** normal; **B:** flat | Enters in the middle of the range and has no favorable location. | **Scale Down 50%** and permit only maker orders at predefined range extremes. | `RR=1.7:1`; `p=.58`, `G=1.7R`, `L=.9R`; `E[R]=+.796R`, half-sized `+.398R`. |
| **#48: Funding-Carry Compression** | **H:** 0.42–0.52; **F:** persistently +15 to +30 or -15 to -30 bps; **O:** near zero; **A:** <3%; **V:** <-1; **B:** aligned with funding | Focuses only on price and ignores that carry is the dominant return component. | **Take Profit Aggressive** when basis compresses; avoid directional leverage. | Carry realization: `RR=1.1:1`; `p=.70`, `G=1.1R`, `L=.6R`; `E[R]=+.65R`. |
| **#49: Alternating Funding Chop** | **H:** <0.40; **F:** flips sign every 1–3 bars; **O:** alternates ±0.50; **A:** 3–5%; **V:** normal; **B:** oscillates around zero | Uses funding as a stable positioning signal when it is only reflecting short-term noise. | **Veto** funding-driven directional trades. | `RR=0:1`; `E[R]≤0` after turnover and queue costs. |
| **#50: Low-Volatility Correlation Breakdown** | **H:** BTC 0.50, ETH/SOL <0.40 or inverse; **F:** mixed; **O:** asset-specific; **A:** BTC <3%, alts 4–7%; **V:** dispersed; **B:** cross-asset divergence | Applies BTC direction to all assets and mistakes beta dispersion for confirmation. | **Scale Down 50%** on ETH/SOL; require local spot-flow evidence. | `RR=1.8:1`; `p=.50`, `G=1.8R`, `L=1R`; `E[R]=+.40R`, half-sized `+.20R`. |
| **#51: Options-Expiry Pinning Range** | **H:** 0.35–0.48; **F:** neutral; **O:** mean-reverting; **A:** <3%; **V:** <-0.5; **B:** flat | Attempts to trend trade through a dealer-pinned strike and repeatedly exits near breakeven after fees. | **Cash Standby** for breakout systems; range systems require half-size. | Breakout `RR=0:1`; `E[R]≤0` until pin releases. |
| **#52: Implied-Realized Volatility Divergence** | **H:** 0.45–0.60; **F:** moderate; **O:** mixed; **A:** realized <3%, implied elevated; **V:** normal; **B:** stable | Buys expensive volatility through directional trades without a realized-vol catalyst. | **Take Profit Aggressive** on short-horizon directional exposure; no volatility chasing. | Directional scalp: `RR=1.2:1`; `p=.58`, `G=1.2R`, `L=.8R`; `E[R]=+.336R`. |
| **#53: Volatility Crush After Scheduled Event** | **H:** falls toward 0.45; **F:** normalizes; **O:** decays toward zero; **A:** falls >35%; **V:** initially +3 then below normal; **B:** compresses | Holds breakout positions for a second trend leg after the event premium has been removed. | **Take Profit Aggressive** immediately upon target or momentum decay. | Exit setup: `RR=1.0:1`; `p=.72`, `G=1R`, `L=.6R`; `E[R]=+.552R`. |
| **#54: Compression with Hidden Spot Accumulation** | **H:** 0.55–0.65; **F:** neutral or slightly negative; **O:** spot +0.55, perp near zero; **A:** <4%; **V:** +1; **B:** flat | Concludes that low volatility means no directional information and shorts quiet accumulation. | **Widen Stop** on small long exposure; retain fixed dollar risk. | `RR=2.6:1`; `p=.43`, `G=2.6R`, `L=1R`; `E[R]=+.548R`. |
| **#55: Compression with Hidden Distribution** | **H:** 0.45–0.55; **F:** slightly positive; **O:** spot -0.55, perp neutral; **A:** <4%; **V:** +1; **B:** modest positive | Buys a stable range while larger holders distribute into passive demand. | **Scale Down 50%** on longs; require a positive spot-flow reversal. | `RR=1.9:1`; `p=.44`, `G=1.9R`, `L=1R`; `E[R]=+.276R`, half-sized `+.138R`. |
| **#56: Whipsaw Breakout Sequence** | **H:** <0.40; **F:** alternates; **O:** repeated ±0.75; **A:** 5–7%; **V:** +2 on each failed move; **B:** expands and collapses | Treats every range break as independent and compounds losses through repeated entries. | **Veto** breakout systems for the next defined lookback window. | `RR=0:1`; `E[R]≤0`; serial correlation of failed signals is strongly adverse. |
| **#57: Breakout Retest Acceptance** | **H:** >0.60; **F:** directionally aligned; **O:** impulse ±0.80, retest ±0.30; **A:** 5–8%; **V:** +2 to +4; **B:** holds directional | Enters the first impulse candle and receives poor price, or treats a successful retest as a failed breakout. | **Widen Stop** on retest entries, placing invalidation beyond the reclaimed level. | `RR=2.5:1`; `p=.48`, `G=2.5R`, `L=1R`; `E[R]=+.68R`. |
| **#58: Trend-Day Initiation after Multi-Day Compression** | **H:** rises >0.70; **F:** moves with trend; **O:** >0.80; **A:** >6%; **V:** +4; **B:** expands orderly | Fades an unusually persistent day based on mean-reversion priors. | **Widen Stop** with smaller position size and no countertrend entries. | `RR=2.9:1`; `p=.45`, `G=2.9R`, `L=1R`; `E[R]=+.755R`. |
| **#59: Failed Breakout Retest** | **H:** declines from >0.60 to <0.45; **F:** remains stale; **O:** reverses >0.70; **A:** 5–8%; **V:** +2 then declines; **B:** returns toward zero | Re-enters in the original breakout direction after the first failure. | **Reverse Position** after retest rejection and confirmed OFI reversal. | `RR=2.3:1`; `p=.46`, `G=2.3R`, `L=1R`; `E[R]=+.518R`. |
| **#60: Volatility Regime Reversion** | **H:** returns toward 0.50; **F:** neutralizes; **O:** decays; **A:** falls toward baseline; **V:** normalizes; **B:** compresses | Keeps crisis-sized stops and trend assumptions after volatility has returned to normal. | **Take Profit Aggressive** on residual event trades; restore standard risk parameters. | Mean-reversion exit: `RR=1.4:1`; `p=.62`, `G=1.4R`, `L=.8R`; `E[R]=+.612R`. |

---

# Cluster 4 — Microstructure OFI, CME Gaps and Weekend Illiquidity

## Regimes 61–80

| ID and Regime | Macro/Microstructural Signatures | Primary Failure Mode of Naive Systems | MaBuy V5.2 Titan Tactical Action | Mathematical Risk-Reward and Payoff Expectation |
|---|---|---|---|---|
| **#61: Persistent Bid-Side OFI Accumulation** | **H:** 0.55–0.70; **F:** neutral to +15 bps; **O:** +0.70 for 6+ bars; **A:** 3–5%; **V:** +1 to +3; **B:** positive but <100 bps | Shorts into repeated absorption because price has not yet broken higher. | **Widen Stop** on small longs; do not fade persistent spot demand. | `RR=2.2:1`; `p=.50`, `G=2.2R`, `L=1R`; `E[R]=+.60R`. |
| **#62: Persistent Ask-Side OFI Distribution** | **H:** 0.55–0.70; **F:** neutral to -15 bps; **O:** <-0.70 for 6+ bars; **A:** 3–6%; **V:** +1 to +3; **B:** negative or flat | Buys minor dips while passive offers repeatedly absorb demand. | **Veto** new longs; short setups require confirmation because passive support may still exist. | `RR=0:1` for new longs; `E[R]≤0`. |
| **#63: Spoofing-Like OFI Reversal** | **H:** <0.45; **F:** neutral; **O:** flips ±0.90 within 1–2 bars; **A:** 5–8%; **V:** +2; **B:** unstable | Treats displayed depth as genuine and enters in the direction of cancel-and-replace liquidity. | **Cash Standby** until executed volume, not displayed size, confirms direction. | `RR=0:1`; `E[R]=0`; displayed liquidity has low signal integrity. |
| **#64: Iceberg Absorption at Support** | **H:** 0.40–0.55; **F:** mildly negative; **O:** aggressive selling but price stable; **A:** 4–7%; **V:** +2; **B:** negative but narrowing | Assumes repeated market selling guarantees breakdown and shorts directly into hidden demand. | **Reverse Position** only after a higher low and positive spot OFI. | `RR=2.4:1`; `p=.43`, `G=2.4R`, `L=1R`; `E[R]=+.462R`. |
| **#65: Iceberg Absorption at Resistance** | **H:** 0.40–0.55; **F:** mildly positive; **O:** aggressive buying but price stable; **A:** 4–7%; **V:** +2; **B:** positive but narrowing | Buys repeated market orders into hidden supply and interprets absorption as bullish strength. | **Take Profit Aggressive** on longs; short only after rejection confirmation. | `RR=1.6:1`; `p=.57`, `G=1.6R`, `L=.8R`; `E[R]=+.752R`. |
| **#66: Maker Adverse-Selection Zone** | **H:** >0.65; **F:** directional; **O:** >±0.75; **A:** >6%; **V:** +3; **B:** expands in same direction | Places passive orders exactly where informed flow is likely to execute against them before continuation. | **Veto** passive entries in the impulse direction; wait for retest or use cash. | `RR=0:1`; `E[R]≤0` after adverse selection. |
| **#67: CME Gap-Up Open** | **H:** 0.55–0.70; **F:** positive; **O:** +0.60; **A:** 4–7%; **V:** +2; **B:** positive | Assumes the gap must fill and shorts a strong open without checking crypto-session continuation. | **Scale Down 50%** on gap-fade trades; prioritize maker orders near validated gap boundaries. | Gap-fade: `RR=1.8:1`; `p=.52`, `G=1.8R`, `L=1R`; `E[R]=+.456R`, half-sized `+.228R`. |
| **#68: CME Gap-Down Open** | **H:** 0.55–0.70; **F:** negative; **O:** -0.60; **A:** 4–8%; **V:** +2; **B:** negative | Buys the gap solely because historical fill frequency is high, ignoring a genuine trend day. | **Scale Down 50%** on gap-fill longs; require rejection of the low and improving OFI. | `RR=1.9:1`; `p=.49`, `G=1.9R`, `L=1R`; `E[R]=+.421R`, half-sized `+.211R`. |
| **#69: CME Gap-Fill Acceptance** | **H:** 0.45–0.60; **F:** neutralizes; **O:** reverses toward zero; **A:** 3–5%; **V:** +1; **B:** compresses | Continues trading the pre-gap direction after price has returned to value. | **Take Profit Aggressive** on gap-related positions. | `RR=1.3:1`; `p=.65`, `G=1.3R`, `L=.7R`; `E[R]=+.65R`. |
| **#70: Weekend Thin-Book Bull Trap** | **H:** >0.65; **F:** positive; **O:** +0.80 but low quality; **A:** >6%; **V:** +1 despite price move; **B:** +150 bps | Treats low-volume weekend upside as institutional accumulation. | **Veto** new longs; take profits on existing longs near liquidity extensions. | `RR=0:1`; `E[R]≤0`. |
| **#71: Weekend Thin-Book Bear Trap** | **H:** >0.65; **F:** negative; **O:** -0.80 but low quality; **A:** >6%; **V:** +1; **B:** -150 bps | Shorts a thin-book selloff that is vulnerable to a liquidity vacuum reversal. | **Reverse Position** only after a confirmed reclaim and real spot participation. | `RR=2.0:1`; `p=.42`, `G=2R`, `L=1R`; `E[R]=+.26R`. |
| **#72: Sunday Open Liquidity Vacuum** | **H:** unstable; **F:** ±30 bps; **O:** noisy; **A:** >5%; **V:** <-0.5 or isolated spikes; **B:** unstable | Uses normal spread, fill, and stop assumptions in a market with sparse depth. | **Cash Standby** through the designated Sunday illiquidity window. | `RR=0:1`; `E[R]=0`; execution uncertainty dominates model expectancy. |
| **#73: Thin-Book Wick and Stop Hunt** | **H:** <0.40; **F:** neutral; **O:** ±0.95 for one bar; **A:** spike >2x baseline; **V:** +2 without follow-through; **B:** immediate dislocation | Places stops at obvious local extrema and interprets a single wick as a structural breakout. | **Widen Stop** only after reducing notional; invalidate on closing basis, not wick alone. | `RR=2.0:1`; `p=.47`, `G=2R`, `L=1R`; `E[R]=+.41R`. |
| **#74: Cross-Exchange Lead/Lag Divergence** | **H:** 0.50–0.70; **F:** mixed; **O:** leader ±0.80, lagger near zero; **A:** 4–7%; **V:** leader +3; **B:** venue-specific | Trades the lagging venue without verifying whether the leading move is genuine or an isolated print. | **Veto** lag-trading unless two independent venues confirm. | `RR=0:1`; `E[R]≤0`; isolated venue prints create toxic maker fills. |
| **#75: Latency-Driven Price Dislocation** | **H:** appears >0.75; **F:** unstable; **O:** extreme; **A:** instantaneous spike; **V:** abnormal; **B:** venue-specific >200 bps | Assumes a stale quote is executable fair value and posts against an informed latency advantage. | **Cash Standby**; cancel all stale passive orders. | `RR=0:1`; `E[R]=0`; adverse selection is structurally asymmetric. |
| **#76: Liquidation-Feed OFI Amplification** | **H:** >0.80; **F:** extreme; **O:** ±0.95; **A:** >8%; **V:** +5; **B:** expands | Treats liquidation prints as organic demand and enters directly into the feed. | **Veto** until forced-flow rate declines for at least three bars. | `RR=0:1`; `E[R]≤0`. |
| **#77: Basis-Arbitrage Rebalance Window** | **H:** 0.40–0.55; **F:** ±40 bps; **O:** mixed; **A:** 3–6%; **V:** +2; **B:** dislocated ±150–300 bps | Enters without modeling transfer, borrow, funding timestamp, and fill-leg risk. | **Scale Down 50%** and use only pre-hedged, simultaneous maker placements. | `RR=1.6:1`; `p=.55`, `G=1.6R`, `L=.9R`; `E[R]=+.565R`, half-sized `+.283R`. |
| **#78: Funding-Timestamp Positioning Spike** | **H:** 0.45–0.65; **F:** approaches extreme timestamp; **O:** temporary ±0.70; **A:** 4–6%; **V:** +2; **B:** moves with funding | Mistakes predictable funding-related repositioning for a durable breakout. | **Take Profit Aggressive** before funding settlement if target is reached; avoid late entries. | `RR=1.2:1`; `p=.61`, `G=1.2R`, `L=.7R`; `E[R]=+.487R`. |
| **#79: Spread-Widening Taker Burst** | **H:** >0.65; **F:** directional; **O:** ±0.90; **A:** >7%; **V:** +4; **B:** jumps >200 bps | Assumes high volume means high liquidity; actually, taker urgency is consuming the book. | **Veto** maker entries into the burst; re-enter only after spread and OFI normalize. | `RR=0:1`; `E[R]≤0`; high volume is not equivalent to executable depth. |
| **#80: Asia–Europe–US Handoff Imbalance** | **H:** 0.55–0.75; **F:** moderate; **O:** sign changes at session handoff; **A:** 4–7%; **V:** +2; **B:** follows incoming session | Carries a local-session signal into a new liquidity regime with different participants and inventory. | **Scale Down 50%** during handoff; widen confirmation window by one 1h bar. | `RR=1.8:1`; `p=.49`, `G=1.8R`, `L=1R`; `E[R]=+.382R`, half-sized `+.191R`. |

---

# Cluster 5 — Black Swans, De-Pegs, Insolvencies and Fork Cascades

## Regimes 81–100

| ID and Regime | Macro/Microstructural Signatures | Primary Failure Mode of Naive Systems | MaBuy V5.2 Titan Tactical Action | Mathematical Risk-Reward and Payoff Expectation |
|---|---|---|---|---|
| **#81: Stablecoin De-Peg** | **H:** >0.75 in affected pairs; **F:** extreme and unreliable; **O:** <-0.80; **A:** BTC proxy >8%; **V:** +6; **B:** venue-dependent and unstable | Treats the quoted stablecoin price as cash-equivalent and executes apparent arbitrage that cannot settle at par. | **Cash Standby**; suspend all affected-quote and collateral-dependent strategies. | `RR=0:1`; `E[R]=0` for new risk; tail loss can approach total collateral impairment. |
| **#82: Protocol Stable-Asset De-Peg** | **H:** >0.70; **F:** unreliable; **O:** affected ecosystem <-0.90; **A:** broad assets 6–15%; **V:** +5; **B:** negative | Buys the de-pegged asset based on historical mean reversion without solvency analysis. | **Veto** all dip-buying and collateral loops. | `RR=0:1`; `E[R]≤0`; recovery probability is discontinuous and path-dependent. |
| **#83: Major Bridge Exploit Contagion** | **H:** >0.75; **F:** negative in related tokens; **O:** <-0.85; **A:** >10%; **V:** +6 to +12; **B:** negative | Assumes contagion is limited to the exploited bridge and continues trading correlated assets normally. | **Cash Standby** for affected chains, protocols, and wrapped assets. | `RR=0:1`; `E[R]=0`; jump and withdrawal risk exceed modeled stop assumptions. |
| **#84: Exchange Insolvency Rumor** | **H:** >0.65; **F:** venue-specific dislocation; **O:** affected venue <-0.90; **A:** >8%; **V:** +5; **B:** venue basis >300 bps | Arbitrages a discount without pricing withdrawal, custody, and clawback risk. | **Veto** venue-specific trades; move only verified unencumbered collateral under policy. | `RR=0:1`; `E[R]≤0`; apparent discount may be compensation for recovery uncertainty. |
| **#85: Exchange Withdrawal Freeze** | **H:** >0.75; **F:** unreliable; **O:** venue-specific extreme; **A:** broad market 7–12%; **V:** +6; **B:** venue premium/discount unstable | Assumes positions can be closed or collateral transferred when the exchange has suspended withdrawals. | **Cash Standby** and apply venue exposure limits immediately. | `RR=0:1`; `E[R]=0`; liquidity and custody risk dominate market direction. |
| **#86: Chain Halt or Consensus Outage** | **H:** unavailable or non-stationary; **F:** distorted; **O:** fragmented; **A:** >10%; **V:** abnormal; **B:** fragmented across venues | Continues technical trading on an asset whose settlement layer is not finalizing reliably. | **Veto** affected asset; no maker orders until chain finality is independently restored. | `RR=0:1`; `E[R]≤0`; mark prices may remain tradable while settlement is impaired. |
| **#87: Contentious Hard Fork Announcement** | **H:** >0.65; **F:** bifurcates; **O:** ±0.80; **A:** >8%; **V:** +5; **B:** unstable | Assumes the legacy ticker, balances, and replay protection will be handled cleanly. | **Cash Standby** through the fork and post-fork asset-crediting period. | `RR=0:1`; `E[R]=0`; operational ambiguity invalidates ordinary payoff estimates. |
| **#88: Fork Execution and Replay Risk** | **H:** non-stationary; **F:** unreliable; **O:** fragmented; **A:** >12%; **V:** +7; **B:** divergent | Posts orders during ticker migration and may incur replay, duplicate, or stale-balance exposure. | **Veto** all affected instruments and cancel resting orders. | `RR=0:1`; `E[R]≤0`; no ordinary stop can reliably bound technical loss. |
| **#89: Oracle Price Failure** | **H:** price series non-stationary; **F:** diverges from spot; **O:** isolated extreme; **A:** protocol-specific; **V:** abnormal; **B:** extreme | Uses manipulated or stale oracle prices as evidence of market value. | **Cash Standby** for oracle-dependent assets and correlated collateral. | `RR=0:1`; `E[R]=0`; liquidation and protocol insolvency risk are discontinuous. |
| **#90: Smart-Contract Exploit with Market Contagion** | **H:** >0.75; **F:** negative; **O:** <-0.90; **A:** >10%; **V:** +6; **B:** negative | Buys related tokens because the exploit appears already priced, without assessing treasury and collateral links. | **Veto** correlated long exposure; no reversal on the first flush. | `RR=0:1`; `E[R]≤0`; contagion paths are nonlinear and difficult to enumerate in real time. |
| **#91: Custodian or Prime-Broker Failure** | **H:** broad market >0.65; **F:** unstable; **O:** broad ±0.70; **A:** >8%; **V:** +5; **B:** venue dispersion | Assumes exchange balances, margin, and collateral are fungible across counterparties. | **Cash Standby**; freeze new exposure until counterparty balances are reconciled. | `RR=0:1`; `E[R]=0`; operational loss can occur without adverse market movement. |
| **#92: Banking-Rail or Fiat-Ramp Disruption** | **H:** 0.55–0.75; **F:** basis dislocated; **O:** mixed; **A:** 5–9%; **V:** +3; **B:** fiat-dependent venue premiums | Mistakes a fiat premium for directional demand and arbitrages a price that cannot be funded or withdrawn. | **Scale Down 50%** only on fully collateralized, venue-diversified positions; otherwise stand aside. | `RR=1.5:1`; `p=.48`, `G=1.5R`, `L=1R`; `E[R]=+.20R`, half-sized `+.10R`. |
| **#93: Sovereign Default / Capital-Control Shock** | **H:** >0.75; **F:** negative; **O:** <-0.80; **A:** >9%; **V:** +5 to +10; **B:** extreme cross-venue dispersion | Assumes price discovery is global and ignores regional capital controls and settlement restrictions. | **Cash Standby**; prohibit cross-venue transfer assumptions. | `RR=0:1`; `E[R]=0`; transferability is a first-order variable. |
| **#94: Internet or Cloud-Infrastructure Outage** | **H:** unavailable for affected venues; **F:** stale; **O:** incomplete; **A:** apparent spikes; **V:** unreliable; **B:** fragmented | Keeps automated orders active when connectivity, market data, and cancel acknowledgements are uncertain. | **Veto** all automated entries; cancel or flatten only according to verified connectivity procedures. | `RR=0:1`; `E[R]≤0`; technological gap risk invalidates backtested execution distributions. |
| **#95: Mining or Validator Shutdown Shock** | **H:** >0.65; **F:** negative; **O:** <-0.70; **A:** >7%; **V:** +4; **B:** negative | Treats network-security deterioration as a normal supply-side headline and buys the dip. | **Scale Down 50%** on broad-market trades; **Veto** affected-chain longs until operational status is confirmed. | Broad rebound: `RR=2.0:1`; `p=.40`, `G=2R`, `L=1R`; `E[R]=+.20R`, half-sized `+.10R`. |
| **#96: Wrapped-Asset De-Peg** | **H:** >0.70; **F:** distorted; **O:** affected wrapper <-0.90; **A:** wrapper >15%; **V:** +7; **B:** wrapper discount >500 bps | Assumes the wrapper is redeemable one-for-one and buys discount without checking custodian solvency. | **Veto** wrapper trades; do not use apparent discount as a signal for native asset direction. | `RR=0:1`; `E[R]≤0`; redemption uncertainty can overwhelm nominal discount. |
| **#97: Governance or Treasury Attack** | **H:** >0.75; **F:** negative; **O:** <-0.85; **A:** >12%; **V:** +6; **B:** negative | Buys a governance-token crash without assessing voting control, treasury drain, or emergency upgrade authority. | **Cash Standby** for affected protocol and economically linked tokens. | `RR=0:1`; `E[R]=0`; fundamental recovery distribution is bimodal. |
| **#98: Emergency Token Unlock or Forced Distribution** | **H:** >0.65; **F:** negative or neutral; **O:** <-0.70; **A:** asset-specific >12%; **V:** +4; **B:** negative | Treats a large unlock as a temporary dip and ignores persistent supply overhang. | **Take Profit Aggressive** on longs; **Veto** immediate dip-buying. | Long continuation has unfavorable profile: `RR=0.9:1`; `p=.43`, `G=.9R`, `L=1R`; `E[R]=-.187R`. |
| **#99: 51% Attack or Finality Reversal** | **H:** non-stationary; **F:** unreliable; **O:** fragmented; **A:** >15%; **V:** abnormal; **B:** venue divergence | Uses exchange prices despite uncertainty over transaction finality and settlement history. | **Veto** affected asset and correlated collateral; no technical reversal. | `RR=0:1`; `E[R]≤0`; settlement risk is not bounded by conventional stops. |
| **#100: Post-Black-Swan Reconstruction Regime** | **H:** falls from non-stationary to 0.45–0.60; **F:** normalizes; **O:** alternates before stabilizing; **A:** remains >8% but declines; **V:** +2 to +5; **B:** narrows | Re-enters full size immediately after the headline passes and mistakes stabilization for restored market integrity. | **Widen Stop** only with 50% notional, verified venue solvency, stable settlement, and two-sided liquidity. | Recovery setup: `RR=3.0:1`; `p=.38`, `G=3R`, `L=1R`; `E[R]=+.14R`; actual account exposure must remain capped until operational risks clear. |

---

# 3. Titan Action Protocols

## 3.1 Veto

A **Veto** blocks new exposure in the affected direction or asset.

Trigger conditions include:

- Unresolved liquidation cascade.
- Unreliable settlement or collateral.
- Unconfirmed regulatory or insolvency headline.
- Severe maker adverse selection.
- Stablecoin, oracle, bridge, or wrapped-asset impairment.
- Cross-venue price disagreement beyond validated tolerances.

A Veto is not a prediction that price must continue in the same direction. It means the expected payoff distribution is not sufficiently bounded for the strategy.

---

## 3.2 Scale Down 50%

Scale down applies when a trade remains potentially valid but the distribution has become wider or less reliable.

Implementation:

- Reduce gross notional to 50%.
- Preserve the original dollar-risk budget where possible.
- Do not compensate for reduced size by tightening stops.
- Require stronger confirmation from spot OFI and cross-venue prices.
- Reduce SOL exposure more aggressively than BTC exposure under broad stress.

---

## 3.3 Take Profit Aggressive

Use when:

- Funding or basis is extreme.
- A squeeze is near exhaustion.
- The event catalyst has already been monetized.
- Momentum decays while volume remains elevated.
- Price reaches a known liquidation, gap, or liquidity objective.

Operationally:

1. Take partial profit at the first objective.
2. Move the remainder to a structurally protected stop.
3. Do not allow a profitable crisis trade to become a full-risk loss.
4. Avoid re-entry solely because price continues after exit.

---

## 3.4 Widen Stop

Widening a stop is permitted only when the market thesis remains valid and the observed noise has increased.

The correct implementation is:

\[
\text{Position Notional}_{new}
=
\text{Position Notional}_{old}
\times
\frac{\text{Old Stop Distance}}{\text{New Stop Distance}}
\]

The account risk budget must remain approximately constant:

\[
R_{account}
\approx
N \times d_{stop} \times \text{price}
\]

Widening a stop without reducing notional is prohibited.

---

## 3.5 Reverse Position

A Reverse Position action is permitted only after:

- The prior directional thesis has invalidated.
- Price has reclaimed or rejected a defined structural level.
- OFI confirms the new direction.
- Spot and perpetual markets are not materially diverging.
- The trade can be executed passively without entering the toxic impulse.

Reversal trades should generally be half-sized on SOL and during event-driven conditions.

---

## 3.6 Cash Standby

Cash Standby means:

- No new directional exposure.
- Cancel stale maker orders.
- Reconcile collateral and venue balances.
- Monitor stablecoin conversion and withdrawal status.
- Continue data collection, but suspend automatic signal execution.
- Resume only after a documented normalization checklist passes.

Cash Standby is mandatory for operational or settlement uncertainty, even if a directional model displays a high historical win rate.

---

# 4. Asset-Specific Risk Adjustments

## BTCUSDT

BTC generally has:

- The deepest order book.
- The most reliable cross-venue price discovery.
- The strongest relationship to macro liquidity, rates, dollar strength, and ETF flows.
- The lowest, but not negligible, gap-through risk.

BTC may receive the least severe haircut during broad stress, but it remains subject to the same Veto rules during custody, stablecoin, and infrastructure events.

## ETHUSDT

ETH requires additional monitoring of:

- ETH/BTC relative flow.
- Staking and liquid-staking ecosystem risk.
- Gas, validator, and protocol-specific headlines.
- ETF-related positioning.
- Correlation changes during BTC-led liquidity migration.

ETH should not automatically inherit BTC directional signals during Regimes #08, #50, #83, #90, or #97.

## SOLUSDT

SOL requires the highest risk haircut because of:

- Greater realized volatility.
- Thinner depth during stress.
- Higher liquidation convexity.
- Greater sensitivity to ecosystem, validator, bridge, and protocol events.
- Larger weekend and session-transition gaps.

Recommended default adjustments:

| Condition | BTC | ETH | SOL |
|---|---:|---:|---:|
| Normal regime | 1.00x risk | 0.85x | 0.65x |
| Elevated volatility | 0.75x | 0.60x | 0.40x |
| Liquidation or event shock | 0.50x | 0.35x | 0.20x |
| Operational/systemic event | Veto or standby | Veto or standby | Veto |

---

# 5. Regime Classification and Conflict Resolution

Multiple regimes may be active simultaneously. The system must resolve conflicts using the following hierarchy:

1. **Operational and settlement risk**
2. **Black-swan and insolvency risk**
3. **Liquidation and forced-flow risk**
4. **Macro event risk**
5. **Microstructure execution risk**
6. **Volatility and trend classification**
7. **Ordinary technical signal**

The most restrictive action wins.

For example:

- A bullish breakout during a stablecoin de-peg remains **Cash Standby**.
- A long signal during a short squeeze may still be valid, but **Take Profit Aggressive** overrides normal trailing logic.
- A favorable macro signal during a maker adverse-selection zone is **Veto** until execution quality improves.
- A volatility breakout with a protocol exploit is not a breakout trade; it is a systemic-risk event.

---

# 6. Minimum Normalization Checklist

The system may transition from crisis controls back to normal operation only when all applicable conditions are satisfied:

- Hurst estimate has stabilized across at least three rolling windows.
- Funding is no longer accelerating toward an extreme.
- OFI has remained directionally coherent for at least two 1h bars.
- Volume has declined from panic levels without collapsing into zero liquidity.
- Spot-perpetual basis has narrowed toward the asset-specific normal range.
- Cross-venue mid-prices are within validated dispersion bands.
- Spreads and top-of-book depth have normalized.
- No unresolved exchange, stablecoin, chain, oracle, custody, or regulatory event remains.
- Maker fill quality is not showing material post-fill adverse selection.
- Portfolio-wide correlation and margin utilization are within limits.

---

# 7. Portfolio-Level Crisis Controls

Regardless of individual regime classification:

- Maximum simultaneous risk is capped at the portfolio level.
- Correlated BTC, ETH, and SOL positions must be aggregated by stress beta.
- No new trade may increase portfolio liquidation sensitivity during Regimes #01–#19 or #81–#99.
- A single venue must not hold collateral that is necessary to support all open positions.
- Stablecoin exposure must be treated as counterparty and liquidity exposure, not as risk-free cash.
- Stops are not assumed to execute at the trigger price during ATR spikes, weekend gaps, or exchange disruption.
- Expected payoff calculations must include:
  - Maker fees or rebates.
  - Missed-fill probability.
  - Adverse-selection loss.
  - Slippage.
  - Funding.
  - Basis convergence risk.
  - Transfer and withdrawal constraints.
- If the model cannot estimate these components with sufficient confidence, the action defaults to **Veto** or **Cash Standby**.

---

# 8. Closing Operating Principle

MaBuy V5.2 Titan should not maximize the number of trades. It should maximize the quality of the **risk-adjusted opportunity set**.

The defining principle of this playbook is:

> When market structure becomes discontinuous, preserving optionality is itself an alpha-producing action.

A system that remains solvent, liquid, and executable through the 100 regimes will possess a durable advantage over naive systems that continue applying ordinary signal logic to extraordinary market conditions.