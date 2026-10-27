*Model and date:* Claude Opus 5.5 Oct. 27, 2026

*Version 1 prompt:* [Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.

*Reply:* 

I implemented a mixture SPRT (mSPRT) rather than alpha-spending. Alpha-spending only controls error at looks you plan in advance against a fixed maximum sample size. The mSPRT stays valid under truly continuous, open-ended monitoring, which is what you asked for.

On your log structure: nothing was attached to the conversation, so I couldn’t reuse your schema. Each look instead emits one flat record with these fields: test_id, look, n_treatment, n_control, delta_hat, var_hat, log_lambda, p_always_valid, ci_low, ci_high, decision, stopped, naive_p and naive_reject. If you paste your schema, the key mapping in _record is the only thing that needs to change.

The construction

At each look, Δ̂ is the difference in means and V is its variance. Under H₀ the estimate is distributed N(0, V). Mixing the alternative over θ ~ N(0, τ²) gives a closed-form likelihood ratio:

Λₙ = √(V/(V+τ²)) · exp(τ²Δ̂² / (2V(V+τ²)))

Under H₀, Λₙ is a nonnegative martingale with mean 1. Ville’s inequality then gives P(supₙ Λₙ ≥ 1/α) ≤ α. This holds for any number of looks and any stopping rule, including “stop when it looks good.”

The decision rule is to reject when the running always-valid p-value, pₙ = min(pₙ₋₁, 1/Λₙ), reaches α. Inverting the test gives a confidence sequence, which is reported as ci_low and ci_high at every look.

The interface is sequential_decisions(stream, tau=..., sigma=...). It takes an iterable of (treatment_batch, control_batch) pairs and yields one decision record per look. The math runs in log space, so there is no overflow at large n. A vectorised engine runs the same math for Monte Carlo, and the script asserts that both paths give identical decisions on 200 streams.

Why naive peeking inflates Type I error

Under H₀, the z-statistic traced across looks is a Brownian motion in information time. It is not a fresh draw at each look. Each look is another chance to cross a fixed line at ±1.96, and the correlation between looks doesn’t save you.

By the law of the iterated logarithm, the running maximum of |z| grows like √(2 log log n), which is unbounded. So with enough looks, the probability of crossing 1.96 at some look goes to 1.

The mSPRT boundary, written on the z scale, grows like √(log n). That is strictly faster than √(log log n), so a null stream eventually stops being able to catch it. Validity comes entirely from the boundary outrunning the random walk.

What the always-valid construction gives up

Fixed-horizon power and precision. Take n = 10,000 per arm with τ = 0.1σ. The confidence-sequence half-width is 0.045σ, against 0.028σ for a fixed-horizon interval. That makes it about 60% wider; equivalently, you’d need roughly 2.6× the sample to match fixed-horizon precision at that point. In the illustrative portfolio, power was 64% versus 73% for a single look at the full horizon. The compensation is early stopping on large effects: the median stop for true positives was 2,800 per arm against a 10,000 maximum.

Unspent alpha. Ville’s bound is attained only with continuous monitoring and an infinite horizon. With a finite horizon and discrete looks, you get less than 5%, so part of the error budget is wasted.

Robustness to τ. Power is concentrated on effects near τ. A badly chosen τ costs power, not validity. However, choosing τ after looking at the data does break validity.

Unbiased estimates at the stopping point. Stopping on a crossing selects for overestimates. Estimates at the stop were a median 1.11× the true effect. Report the confidence sequence, not a fixed-horizon CI computed at the stop.

Robustness to model failures. The guarantee is only as good as its assumptions: iid observations, independent arms, a fixed allocation ratio, and a correct variance. Novelty effects, day-of-week drift, and session-level metrics randomized at the user level all violate them.

Results on 1,000 pure-null streams

Each stream had 200 looks of 50 observations per arm, up to 10,000 per arm, with α = 0.05.

Method	Empirical FPR	95% CI
mSPRT, known σ	2.6%	1.8–3.8%
mSPRT, plug-in σ (burn-in 100)	2.7%	1.9–3.9%
Naive z-test at every look	42.5%	39.5–45.6%
Fixed horizon, single look	5.8%	4.5–7.4%

The mSPRT result is not “close to 5%,” and nothing is wrong. That’s the expected behavior, and “close to 5%” is the wrong target. For an always-valid test, 5% is a ceiling, not a calibration point.

The fixed-horizon row is the real sanity check. It lands on 5% within Monte Carlo noise, which confirms the simulator is calibrated. Extending the horizon shows the mSPRT FPR climbing slowly toward the bound but never reaching it:

Horizon (per arm)	mSPRT FPR
1,000	0.95%
10,000	2.2%
100,000	3.5%
1,000,000	3.6%

What would actually indicate a bug is an FPR significantly above 5%. In my experience, the usual causes are:

variance underestimated, especially with plug-in estimates at small n or for low-rate conversion metrics;
τ in the wrong units (it must be in effect units, not standardized units, unless σ = 1);
τ chosen from the data;
unbalanced or drifting allocation;
correlated observations, such as multiple events per randomized user.

For a pure-null unit test, run around 1,000 null streams and assert that the Wilson lower bound on the FPR is at most α. Don’t assert that the FPR equals α.

To run this on your 200-test portfolio, replace the illustrative true effects array at the bottom of msprt.py with your known effects. If your data is binary or heavy-tailed rather than Gaussian, use run_on_raw with your actual observations and the plug-in variance (known_sigma=False).

"""
Always-valid sequential A/B testing via the mixture SPRT (mSPRT).

Reference: Johari, Koomen, Pekelis & Walsh, "Always Valid Inference:
Continuous Monitoring of A/B Tests" (Operations Research, 2022).

Model
-----
At look k we have n_T, n_C observations per arm and the estimate
    delta_hat = mean(T) - mean(C),   V = var(delta_hat) = s_T^2/n_T + s_C^2/n_C.
Under H0 (theta = theta0) delta_hat ~ N(theta0, V). Mixing the alternative
over theta ~ N(theta0, tau^2) gives the closed-form mixture likelihood ratio

    Lambda = sqrt(V / (V + tau^2)) * exp( tau^2 (delta_hat - theta0)^2 / (2 V (V + tau^2)) )

which is a nonnegative martingale with mean 1 under H0. Ville's inequality
gives P_H0(sup_k Lambda_k >= 1/alpha) <= alpha, for ANY number of looks
and ANY data-dependent stopping rule.

Validity assumptions (break these and the guarantee breaks):
  * observations i.i.d. within arm, arms independent
  * fixed allocation ratio across looks (balanced by default)
  * variance known, or plug-in estimated after a burn-in (approximate)
  * tau and alpha fixed BEFORE seeing data

Log structure
-------------
No log schema was attached, so each look emits one flat dict
(LOOK_LOG_FIELDS below). Rename keys in `_record` to match your own logs.

numpy + scipy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np
from scipy import stats

LOOK_LOG_FIELDS = (
    "test_id", "look", "n_treatment", "n_control", "delta_hat", "var_hat",
    "log_lambda", "p_always_valid", "ci_low", "ci_high",
    "decision", "stopped", "naive_p", "naive_reject",
)


# --------------------------------------------------------------------------
# Core math (vectorised, log-space to avoid overflow at large n)
# --------------------------------------------------------------------------
def log_mixture_lr(delta_hat, V, tau, theta0=0.0):
    """log Lambda for a N(theta0, tau^2) mixture over the effect."""
    tau2 = tau ** 2
    d = np.asarray(delta_hat) - theta0
    return 0.5 * np.log(V / (V + tau2)) + tau2 * d ** 2 / (2.0 * V * (V + tau2))


def confidence_sequence(delta_hat, V, tau, alpha):
    """Always-valid CI: the set of theta0 the mSPRT would not reject."""
    tau2 = tau ** 2
    half = np.sqrt(V * (V + tau2) / tau2 * (2 * np.log(1 / alpha) + np.log((V + tau2) / V)))
    return delta_hat - half, delta_hat + half


# --------------------------------------------------------------------------
# Streaming test
# --------------------------------------------------------------------------
@dataclass
class MSPRT:
    tau: float                       # mixing scale, in the SAME units as the effect
    alpha: float = 0.05
    theta0: float = 0.0
    sigma: Optional[float] = None    # known per-observation sd; None -> plug-in
    burn_in: int = 100               # min obs per arm before plug-in decisions
    test_id: str = "test"

    # running sufficient statistics: [n, sum, sum_sq] per arm
    _t: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)
    _c: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)
    _look: int = field(default=0, repr=False)
    _p: float = field(default=1.0, repr=False)
    _stopped: bool = field(default=False, repr=False)

    @staticmethod
    def _accumulate(acc, x):
        x = np.asarray(x, dtype=float).ravel()
        acc += (x.size, x.sum(), np.dot(x, x))

    @staticmethod
    def _moments(acc):
        n, s, ss = acc
        mean = s / n
        var = (ss - n * mean ** 2) / (n - 1) if n > 1 else np.nan
        return n, mean, var

    def update(self, treatment_batch, control_batch) -> dict:
        """Ingest one look's new observations; return that look's log record."""
        self._look += 1
        if not self._stopped:                    # data after stopping is ignored
            self._accumulate(self._t, treatment_batch)
            self._accumulate(self._c, control_batch)

        n_t, m_t, v_t = self._moments(self._t)
        n_c, m_c, v_c = self._moments(self._c)
        if self.sigma is not None:
            v_t = v_c = self.sigma ** 2
        delta = m_t - m_c
        V = v_t / n_t + v_c / n_c

        ready = self.sigma is not None or min(n_t, n_c) >= self.burn_in
        if ready and np.isfinite(V) and V > 0:
            llr = float(log_mixture_lr(delta, V, self.tau, self.theta0))
            # always-valid p-value is monotone: p_k = min(p_{k-1}, 1/Lambda_k)
            self._p = min(self._p, float(np.exp(-llr)))
            lo, hi = confidence_sequence(delta, V, self.tau, self.alpha)
        else:
            llr, lo, hi = np.nan, -np.inf, np.inf

        if not self._stopped and self._p <= self.alpha:
            self._stopped = True
            decision = "reject_h0"
        elif self._stopped:
            decision = "reject_h0"                # sticky
        else:
            decision = "continue"

        # Naive fixed-horizon z-test at this look (for comparison only)
        if np.isfinite(V) and V > 0:
            naive_p = float(2 * stats.norm.sf(abs(delta - self.theta0) / np.sqrt(V)))
        else:
            naive_p = np.nan

        return self._record(n_t, n_c, delta, V, llr, lo, hi, decision, naive_p)

    def _record(self, n_t, n_c, delta, V, llr, lo, hi, decision, naive_p):
        return {
            "test_id": self.test_id, "look": self._look,
            "n_treatment": int(n_t), "n_control": int(n_c),
            "delta_hat": float(delta), "var_hat": float(V),
            "log_lambda": llr, "p_always_valid": self._p,
            "ci_low": float(lo), "ci_high": float(hi),
            "decision": decision, "stopped": self._stopped,
            "naive_p": naive_p, "naive_reject": bool(naive_p <= self.alpha),
        }


def sequential_decisions(stream: Iterable[tuple], **kwargs) -> Iterator[dict]:
    """
    stream: iterable of (treatment_batch, control_batch), one tuple per look.
    Yields one log record per look; `decision` is the stopping decision.
    """
    test = MSPRT(**kwargs)
    for t_batch, c_batch in stream:
        yield test.update(t_batch, c_batch)


# --------------------------------------------------------------------------
# Fast vectorised engine for Monte Carlo (same math as MSPRT.update)
# --------------------------------------------------------------------------
def _engine(st, sc, sst, ssc, n, tau, alpha, known_sigma, sigma, burn_in):
    """Cumulative per-arm sums (m, looks) -> mSPRT and naive decisions."""
    m, n_looks = st.shape
    delta = st / n - sc / n
    if known_sigma:
        V = np.broadcast_to(2 * sigma ** 2 / n, delta.shape)
        ready = np.ones_like(delta, dtype=bool)
    else:
        vt = (sst - n * (st / n) ** 2) / (n - 1)
        vc = (ssc - n * (sc / n) ** 2) / (n - 1)
        V = vt / n + vc / n
        ready = np.broadcast_to(n >= burn_in, delta.shape)

    llr = np.where(ready, log_mixture_lr(delta, V, tau), -np.inf)
    hit = llr >= np.log(1 / alpha)
    msprt_reject = hit.any(1)
    stop_look = np.where(msprt_reject, hit.argmax(1) + 1, n_looks)
    naive_hit = np.abs(delta) / np.sqrt(V) >= stats.norm.isf(alpha / 2)
    return {
        "msprt_reject": msprt_reject,
        "stop_look": stop_look,
        "stop_n_per_arm": n[stop_look - 1],
        "stop_delta": delta[np.arange(m), stop_look - 1],
        "naive_reject_any_look": naive_hit.any(1),
        "fixed_horizon_reject": naive_hit[:, -1],
    }


def run_on_raw(t, c, tau=0.1, alpha=0.05, known_sigma=True, sigma=1.0, burn_in=100):
    """t, c: raw observations shaped (tests, looks, batch). Balanced batches."""
    n = t.shape[2] * np.arange(1, t.shape[1] + 1)
    return _engine(t.sum(2).cumsum(1), c.sum(2).cumsum(1),
                   (t ** 2).sum(2).cumsum(1), (c ** 2).sum(2).cumsum(1),
                   n, tau, alpha, known_sigma, sigma, burn_in)


def simulate_streams(effects, n_looks, batch, sigma=1.0, tau=0.1, alpha=0.05,
                     known_sigma=True, burn_in=100, seed=0):
    """
    Gaussian A/B streams with balanced batches, simulated exactly through the
    per-batch sufficient statistics (sum ~ Normal, within-batch SS ~ chi^2),
    so memory is O(tests x looks) instead of O(tests x looks x batch).
    """
    rng = np.random.default_rng(seed)
    effects = np.asarray(effects, dtype=float)
    m = effects.size

    def arm(mu):
        s = rng.normal(batch * mu[:, None], sigma * np.sqrt(batch), (m, n_looks))
        w = sigma ** 2 * rng.chisquare(batch - 1, (m, n_looks))   # within-batch SS
        return s.cumsum(1), (w + s ** 2 / batch).cumsum(1)

    st, sst = arm(effects)
    sc, ssc = arm(np.zeros(m))
    n = batch * np.arange(1, n_looks + 1)
    return _engine(st, sc, sst, ssc, n, tau, alpha, known_sigma, sigma, burn_in)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return centre - half, centre + half


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ALPHA, SIGMA, TAU = 0.05, 1.0, 0.1
    N_LOOKS, BATCH, N_NULL = 200, 50, 1000          # 10,000 obs/arm max

    # 1) Streaming API on a single pure-null stream (the unit-testable path)
    rng = np.random.default_rng(123)
    null_stream = ((rng.normal(0, 1, BATCH), rng.normal(0, 1, BATCH)) for _ in range(N_LOOKS))
    log = list(sequential_decisions(null_stream, tau=TAU, alpha=ALPHA, sigma=SIGMA,
                                    test_id="null_demo"))
    print("single null stream, final look:", {k: log[-1][k] for k in
          ("look", "decision", "p_always_valid", "ci_low", "ci_high")})

    # Cross-check: streaming class and vectorised engine agree on decisions
    rng = np.random.default_rng(7)
    t = rng.normal(0.0, SIGMA, (200, 50, BATCH))
    c = rng.normal(0.0, SIGMA, (200, 50, BATCH))
    t[:100] += 0.08                                    # half non-null, so both outcomes occur
    res_chk = run_on_raw(t, c, TAU, ALPHA, sigma=SIGMA)
    stream_rejects = []
    for i in range(200):
        recs = list(sequential_decisions(zip(t[i], c[i]), tau=TAU, alpha=ALPHA, sigma=SIGMA))
        stream_rejects.append(recs[-1]["stopped"])
    assert np.array_equal(stream_rejects, res_chk["msprt_reject"]), "engines disagree"
    print(f"streaming class == vectorised engine on 200 streams "
          f"({int(res_chk['msprt_reject'].sum())} rejections): OK\n")

    # 2) 1,000 pure-null streams
    print(f"=== {N_NULL} pure-null streams, {N_LOOKS} looks x {BATCH} obs/arm, alpha={ALPHA} ===")
    for label, known in (("mSPRT, known sigma", True), ("mSPRT, plug-in sigma", False)):
        r = simulate_streams(np.zeros(N_NULL), N_LOOKS, BATCH, SIGMA, TAU, ALPHA,
                             known_sigma=known, seed=2026)
        k = r["msprt_reject"].sum()
        lo, hi = wilson(k, N_NULL)
        print(f"{label:<24s} FPR = {k / N_NULL:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    k = r["naive_reject_any_look"].sum(); lo, hi = wilson(k, N_NULL)
    print(f"{'naive z-test, every look':<24s} FPR = {k / N_NULL:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    k = r["fixed_horizon_reject"].sum(); lo, hi = wilson(k, N_NULL)
    print(f"{'fixed horizon, 1 look':<24s} FPR = {k / N_NULL:.3f}  (95% CI {lo:.3f}-{hi:.3f})")

    # Horizon dependence: the Ville bound is approached only as looks -> inf
    print("\nmSPRT FPR vs horizon (known sigma, 4,000 streams each, simulated in chunks):")
    for looks in (20, 200, 2000, 20000):
        chunk = max(1, min(4000, 5_000_000 // looks))
        rej = np.concatenate([
            simulate_streams(np.zeros(chunk), looks, BATCH, SIGMA, TAU, ALPHA,
                             seed=looks * 1000 + j)["msprt_reject"]
            for j in range(4000 // chunk)])
        r = {"msprt_reject": rej}
        print(f"  {looks:>6d} looks ({looks * BATCH:>9,d}/arm): FPR = {r['msprt_reject'].mean():.4f}")

    # 3) Illustrative 200-test portfolio (swap in your own known effects here)
    rng = np.random.default_rng(99)
    true = np.where(rng.random(200) < 0.6, 0.0, rng.normal(0, 0.06, 200))
    r = simulate_streams(true, N_LOOKS, BATCH, SIGMA, TAU, ALPHA, seed=5)
    nz = true != 0
    print(f"\n=== Illustrative portfolio: 200 tests, {(~nz).sum()} null, {nz.sum()} non-null ===")
    print(f"null FPR      mSPRT {r['msprt_reject'][~nz].mean():.3f} | naive {r['naive_reject_any_look'][~nz].mean():.3f}")
    print(f"power (|θ|>0) mSPRT {r['msprt_reject'][nz].mean():.3f} | fixed-horizon {r['fixed_horizon_reject'][nz].mean():.3f}")
    rej = r["msprt_reject"] & nz
    sign_err = (np.sign(r["stop_delta"]) != np.sign(true)) & rej
    exagg = np.abs(r["stop_delta"][rej]) / np.abs(true[rej])
    print(f"median n/arm at stop (rejections): {np.median(r['stop_n_per_arm'][rej]):,.0f}")
    print(f"sign errors among rejections: {sign_err.sum()}")
    print(f"median |estimate at stop| / |true effect|: {np.median(exagg):.2f}  (winner's curse)")

*Version 2 prompt:*
[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects, in a pandas DataFrame with columns true, n, se, est and p.
Implement an always-valid sequential test -- a mixture sequential
probability ratio test or an alpha-spending boundary -- that allows
continuous monitoring without inflating the Type I error rate. Give me
a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Each experiment's stream is n
observations per arm (n from 40 to 399), arriving one pair per look:
treatment ~ Normal(true, 1), control ~ Normal(0, 1). The SD is known
to be 1, so the test should be allowed to stop from the first look,
with no minimum number of observations before it can make a decision.
Make the test one-sided (stop only for improvements) and tune it to
effects around 0.30. The function must be testable against a
pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams (true = 0, n from 40 to
399, seed 42). Report the empirical false positive rate and its Monte
Carlo standard error. At or below 5% is expected; tell me what is
wrong only if it is significantly above 5%.

*Reply:*
I’ve built a one-sided mixture SPRT and checked it on 1,000 pure-null streams. The empirical false positive rate is 2.5% (25 of 1,000), with a Monte Carlo SE of 0.49 points. That is about 3.6 SEs below 5%, so nothing is wrong.

For contrast, naive peeking on the same design (a one-sided z-test at every look) gives 32.5% ± 1.5%.

The construction

Each look gives one pair. Let D = treatment − control, so D ~ N(δ, 2), and σ² = 2 is known exactly.

Prior over effects. I average the likelihood ratio for δ against 0 over a half-normal prior on δ > 0, with scale τ.
Statistic. After m pairs with running sum S, the mixture likelihood ratio has a closed form:
Λₘ = 2·(1 + mτ²/σ²)^(−½) · exp(b²/2a) · Φ(b/√a), where a = 1/τ² + m/σ² and b = S/σ².
Why it’s valid. Under δ = 0, Λₘ is a nonnegative martingale starting at 1, and it only shrinks for δ < 0. Ville’s inequality then gives P(Λₘ ever ≥ 1/α) ≤ α over all looks, forever.
Stopping rule. Stop the first time Λₘ ≥ 20.
One-sided. Because the prior puts no mass on harm, the test can only stop for improvements.
No burn-in. The guarantee holds from look 1. In practice, at m = 1 the boundary is z ≈ 9, so an early stop needs an enormous signal.

Tuning. I grid-searched τ by simulating power at δ = 0.30, with n uniform on 40–399, using a separate seed (2024) fixed before the null run. Power is flat over τ ∈ [0.25, 0.55], and τ = 0.35 was best: 67% power, with a mean stopping look of about 113 among rejections.

The file contains:

msprt_decisions(treatment, control, tau): takes a whole stream and returns the per-look stop (absorbing), log_lr and always-valid p-value min(1, 1/max Λ).
MSPRT(tau).update(t, c): the same thing for a live stream, one pair at a time. A check confirms the two give identical decisions.
run_portfolio(true_effects, ns, tau, rng): simulates and tests a set of experiments.

No DataFrame was attached, but run_portfolio(df["true"], df["n"], 0.35, rng) will run the 200-test portfolio directly. It has to simulate streams from true and n, because est, se and p are end-of-test summaries with no path to monitor.

Why naive peeking inflates Type I error

A fixed-n z-test controls P(Z_n ≥ 1.645) at one pre-chosen n. Peeking at every look instead asks whether the maximum of Z₁, …, Z_n ever reaches 1.645, and that event is much more likely.

Under H0, the running sum is a random walk, and the z-statistics at successive looks are highly correlated but not identical. Every look is a fresh chance for noise to cross the line, and the chances accumulate. By the law of the iterated logarithm, the maximum of Zₘ keeps growing like √(2 log log m). So with unlimited looks, the probability of a false stop goes to 1, not just above 5%. With up to 399 looks you get the 32.5% above.

What the always-valid test gives up

The error guarantee is exact and holds under any monitoring or stopping behaviour. What you pay with is power and sample size.

1. A wider boundary at every fixed n. On the z-scale the boundary is:

Look m	1	10	40	100	200	399
z boundary	9.02	3.67	2.87	2.77	2.79	2.86

A fixed-n test needs only 1.645. The boundary is tightest near the sample size where the prior “expects” to detect δ ≈ 0.3–0.35, and it grows like √(log m) beyond that. That slow growth is exactly what beats the law-of-the-iterated-logarithm drift.

2. Lower power at the horizon. Averaged over n ∈ [40, 399] at δ = 0.30, a single look at the planned n gives 85.7% power, versus 67% for the mSPRT. That gap is the price of being allowed to look continuously.

3. Unspent alpha on finite horizons. Ville’s bound covers an infinite horizon. When you truncate at n ≤ 399, part of the 5% is never used, which is why the realised rate is 2.5% rather than 5%. You could reclaim some of it with a truncated boundary or alpha-spending, but that ties you to the horizon again.

4. Sensitivity to the tuning effect. Effects far from τ are detected more slowly than a test designed for them would manage. The power curve over τ is flat, so the cost is modest near 0.3, but it is real at, say, 0.05.

5. No harm detection. One-sidedness means the test is blind to harm by construction. If you need to stop for regressions, run a separate harm-side mSPRT as a guardrail and split α between the two tests.

What you gain. You can stop as soon as the evidence is in. Among the false positives, stopping looks ran from 11 to 229, and true effects typically stop well before n. You can also stop or continue for any reason, and the always-valid p-value stays honest.

"""
One-sided mixture SPRT (mSPRT) for a two-arm test with known variance.

Data model per look k (one pair per look):
    treatment_k ~ N(delta, 1), control_k ~ N(0, 1)
    D_k = treatment_k - control_k ~ N(delta, sigma2), sigma2 = 2 (known)

H0: delta <= 0   vs   H1: delta > 0   (stop only for improvements)

Mixing distribution: half-normal on delta > 0 with scale tau.
After m pairs with S_m = sum(D_1..D_m), the mixture likelihood ratio is

    Lambda_m = 2 / sqrt(1 + m tau^2 / sigma2)
               * exp( b^2 / (2a) ) * Phi( b / sqrt(a) )
    with a = 1/tau^2 + m/sigma2,  b = S_m / sigma2.

Lambda_m is a nonnegative martingale under delta = 0 (supermartingale for
delta < 0) with Lambda_0 = 1, so by Ville's inequality
    P_H0( exists m: Lambda_m >= 1/alpha ) <= alpha,
for ANY number of looks, including stopping at look 1. No burn-in needed.

numpy + scipy only.
"""
import numpy as np
from scipy.special import log_ndtr

SIGMA2 = 2.0          # Var(treatment - control) with unit SDs
ALPHA = 0.05


def log_mixture_lr(S, m, tau, sigma2=SIGMA2):
    """log Lambda_m for cumulative sum(s) S at look(s) m (vectorised)."""
    S = np.asarray(S, dtype=float)
    m = np.asarray(m, dtype=float)
    a = 1.0 / tau**2 + m / sigma2
    b = S / sigma2
    return (np.log(2.0) - 0.5 * np.log1p(m * tau**2 / sigma2)
            + b**2 / (2.0 * a) + log_ndtr(b / np.sqrt(a)))


def msprt_decisions(treatment, control, tau, alpha=ALPHA, sigma2=SIGMA2):
    """
    Process a stream of (treatment, control) pairs, one pair per look.

    Returns a dict of per-look arrays (length = number of looks):
        stop      : bool, True from the first look where Lambda >= 1/alpha on
                    (the decision is absorbing: once stopped, it stays stopped)
        log_lr    : log Lambda_m
        p_always  : always-valid p-value, min(1, 1 / max_{j<=m} Lambda_j)
    plus 'stop_look' (1-based look of rejection, or None).
    """
    d = np.asarray(treatment, float) - np.asarray(control, float)
    m = np.arange(1, d.size + 1)
    llr = log_mixture_lr(np.cumsum(d), m, tau, sigma2)
    crossed = llr >= np.log(1.0 / alpha)
    stop = np.maximum.accumulate(crossed)
    p_always = np.minimum(1.0, np.exp(-np.maximum.accumulate(llr)))
    first = int(np.argmax(crossed)) + 1 if crossed.any() else None
    return {"stop": stop, "log_lr": llr, "p_always": p_always,
            "stop_look": first}


class MSPRT:
    """Streaming version: call .update(t, c) as each pair arrives."""

    def __init__(self, tau, alpha=ALPHA, sigma2=SIGMA2):
        self.tau, self.alpha, self.sigma2 = tau, alpha, sigma2
        self.m, self.S, self.stopped, self.max_llr = 0, 0.0, False, 0.0

    def update(self, t, c):
        self.m += 1
        self.S += t - c
        llr = float(log_mixture_lr(self.S, self.m, self.tau, self.sigma2))
        self.max_llr = max(self.max_llr, llr)
        if llr >= np.log(1.0 / self.alpha):
            self.stopped = True
        return {"look": self.m, "stop": self.stopped, "log_lr": llr,
                "p_always": min(1.0, np.exp(-self.max_llr))}


# --------------------------------------------------------------------------
# Simulation helpers
# --------------------------------------------------------------------------
def simulate_streams(true_effects, ns, rng):
    """Yield (treatment, control) arrays per experiment."""
    for delta, n in zip(true_effects, ns):
        yield rng.normal(delta, 1.0, n), rng.normal(0.0, 1.0, n)


def run_portfolio(true_effects, ns, tau, rng, alpha=ALPHA):
    """Returns (rejected bool array, stop_look array with n+1 = no stop)."""
    rej, looks = [], []
    for (t, c), n in zip(simulate_streams(true_effects, ns, rng), ns):
        r = msprt_decisions(t, c, tau, alpha)
        rej.append(r["stop_look"] is not None)
        looks.append(r["stop_look"] if r["stop_look"] else n + 1)
    return np.array(rej), np.array(looks)


def naive_peeking_rejects(true_effects, ns, rng, alpha=ALPHA):
    """One-sided z-test at every look; reject if ever z >= z_{1-alpha}."""
    from scipy.stats import norm
    zc = norm.ppf(1 - alpha)
    out = []
    for t, c in simulate_streams(true_effects, ns, rng):
        d = t - c
        m = np.arange(1, d.size + 1)
        z = np.cumsum(d) / np.sqrt(SIGMA2 * m)
        out.append((z >= zc).any())
    return np.array(out)


def tune_tau(delta=0.30, n_lo=40, n_hi=399, n_sims=2000, seed=2024,
             grid=np.round(np.arange(0.05, 1.01, 0.05), 2)):
    """Pick tau maximising power at delta over n ~ U{n_lo..n_hi}.
    Uses its own seed, fixed before any null evaluation."""
    rng = np.random.default_rng(seed)
    ns = rng.integers(n_lo, n_hi + 1, n_sims)
    D = [rng.normal(delta, 1, n) - rng.normal(0, 1, n) for n in ns]
    cums = [np.cumsum(d) for d in D]
    res = {}
    for tau in grid:
        thr = np.log(1 / ALPHA)
        hits, looks = 0, []
        for S, n in zip(cums, ns):
            llr = log_mixture_lr(S, np.arange(1, n + 1), tau)
            idx = np.flatnonzero(llr >= thr)
            if idx.size:
                hits += 1
                looks.append(idx[0] + 1)
        res[float(tau)] = (hits / n_sims, np.mean(looks) if looks else np.nan)
    best = max(res, key=lambda k: res[k][0])
    return best, res


if __name__ == "__main__":
    # 1) tune tau for delta ~ 0.30 (separate seed)
    tau, grid = tune_tau()
    print("tau grid (power at delta=0.30, mean stop look | rejected):")
    for k, (pw, ml) in grid.items():
        print(f"  tau={k:.2f}  power={pw:.3f}  mean_stop={ml:6.1f}")
    print(f"selected tau = {tau}")

    # 2) pure-null evaluation: 1,000 streams, n ~ U{40..399}, seed 42
    rng = np.random.default_rng(42)
    N = 1000
    ns = rng.integers(40, 400, N)
    rej, looks = run_portfolio(np.zeros(N), ns, tau, rng)
    fpr = rej.mean()
    se = np.sqrt(fpr * (1 - fpr) / N)
    print(f"\nmSPRT null FPR = {fpr:.3f}  (MC SE {se:.4f}), "
          f"{rej.sum()} / {N} rejections")
    print(f"z vs 5%: {(fpr - 0.05) / np.sqrt(0.05 * 0.95 / N):+.2f}")
    print("stop looks of false positives:", np.sort(looks[rej]))

    # 3) contrast: naive peeking on same design (different draws, seed 43)
    naive = naive_peeking_rejects(np.zeros(N), ns, np.random.default_rng(43))
    print(f"\nnaive peeking FPR = {naive.mean():.3f} "
          f"(MC SE {np.sqrt(naive.mean()*(1-naive.mean())/N):.4f})")

    # 4) streaming class agrees with vectorised version
    r2 = np.random.default_rng(0)
    t, c = r2.normal(0.3, 1, 300), r2.normal(0, 1, 300)
    v = msprt_decisions(t, c, tau)
    s = MSPRT(tau)
    stream = [s.update(a, b)["stop"] for a, b in zip(t, c)]
    assert np.array_equal(v["stop"], np.array(stream))
    print("streaming == vectorised: OK")
