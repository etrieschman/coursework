# %%
"""Starter code for the problem "Beliefs and Their Failures" (6.S974).

Fill in every block marked TODO: Myopic.choose, Thompson.choose,
Thompson.update and Optimism.choose. Keep the method names and arguments as
they are, because simulate() and trapped_by() call them.

The model
---------
On each trip the operator sails the strait (S) or the Cape (L).
  S costs 8 + 20 * 1{hit}. Each S trip is hit with probability q.
  L costs 13 and is never hit.
The expected cost of S is 8 + 20 q, so S is the better route when q < 0.25.
The operator sees the outcome of S only on trips where it sails S.

How the code is organized
-------------------------
A rule is a class with two methods:
  choose(t)          returns a boolean array with one entry per run;
                     True means "sail S on trip t" (t starts at 1).
  update(sail, hit)  records the outcome of the trip. Both arguments are
                     boolean arrays. A hit only counts where sail is True.
Each rule keeps R independent operators side by side in numpy arrays, so
one call to choose() makes the decision for all R runs at once. Write your
rules with array operations (np.where, &, |, ~), not loops over runs.

Getting started
---------------
    import numpy as np
    from strait import Myopic, simulate, trapped_by, recursion
    from strait import plot_regret, plot_share

    # Exact probability of switching to L by the n-th S trip,
    # for the Beta(1,5) operator at q = 0.05.
    tb, EL = recursion(1, 5, 0.05, 400)
    print(tb[2], tb[400])

    # The same probability by simulation, once Myopic.choose works.
    # p[n-1] estimates tb[n].
    p = trapped_by(lambda R, rng: Myopic(1, 5, R, rng), 0.05, 400)
    print(p[1], p[-1])

    # Regret over 1000 trips with q = 0.05 on every trip.
    res = simulate(lambda R, rng: Myopic(1, 5, R, rng), np.full(1000, 0.05))
    print(res["total"], res["se_total"])

    # Figures for parts (d) and (e).
    plot_regret({"myopic": res["cum_regret"]}, "q = 0.05", "fig_regret.pdf")
    plot_share({"myopic": res["share_S"]}, "fig_share.pdf")

Checking your rules
-------------------
Run self_check() after you fill in each TODO, either from Python or with
    python strait.py
It tests the three rules on small cases and prints PASS or FAIL for each
test. Passing does not prove
a rule is right, but a failure points to a specific mistake.

For part (b), tb[2] should equal your answer to (b)(ii) exactly. The
simulated p[1] should be close to it, within about three standard errors,
sqrt(p (1 - p) / R).

Requires Python 3.9 or later with numpy and matplotlib.
"""

from collections import defaultdict
from fractions import Fraction
from math import comb

import numpy as np

C_S0, C_HIT, C_L = 8.0, 20.0, 13.0
QSTAR = (C_L - C_S0) / C_HIT  # 0.25, the break-even hit rate


def expected_cost(q):
    """Expected cost of S and of L when the hit rate is q."""
    return C_S0 + C_HIT * q, C_L


def gaps(q):
    """Regret per trip of sailing S and of sailing L when the hit rate is q.

    The regret of a route is its expected cost minus the expected cost of
    the better route. The better route has regret 0.
    """
    cS, cL = expected_cost(q)
    best = min(cS, cL)
    return cS - best, cL - best


# %%
# ======================================================================
# Rules
# ======================================================================
class Myopic:
    """Sail S when the posterior mean of the hit rate is below 0.25.

    The prior is Beta(a, b). After n S trips with h hits, the posterior is
    Beta(a + h, b + n - h), with mean (a + h) / (a + b + n). The rule uses
    only this mean. It never sails S just to learn, so once it chooses L it
    gets no new data and chooses L forever.
    """

    def __init__(self, a, b, R, rng):
        self.a, self.b = a, b
        self.n = np.zeros(R)  # S trips so far, one entry per run
        self.h = np.zeros(R)  # hits so far

    def mean(self):
        return (self.a + self.h) / (self.a + self.b + self.n)

    def choose(self, t):
        return 4 * (self.a + self.h) < (self.a + self.b + self.n)

    def update(self, sail, hit):
        self.n += sail
        self.h += sail & hit


class Experiment(Myopic):
    """Sail S on the first m trips whatever happens, then follow Myopic.

    The forced trips give the operator data on S before it is allowed to
    quit, so a few early hits cannot end the experiment.
    """

    def __init__(self, a, b, m, R, rng):
        super().__init__(a, b, R, rng)
        self.m = m

    def choose(self, t):
        if t <= self.m:
            return np.ones_like(self.n, dtype=bool)
        return super().choose(t)


class Thompson:
    """Thompson sampling with a Beta(a, b) prior on the hit rate of S.

    On each trip, draw one hit rate from the posterior and act as if it
    were the true rate. While the posterior is wide, some draws fall below
    0.25 even when the mean is above it, so the rule keeps trying S. As
    data accumulate, the posterior narrows and the rule tries the worse
    route less and less often.
    """

    def __init__(self, a, b, R, rng):
        self.a, self.b, self.rng = a, b, rng
        self.hits = np.zeros(R)
        self.miss = np.zeros(R)

    def choose(self, t):
        # TODO: draw one hit rate per run from the posterior (use
        # self.rng.beta) and return True where the draw is below QSTAR.
        raise NotImplementedError

    def update(self, sail, hit):
        # TODO: add a hit or a miss, but only in runs where sail is True.
        raise NotImplementedError


class Optimism:
    """The optimistic rule: UCB1 for the ship.

    The cost of L is known, so only S needs a bound. The rule judges S by
    a lower confidence bound on its hit rate: the observed rate minus a
    bonus,
        max(0, h/n - sqrt(2 ln t / n)).
    Subtracting the bonus lowers the estimated cost of S. The rule sails S on
    trip 1, and afterwards whenever the bound is
    below 0.25.
    The bonus sqrt(2 ln t / n) shrinks as S is sailed more (n grows) and
    grows slowly when S is not sailed (t grows), so S is always tried
    again eventually.
    """

    def __init__(self, R, rng):
        self.n = np.zeros(R)
        self.h = np.zeros(R)

    def choose(self, t):
        # TODO: return True where n == 0. In the other runs, compute the
        # bound max(0, h/n - sqrt(2 ln t / n)) and return True where it is
        # below QSTAR. (Avoid dividing by zero in the runs with n == 0.)
        raise NotImplementedError

    def update(self, sail, hit):
        self.n += sail
        self.h += sail & hit


# ======================================================================
# Simulation
# ======================================================================
def simulate(make_rule, q_path, R=10_000, seed=0, checkpoints=()):
    """Run R independent operators for len(q_path) trips.

    make_rule(R, rng) must return a rule, for example
        lambda R, rng: Thompson(1, 5, R, rng).
    q_path[t-1] is the hit rate on trip t. Regret on each trip is measured
    against an operator who knows the current hit rate.

    Returns a dictionary with
      share_S     share_S[t-1] is the fraction of runs that sail S on trip t
      regret      regret[t-1] is the average regret on trip t
      cum_regret  running sum of regret
      total       average total regret per run
      se_total    standard error of total regret, across runs
      run_total   total regret of each run (length R)
      run_cum     for each trip t in checkpoints, the regret of each run up
                  to and including trip t. Subtract two of these to get the
                  regret of each run over a phase, then average and take
                  the standard error.
    """
    rng = np.random.default_rng(seed)
    rule = make_rule(R, rng)
    T = len(q_path)
    share_S = np.zeros(T)
    reg = np.zeros(T)
    run_total = np.zeros(R)
    run_cum = {}
    for t in range(1, T + 1):
        q = q_path[t - 1]
        gS, gL = gaps(q)
        sail = rule.choose(t)
        hit = rng.random(R) < q
        rule.update(sail, hit)
        r = np.where(sail, gS, gL)
        run_total += r
        share_S[t - 1] = sail.mean()
        reg[t - 1] = r.mean()
        if t in checkpoints:
            run_cum[t] = run_total.copy()
    return dict(
        share_S=share_S,
        regret=reg,
        cum_regret=np.cumsum(reg),
        total=run_total.mean(),
        se_total=run_total.std(ddof=1) / np.sqrt(R),
        run_total=run_total,
        run_cum=run_cum,
    )


def trapped_by(make_rule, q, N, R=10_000, seed=1):
    """Simulated probability that a myopic operator has switched to L.

    Returns an array out with out[n-1] = fraction of runs that have switched
    to L after at most n S trips. Use it only for Myopic: a myopic operator
    that chooses L once gets no new data and never sails S again. For
    Thompson sampling or optimism, choosing L once does not mean L forever.
    """
    rng = np.random.default_rng(seed)
    rule = make_rule(R, rng)
    quit_ = ~rule.choose(1)  # a prior mean of 1/4 or more means L from trip 1
    out = []
    for t in range(1, N + 1):
        sail = ~quit_
        hit = rng.random(R) < q
        rule.update(sail, hit)
        quit_ |= ~rule.choose(t + 1)
        out.append(quit_.mean())
    return np.array(out)


def plot_regret(curves, title, filename):
    """Plot cumulative regret against the trip number, with a log scale for
    the trip number. curves maps a label to a cum_regret array from
    simulate(). The legend gives each curve's value at the last trip."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, y in curves.items():
        t = np.arange(1, len(y) + 1)
        ax.plot(t, y, label=f"{label} ({y[-1]:.1f})")
    ax.set_xscale("log")
    ax.set_xlabel("trip $T$ (log scale)")
    ax.set_ylabel("cumulative regret $R_T$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_share(shares, filename, window=25, change=None):
    """Plot the share of runs on S, averaged over the last `window` trips.
    shares maps a label to a share_S array from simulate(). If change is a
    trip number, a dotted line marks it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for label, s in shares.items():
        ma = np.convolve(s, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(s) + 1), ma, label=label)
    if change is not None:
        ax.axvline(change, color="grey", ls=":", lw=1)
    ax.set_xlabel("trip $t$")
    ax.set_ylabel(f"share of runs on S\n({window}-trip average)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# ======================================================================
# Exact calculations (complete; read these before part b)
#
# A myopic operator with prior Beta(a, b) switches to L for good as soon as
# (a + h) / (a + b + n) >= 1/4. The recursion tracks P[h], the probability
# of having h hits after n S trips without having switched yet. Each step
# adds one S trip, then removes the states where the operator switches.
# The tie test uses exact fractions.
# ======================================================================
THRESH = Fraction(1, 4)


def quits(a, b, n, h):
    """True if the operator switches to L after n S trips with h hits."""
    return Fraction(a + h, a + b + n) >= THRESH


def quit_line(a, b, n):
    """Smallest number of hits that makes the operator switch after n S trips."""
    h = 0
    while not quits(a, b, n, h):
        h += 1
    return h


def recursion(a, b, q, T, start=None):
    """Exact switching probabilities for the myopic rule.

    Returns (trapped_by, EL):
      trapped_by[n]  probability of having switched to L after at most n S trips
      EL             expected number of L trips among trips 1..T
    An operator that switches right after its n-th S trip sails L on the
    remaining T - n trips.
    start = (n0, {h: probability}) starts the myopic phase after n0 forced
    S trips; experiment() uses this.
    """
    n0, P = start if start is not None else (0, {0: 1.0})
    trapped, EL = 0.0, 0.0
    keep = {}
    for h, p in P.items():
        if quits(a, b, n0, h):
            trapped += p
            EL += p * (T - n0)
        else:
            keep[h] = p
    P = keep
    trapped_by = {n0: trapped}
    for n in range(n0 + 1, T + 1):
        nxt = defaultdict(float)
        for h, p in P.items():
            nxt[h + 1] += p * q
            nxt[h] += p * (1 - q)
        P = {}
        for h, p in nxt.items():
            if quits(a, b, n, h):
                trapped += p
                EL += p * (T - n)
            else:
                P[h] = p
        trapped_by[n] = trapped
    return trapped_by, EL


def experiment(a, b, m, q, T):
    """Exact results for m forced S trips followed by the myopic rule.

    Returns four numbers:
      the probability of sailing L on trip m + 1;
      the probability of having switched to L by the T-th S trip
        (for the cases in this problem set, T = 400 gives the probability
        of ending up on L for good to the precision you need);
      the expected number of L trips among trips 1..T;
      the expected number of S trips among trips 1..T.
    """
    start = {h: comb(m, h) * q**h * (1 - q) ** (m - h) for h in range(m + 1)}
    p_next = sum(p for h, p in start.items() if quits(a, b, m, h))
    tb, EL = recursion(a, b, q, T, start=(m, start))
    return p_next, tb[T], EL, T - EL


# ======================================================================
# Self-check (complete; run it after filling in the TODOs)
# ======================================================================
def self_check():
    """Test Myopic, Thompson and Optimism on small cases."""

    def report(name, ok, hint):
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "\n      " + hint))

    rng = np.random.default_rng(0)

    # Myopic: the simulated switching probability must match the recursion.
    try:
        R = 200_000
        tb, _ = recursion(1, 5, 0.05, 10)
        p = trapped_by(lambda R, rng: Myopic(1, 5, R, rng), 0.05, 10, R=R, seed=2)
        ok = all(
            abs(p[n - 1] - tb[n]) <= 4 * np.sqrt(tb[n] * (1 - tb[n]) / R)
            for n in range(1, 11)
        )
        report(
            "Myopic matches recursion for n = 1..10",
            ok,
            "Check the tie rule: a posterior mean of exactly 1/4 must give L.",
        )
    except NotImplementedError:
        print("SKIP  Myopic is not filled in yet")

    # Thompson: L trips must not change the posterior.
    try:
        th = Thompson(1, 5, 4, rng)
        sail = np.array([False, False, True, True])
        hit = np.array([True, False, True, False])
        th.update(sail, hit)
        ok = np.array_equal(th.hits, [0, 0, 1, 0]) and np.array_equal(
            th.miss, [0, 0, 0, 1]
        )
        report(
            "Thompson counts change only on S trips",
            ok,
            "A run that sails L learns nothing about S; leave its counts alone.",
        )
        c = th.choose(2)
        ok = isinstance(c, np.ndarray) and c.dtype == bool and c.shape == (4,)
        report(
            "Thompson.choose returns one True/False per run",
            ok,
            "Return a boolean numpy array of length R.",
        )

        class FakeRNG:
            """Returns fixed draws and records the Beta parameters it was given."""

            def beta(self, a, b, size=None):
                self.args = (np.asarray(a, float), np.asarray(b, float))
                return np.array([0.10, 0.25, 0.30, 0.20])

        fake = FakeRNG()
        th.rng = fake
        c = th.choose(3)
        ok = (
            hasattr(fake, "args")
            and np.array_equal(fake.args[0], [1, 1, 2, 1])
            and np.array_equal(fake.args[1], [5, 5, 5, 6])
        )
        report(
            "Thompson draws from Beta(a + hits, b + misses)",
            ok,
            "Pass a + hits and b + misses to self.rng.beta, one entry per run.",
        )
        ok = np.array_equal(c, [True, False, False, True])
        report(
            "Thompson sails S only when the draw is below 0.25",
            ok,
            "Compare each draw with QSTAR; a draw of exactly 0.25 means L.",
        )
    except NotImplementedError:
        print("SKIP  Thompson is not filled in yet")

    # Optimism: S first; after 8 hits in 8 trips, L at t = 9 and S at t = 100.
    try:
        op = Optimism(3, rng)
        ok1 = bool(np.all(op.choose(1)))
        op.n[:] = 8
        op.h[:] = 8
        ok2 = not np.any(op.choose(9))
        ok3 = bool(np.all(op.choose(100)))
        report(
            "Optimism sails S before any data", ok1, "With n = 0 the rule must sail S."
        )
        report(
            "Optimism: 8 hits in 8 trips gives L at t = 9",
            ok2,
            "The bound is 1 - sqrt(2 ln 9 / 8) = 0.259, which is not below 0.25.",
        )
        report(
            "Optimism: the same data give S at t = 100",
            ok3,
            "The bonus must use ln t (the trip number), not ln n.",
        )
    except NotImplementedError:
        print("SKIP  Optimism is not filled in yet")


if __name__ == "__main__":
    self_check()
