# %%
import numpy as np
import cvxpy as cp
import polars as pl
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
pl.Config.set_tbl_cols(-1)

outpath = (
    Path(__file__).resolve().parent.parent / "overleaf" / "6_S974" / "tabs_and_figs"
)
outpath.parent.mkdir(parents=True, exist_ok=True)


# %%
# ===========================
print("Problem 1 - Fictitious play")
# ===========================
print("Part 1.a - Zero-sum NE")
# ---------------------------
# constants
v = 1 / 2


# intantiate payoff matrix for input v
def U(v):
    return np.array(
        [[[v, v], [0, 1], [1, 0]], [[1, 0], [v, v], [0, 1]], [[0, 1], [1, 0], [v, v]]]
    )


# wrap problem in method to solve for each commander
def zero_sum_LP(U, i):
    n = U.shape[0]
    Ui = U[:, :, 0] if i == 0 else U[:, :, 1].T
    sigma = cp.Variable(n, nonneg=True)
    z = cp.Variable()
    # constraints
    const = [cp.sum(sigma) == 1]
    const += [z <= sigma @ Ui[i, :] for i in range(n)]
    # problem
    prob = cp.Problem(cp.Maximize(z), const)
    prob.solve()
    print(f"\tsigma: {np.round(sigma.value, 3)}")


Uz = U(1 / 2) - 1 / 2
for comm in [0, 1]:
    print(f"Commander {comm + 1} optimal strategy:")
    zero_sum_LP(Uz, comm)

# %%
# ===========================
print("Part 1.c - Fictitious play")
# ---------------------------
T = 10_000


def best_response(Ui, eta):
    # np.argmax breaks ties by index order
    # so no additional tie-breaking is needed
    # skip the normalization of eta into mu
    return int(np.argmax(Ui @ eta))


def play(Uv, eta1_0, eta2_0, T=T):
    # initialize
    eta1 = eta1_0.copy()
    eta2 = eta2_0.copy()
    sigma1 = np.zeros((T, 4))
    sigma2 = np.zeros((T, 4))

    for t in range(T):
        # 2. best response
        br1 = best_response(Uv[:, :, 0], eta1)
        br2 = best_response(Uv[:, :, 1].T, eta2)
        # 3. update counts
        eta1[br2] += 1
        eta2[br1] += 1
        # 4. store strategies and br
        sigma1[t:, br1] += 1
        sigma2[t:, br2] += 1
        sigma1[t, 3] = br1
        sigma2[t, 3] = br2

    # post-process sigma
    sigma1[:, :3] = sigma1[:, :3] / np.arange(1, T + 1)[:, None]
    sigma2[:, :3] = sigma2[:, :3] / np.arange(1, T + 1)[:, None]

    return sigma1, sigma2


def plot_proportions(sigma1, sigma2, title, subtitle):
    T = sigma1.shape[0]
    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, sigma, name in zip(axes, [sigma1, sigma2], ["Commander 1", "Commander 2"]):
        for a, label in enumerate(["Attack", "Defend", "Grow"]):
            ax.plot(t, sigma[:, a], label=label)
        ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Encounter $t$")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Empirical proportion")
    axes[1].legend()
    fig.suptitle(f"{title}\n{subtitle}")
    fig.tight_layout()
    plt.savefig(outpath / f"hw3_problem1_partc_{title.replace(' ', '_')}.png", dpi=300)
    return fig


# ------------------
print("Experiment I")
# ------------------
sigma1, sigma2 = play(
    Uv=U(1 / 2),
    eta1_0=[1, 0, 0],
    eta2_0=[0, 0, 1],
)
plot_proportions(
    sigma1, sigma2, "Experiment I", "$v = 1/2,\;\eta_1=[1,0,0],\;\eta_2=[0,0,1]$"
)
plt.show()

# ------------------
print("Experiment II")
# ------------------
sigma1, sigma2 = play(
    Uv=U(0),
    eta1_0=[1, 0, 0],
    eta2_0=[0, 0, 1],
)
plot_proportions(
    sigma1, sigma2, "Experiment II", "$v = 0,\;\eta_1=[1,0,0],\;\eta_2=[0,0,1]$"
)
plt.show()

# ------------------
print("Experiment III")
# ------------------
sigma1, sigma2 = play(
    Uv=U(0),
    eta1_0=[1, 1, 1],
    eta2_0=[1, 1, 1],
)
plot_proportions(
    sigma1, sigma2, "Experiment III", "$v = 0,\;\eta_1=[1,1,1],\;\eta_2=[1,1,1]$"
)
plt.show()

# %%
# ===========================
print("Problem 2 - Beliefs and failures")
# ===========================
print("Part 2.b.iii - Probability of switching")
# ---------------------------
from hw3_strait import recursion, Myopic, trapped_by

R = 10_000
n = 400
q = 0.05
alpha, beta = 1, 5

# Exact probability of switching to L
tb, EL = recursion(alpha, beta, q, n)
print("Computed P(tau <= 400) =\t", np.round(tb[n], 4))

# The same probabilities by simulation
p = trapped_by(lambda R, rng: Myopic(alpha, beta, R, rng), q, n, R)
pn = p[n - 1]
se = np.sqrt(pn * (1 - pn) / R)
print("Simulated P(tau <= 400) =\t", np.round(pn, 4))
print("Standard error =\t\t", np.round(se, 4))
print(
    "Confidence interval:\t\t (",
    np.round(pn - 1.96 * se, 4),
    ",",
    np.round(pn + 1.96 * se, 4),
    ")",
)

# Plot
ns = np.arange(1, n + 1)
exact = np.array([tb[k] for k in ns])  # tb is a dict
se_n = np.sqrt(p * (1 - p) / R)  # standard error at every n

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ns, exact, color="black", label="Exact (recursion)")
ax.plot(ns, p, color="tab:blue", linestyle="--", label="Simulated")
ax.fill_between(
    ns,
    p - 1.96 * se_n,
    p + 1.96 * se_n,
    color="tab:blue",
    alpha=0.2,
    label="Simulated 95% CI",
)
ax.set_xscale("log")
ax.set_xlabel("Number of S trips $n$ (log scale)")
ax.set_ylabel(r"$\Pr(\tau \leq n)$")
ax.set_title(f"Optimistic operator, Beta({alpha}, {beta}), $q = {q}$")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(outpath / "hw3_problem2_partb.png", bbox_inches="tight")
plt.show()

# %%
# ===========================
print("Part 2.c - A committed experiment")
# ---------------------------
from hw3_strait import experiment

# experiment parameters
alpha, beta = 2, 4
q = 0.05
R = 10_000
T = 400

# loop m in the experiment
rows = []
for m in range(40):
    pL_next, tbT, EL, ES = experiment(alpha, beta, m, q, T)
    rows.append({"m": m, "p_Lnext": pL_next, "tbT": tbT, "EL": EL, "ES": ES})
    if tbT < 0.005:
        print(f"m = {m}: tbT = {tbT:.4f} < 0.005, stopping early")
        break

df = pl.DataFrame(rows)
small_m = df.filter(pl.col("tbT") < 0.02).select(pl.col("m")).min().to_numpy()[0][0]
small_m_tbT = df.filter(pl.col("m") == small_m).select(pl.col("tbT")).to_numpy()[0][0]
pL_mplus1 = df.filter(pl.col("m") == small_m).select(pl.col("p_Lnext")).to_numpy()[0][0]
print("Smallest m such that tbT < 0.02:", small_m)
print("P(tau <= 400) for that m:", np.round(small_m_tbT, 4))
print("P(take L on m+1):", np.round(pL_mplus1, 4))

# %%
# ===========================
print("Part 2.d.ii - Exploration rules")
# ---------------------------
from hw3_strait import gaps, simulate, Experiment, Thompson, Optimism

T = 1_000
R = 10_000
explorations = {
    "Myopic": (Myopic, {"a": 1, "b": 5}),
    "Experiment": (Experiment, {"a": 2, "b": 4, "m": 15}),
    "Thompson": (Thompson, {"a": 1, "b": 5}),
    "Optimism": (Optimism, {}),
}
rows = []
for exploration, (cls, args) in explorations.items():
    constructor = lambda R, rng: cls(**args, R=R, rng=rng)  # noqa: B023
    for q in [0.05, 0.35]:
        sim = simulate(constructor, np.ones(T) * q, R, 1)
        gS, gL = gaps(q)  # regret gaps
        if exploration == "Myopic":
            __, EL = recursion(**args, q=q, T=T)
        elif exploration == "Experiment":
            __, __, EL, __ = experiment(**args, q=q, T=T)
        else:
            EL = None
        exact = None if EL is None else EL * gL + (T - EL) * gS
        rows.append(
            {
                "exploration": exploration,
                "q": q,
                "mean": sim["total"],
                "se": sim["se_total"],
                "exact": exact,
            }
        )
df = pl.DataFrame(rows)
df.to_pandas().to_latex(
    outpath / "hw3_problem2_partd.tex",
    index=False,
    float_format="%.3f",
    na_rep="--",
)
print(df)

# %%
# ---------------------------
print("Part 2.d.iii - Exploration rules")
# ---------------------------
T = 20_000
rows = []
for exploration, (cls, args) in explorations.items():
    constructor = lambda R, rng: cls(**args, R=R, rng=rng)  # noqa: B023
    for q in [0.05, 0.35]:
        sim = simulate(constructor, np.ones(T) * q, R, 1)
        rows.append(
            {
                "exploration": exploration,
                "q": q,
                "cum_regret": sim["cum_regret"],
            }
        )
df = pl.DataFrame(rows)
for q in [0.05, 0.35]:
    fig, ax = plt.subplots(figsize=(6, 4))
    for exploration in explorations:
        cum_regret = (
            df.filter((pl.col("q") == q) & (pl.col("exploration") == exploration))
            .select(pl.col("cum_regret"))
            .to_numpy()[0][0]
        )
        ax.plot(
            np.arange(1, len(cum_regret) + 1),
            cum_regret,
            alpha=0.6,
            linewidth=1.5,
            label=f"{exploration} ({cum_regret[-1]:.1f})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("trip $T$ (log scale)")
    ax.set_ylabel("cumulative regret $R_T$")
    ax.set_title(f"Cumulative regret for $q = {q}$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath / f"hw3_problem2_partd_q{int(q * 100)}.png", dpi=300)


# %%
# ===========================
print("Part 2.e - The hit rate falls")
# ---------------------------
from hw3_strait import plot_share

T = 1_500
R = 10_000
explorations = {
    "Myopic": (Myopic, {"a": 1, "b": 5}),
    "Thompson": (Thompson, {"a": 1, "b": 5}),
    "Optimism": (Optimism, {}),
}
qT = np.ones(T)
qT[:300] *= 0.5
qT[300:] *= 0.05
shares = {}
for exploration, (cls, args) in explorations.items():
    constructor = lambda R, rng: cls(**args, R=R, rng=rng)  # noqa: B023
    sim = simulate(constructor, qT, R, 1)
    shares[exploration] = sim["share_S"]

plot_share(shares, outpath / "hw3_problem2_parte.png")

# %%
# ===========================
# Part 2 - Run self_check
print("Part 2 - Run self_check")
# ---------------------------
from hw3_strait import self_check

self_check()

# %%
# ===========================
print("Problem 3 - Planning with a model")
# ===========================
print("Part 3.c.ii - Exact values expected costs")
# ---------------------------
print("Part 3.d.i - Rollout exact costs")
# ---------------------------
from hw3_shoot_look_shoot import (
    SMALL,
    LARGE,
    DOCTRINES,
    value_tables,
    start,
    rollout_from_tables,
)

doctrines_plus = {**DOCTRINES.copy(), "minimum": None}
rows = []
for r, raid in {"small": SMALL, "large": LARGE}.items():
    N, S, K = start(raid)
    for doctrine, policy in doctrines_plus.items():
        tables = value_tables(raid, policy)
        rollout_policy = rollout_from_tables(tables)
        rollout_tables = value_tables(raid, rollout_policy)
        ec = tables[K][N + (S,)]
        rollout_ec = rollout_tables[K][N + (S,)]
        rows.append(
            {
                "raid": r,
                "doctrine": doctrine,
                "expected cost": ec,
                "from rollout": rollout_ec,
            }
        )

df = pl.DataFrame(rows).drop("from rollout").sort(by="expected cost", descending=False)
df.to_pandas().to_latex(
    outpath / "hw3_problem3_partc_subpartii.tex",
    index=False,
    float_format="%.3f",
    na_rep="--",
)
print(df)

df = pl.DataFrame(rows).sort(by="expected cost", descending=False)
df.to_pandas().to_latex(
    outpath / "hw3_problem3_partd_subparti.tex",
    index=False,
    float_format="%.3f",
    na_rep="--",
)
print(df)

# %%
# ===========================
print("Part 3.d.ii - Iterated fire-everything rollout")
# ---------------------------
eps = 1e-9
n_rounds = 10

rows = []
for r, raid in {"small": SMALL, "large": LARGE}.items():
    N, S, K = start(raid)
    policy = DOCTRINES["fire everything now"]
    raid_rows = []
    for rnd in tqdm(range(n_rounds + 1)):
        tables = value_tables(raid, policy)
        ec = tables[K][N + (S,)]
        if raid_rows and np.abs(ec - raid_rows[-1]["expected cost"]) < eps:
            break
        raid_rows.append({"raid": r, "round": rnd, "expected cost": ec})
        policy = rollout_from_tables(tables)
    rows.extend(raid_rows)
df = pl.DataFrame(rows)
df.to_pandas().to_latex(
    outpath / "hw3_problem3_partd_subpartii.tex",
    index=False,
    float_format="%.3f",
    na_rep="--",
)
print(df)


# %%
# ===========================
print("Part 3 - Run self_check")
# ---------------------------
from hw3_shoot_look_shoot import self_check

self_check()

# %%
