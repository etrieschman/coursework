# %%
import numpy as np
import cvxpy as cp
import polars as pl
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
pl.Config.set_tbl_cols(-1)

outpath = Path(__file__).resolve().parent.parent / "overleaf" / "6_S974"
outpath.parent.mkdir(parents=True, exist_ok=True)


# %%
# ===========================
# Problem 1 - Fictitious play
print("Problem 1 - Fictitious play")
# ===========================
# Part 1.a - Zero-sum NE
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
# Part 1.c - Fictitious play
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
# Experiment I
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
# Experiment II
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
# Experiment III
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
# Problem 2 - Beliefs and failures
print("Problem 2 - Beliefs and failures")
# ===========================
# Part 2.b.iii - Probability of switching
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
# Part 2.c - A committed experiment
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
df


# %%
