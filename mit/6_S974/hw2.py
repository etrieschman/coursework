# %%
import numpy as np
import cvxpy as cp
import polars as pl

pl.Config.set_tbl_cols(-1)


# %%
# =======================
# Problem 3 Part (f)
# =======================
def private_recs(p):
    # variables
    q0 = cp.Variable()
    q1 = cp.Variable()
    # constraints
    const = [
        q0 >= 0,
        q0 <= 1,
        q1 >= 0,
        q1 <= 1,
    ]
    const += [
        (q1 / 3 + 2 * cp.square(q1)) * p + (q0 / 3 + 2 * cp.square(q0)) * (1 - p)
        <= q1 * p
    ]
    const += [
        (1 / 3 + 5 * q1 / 3 - 2 * cp.square(q1)) * p
        + (1 / 3 + 5 * q0 / 3 - 2 * cp.square(q0)) * (1 - p)
        >= (1 - q1) * p,
    ]
    # objective
    obj = (1 - p) * (2 * cp.square(q0) + q0 / 3) + p * (
        2 * cp.square(q1) - 2 * q1 / 3 + 1
    )
    # problem
    prob = cp.Problem(cp.Minimize(obj), const)
    prob.solve()
    return float(q0.value), float(q1.value), float(prob.value), prob.status


results = []
for p in np.arange(1, 11):
    q0, q1, obj, status = private_recs(p / 11)
    public_cost = p / 11
    soc_opt_cost = (p / 11) * (17 / 18)
    public_gain = public_cost - obj
    soc_opt_gap = obj - soc_opt_cost
    results.append(
        (
            f"{p}/11",
            round(q0, 4),
            round(q1, 4),
            round(obj, 4),
            round(public_cost, 4),
            round(public_gain, 4),
            round(soc_opt_cost, 4),
            round(soc_opt_gap, 4),
        )
    )


df = pl.DataFrame(
    results,
    orient="row",
    schema=[
        "p",
        "rule_q0",
        "rule_q1",
        "expected_cost",
        "public_cost",
        "public_gain",
        "soc_opt_cost",
        "soc_opt_gap",
    ],
)
from pathlib import Path

out = (
    Path(__file__).resolve().parent.parent
    / "overleaf"
    / "6_S974"
    / "hw2_problem3_partf.tex"
)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(df.to_pandas().to_latex(index=False, escape=True, float_format="%.4f"))

# %%
