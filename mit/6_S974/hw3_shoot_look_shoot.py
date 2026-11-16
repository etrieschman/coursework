"""Shoot, look, shoot: doctrines and exact values.

Sections:
  1. The model
  2. The placement rule
  3. Doctrines
  4. Salvo plans
  5. Exact values

The state is the number of live missiles of each type, the interceptors left
and the salvos left. A salvo plan is a pair (fire, spread): how many
interceptors to fire now, and whether to put at most one on each missile.

Usage examples:
  benchmark(LARGE)                       best policy over salvo plans
  benchmark(LARGE, equal_share_spread)   exact cost of a doctrine
  value_tables(LARGE, equal_share_spread)  all value tables of a doctrine
  rollout_from_tables(value_tables(LARGE, fire_everything_now))
                                         rollout of a doctrine, exact scores
  self_check()                           tests your functions

You complete three functions marked TODO: place and value_tables for part (c),
and rollout_from_tables for part (d). Run self_check() after each one; it uses
a small test raid that is not in the problem set.
"""
import itertools
import numpy as np

# ---------------------------------------------------------------- 1. The model
TYPES = ("H", "M", "L")
VALUE = {"H": 10, "M": 3, "L": 1}       # value of the asset each missile threatens
KILL = {"H": 0.5, "M": 0.7, "L": 0.85}  # one interceptor kills the missile w.p. KILL

LARGE = {"missiles": (5, 10, 15), "interceptors": 36, "salvos": 3}
SMALL = {"missiles": (1, 2, 3), "interceptors": 7, "salvos": 3}


def start(raid):
    return raid["missiles"], raid["interceptors"], raid["salvos"]


def cost(live):
    """Total value of the missiles still alive."""
    return sum(VALUE[t] * n for t, n in zip(TYPES, live))


# ---------------------------------------------------------- 2. The placement rule

def place(live, fire, spread):
    """Assign `fire` shots one at a time. Each goes to the missile where a shot
    saves the most: value x kill probability x chance the missile survives the
    shots already on it. With spread=True, at most one shot per missile; shots
    that cannot be placed are not fired. Within a type, shots go to the
    least-shot missile, so they spread evenly. Ties go to the earlier type
    (H, then M, then L). Returns the number of shots on each type."""
    shots = [0, 0, 0]
    # TODO: add `fire` shots one at a time. For each type with live missiles,
    # the least-shot missile has shots[i] // live[i] shots. Skip the type if
    # spread is True and that number is already 1. Give the shot to the type
    # with the largest gain (use a strict >, so ties go to the earlier type).
    # Stop early if no type can take a shot.
    raise NotImplementedError
    return shots

def split(n, x):
    """x shots spread evenly over n missiles: how many get r+1 shots, how many get r."""
    if n == 0:
        return []
    r, extra = divmod(x, n)
    return [(extra, r + 1), (n - extra, r)]


# -------------------------------------------------------------------- 3. Doctrines
# A doctrine is a fixed rule for choosing a salvo plan. In the last salvo every
# doctrine fires everything it has left.

def fire_everything_now(live, s, k):
    return (s, False)

def one_per_missile(live, s, k):
    return (s, False) if k == 1 else (min(s, sum(live)), True)

def equal_share_concentrated(live, s, k):
    return (s, False) if k == 1 else (s // k, False)

def equal_share_spread(live, s, k):
    return (s, False) if k == 1 else (s // k, True)

DOCTRINES = {
    "fire everything now": fire_everything_now,
    "one per missile per salvo": one_per_missile,
    "equal share, concentrated": equal_share_concentrated,
    "equal share, spread": equal_share_spread,
}


# ------------------------------------------------------------- 4. Salvo plans

def salvo_plans(s, k):
    """The candidates: every salvo plan. In the last salvo, fire everything."""
    if k == 1:
        return [(s, False)]
    return [(fire, spread) for fire in range(s + 1) for spread in (True, False)]


# ------------------------------------------------------------- 5. Exact values

def survivor_distribution(n, x, t):
    """Probability that j of n missiles of type t survive x shots, for j = 0..n."""
    dist = np.array([1.0])
    for count, r in split(n, x):
        survive = (1 - KILL[t]) ** r
        for _ in range(count):
            dist = np.convolve(dist, [1 - survive, survive])
    return dist


def expected_cost(V_next, live, s, plan):
    """Expected value of V_next at the state reached by firing `plan` from (live, s).
    V_next[h, m, l, s'] is the cost-to-go with one salvo fewer."""
    shots = place(live, *plan)
    dH, dM, dL = (survivor_distribution(live[i], shots[i], t) for i, t in enumerate(TYPES))
    after = V_next[:live[0] + 1, :live[1] + 1, :live[2] + 1, s - sum(shots)]
    return float(np.einsum("i,j,k,ijk->", dH, dM, dL, after))


def value_tables(raid, policy=None):
    """Value iteration on counts. Returns [V_0, ..., V_K]; V_k[h, m, l, s] is the
    expected cost with k salvos left. With policy None, V_k is the minimum over
    salvo plans. Otherwise it is the cost of following `policy`.
    With K salvos left, only the start state is computed."""
    N, S, K = raid["missiles"], raid["interceptors"], raid["salvos"]
    V = np.zeros(tuple(n + 1 for n in N) + (S + 1,))
    for live in itertools.product(*(range(n + 1) for n in N)):
        V[live] = cost(live)                     # k = 0: the raid is over
    tables = [V]
    for k in range(1, K + 1):
        V = np.full(V.shape, np.nan)
        # TODO: for every count vector `live` and every s in 0..S (only the start
        # state N, S when k == K), set V[live + (s,)] using expected_cost and
        # tables[k - 1]: the minimum over salvo_plans(s, k) when policy is None,
        # and the value of the plan policy(live, s, k) otherwise.
        raise NotImplementedError
        tables.append(V)
    return tables

def benchmark(raid, policy=None):
    """Exact cost from the start: of the best policy over salvo plans (policy None),
    or of a given policy."""
    N, S, K = start(raid)
    return value_tables(raid, policy)[K][N + (S,)]


def rollout_from_tables(tables):
    """Rollout with exact scores. `tables` is value_tables(raid, base) for a base
    doctrine. Returns a policy: a function (live, s, k) -> salvo plan that fires
    the plan with the lowest expected cost when the base is followed afterwards."""
    def policy(live, s, k):
        # TODO (part d): score every plan in salvo_plans(s, k) with expected_cost
        # and tables[k - 1], and return the plan with the lowest score.
        raise NotImplementedError
    return policy


# ------------------------------------------------------------------ Self check
# The checks use a test raid that does not appear in the problem set.
TEST = {"missiles": (1, 1, 2), "interceptors": 4, "salvos": 2}


def self_check():
    """Tests place, value_tables and rollout_from_tables."""
    def check(name, test):
        try:
            test()
            print(f"{name}: ok")
        except NotImplementedError:
            print(f"{name}: not done yet")
        except AssertionError as err:
            print(f"{name}: WRONG {err}")

    def test_place():
        for args, want in [(((2, 3, 4), 5, True), [2, 3, 0]), (((2, 3, 4), 5, False), [4, 1, 0]),
                           (((2, 1, 0), 6, True), [2, 1, 0]), (((0, 3, 4), 9, False), [0, 5, 4])]:
            got = place(*args)
            assert got == want, f"place{args} gave {got}, expected {want}"

    def test_values():
        expected = {"fire everything now": 4.15, "one per missile per salvo": 6.2,
                    "equal share, concentrated": 3.4125, "equal share, spread": 3.1125}
        for name, doctrine in DOCTRINES.items():
            got = benchmark(TEST, doctrine)
            assert abs(got - expected[name]) < 1e-6, f"{name}: got {got:.4f}, expected {expected[name]}"
        got = benchmark(TEST)
        assert abs(got - 3.1125) < 1e-6, f"best policy: got {got:.4f}, expected 3.1125"

    def test_rollout():
        policy = rollout_from_tables(value_tables(TEST, fire_everything_now))
        got = benchmark(TEST, policy)
        assert abs(got - 3.1125) < 1e-6, f"rollout of fire everything now: got {got:.4f}, expected 3.1125"

    check("place", test_place)
    check("value_tables and benchmark", test_values)
    check("rollout_from_tables", test_rollout)


if __name__ == "__main__":
    self_check()
