"""Discrete-time Signal Temporal Logic with quantitative (robustness)
semantics, on numpy arrays: the operators responsibility/right_of_way.py
writes its traffic rules in.

A signal is an array [T] of robustness values over the steps of a clip
(0.1 s apart): positive where the property holds, negative where it fails,
its magnitude how far from the boundary (in the predicate's unit, metres in
right_of_way.py). Every operator returns the robustness of the composed
formula at every step, so formulas nest: ``always(implies(p, q))[0]`` is the
robustness of G(p -> q) over the whole clip. The semantics are the standard
ones (Donze & Maler 2010) on a finite trace:

    not          -r
    and / or     pointwise min / max
    implies      max(-p, q)
    prev         the value one step earlier; ``first`` at step 0 (no
                 earlier step is known)
    always       G phi at t: min over [t, end] (with a window [a, b] in
                 steps: over [t + a, t + b], cut at the end of the trace)
    eventually   F phi at t: max over the same range
    until        phi U psi at t: max over t' >= t of
                 min(psi(t'), min over [t, t') of phi), psi must occur
    weak_until   phi W psi: phi U psi, or phi for the rest of the trace

An empty minimum is +inf and an empty maximum -inf (a vacuously true
G, a never satisfied F), as in the Boolean semantics. The traces here are
at most 91 steps long, so the quadratic loops cost nothing.
"""

from typing import Optional

import numpy as np

INF = np.inf


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def not_(x) -> np.ndarray:
    return -_arr(x)


def and_(*xs) -> np.ndarray:
    return np.minimum.reduce([_arr(x) for x in xs])


def or_(*xs) -> np.ndarray:
    return np.maximum.reduce([_arr(x) for x in xs])


def implies(p, q) -> np.ndarray:
    return np.maximum(-_arr(p), _arr(q))


def prev(x, first: float = -INF) -> np.ndarray:
    x = _arr(x)
    return np.concatenate([[first], x[:-1]])


def _window(t: int, n: int, a: int, b: Optional[int]):
    return t + a, n if b is None else min(n, t + b + 1)


def always(x, a: int = 0, b: Optional[int] = None) -> np.ndarray:
    x = _arr(x)
    n = len(x)
    out = np.empty(n)
    for t in range(n):
        lo, hi = _window(t, n, a, b)
        out[t] = x[lo:hi].min() if lo < hi else INF
    return out


def eventually(x, a: int = 0, b: Optional[int] = None) -> np.ndarray:
    x = _arr(x)
    n = len(x)
    out = np.empty(n)
    for t in range(n):
        lo, hi = _window(t, n, a, b)
        out[t] = x[lo:hi].max() if lo < hi else -INF
    return out


def until(phi, psi) -> np.ndarray:
    phi, psi = _arr(phi), _arr(psi)
    n = len(phi)
    out = np.full(n, -INF)
    for t in range(n):
        held = INF  # min of phi over [t, t')
        for s in range(t, n):
            out[t] = max(out[t], min(psi[s], held))
            held = min(held, phi[s])
    return out


def weak_until(phi, psi) -> np.ndarray:
    return np.maximum(until(phi, psi), always(phi))
