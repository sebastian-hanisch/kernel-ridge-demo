"""Exakte Rechnung: Bareiss-Referenz für das Gleitkomma-System und das Handbeispiel in Brüchen.

Bareiss: löst A x = b mit ganzen Zahlen (A und b sind Gleitkommazahlen, also exakte Dualbrüche) und liefert die auf Gleitkomma gerundete exakte Lösung. Das ist
die Referenz der Wege-Anzeige (kein Weg): sie rechnet dasselbe System wie die Löser, nur ohne Rundung. Live bis n <= 40 (krd_constants.BAREISS_LIMIT).
Handbeispiel: drei Punkte x = (-1, 0, 1), y = (1, 0, 1), Polynomkern vom Grad 2 k(x, x') = (x x' + 1)^2 (ganzzahlige Gram-Matrix, deshalb alles in Brüchen;
der Gauß-Kern wäre nicht rational)."""

import math
from fractions import Fraction as F

import numpy as np


def exact_solve(A, b):
    """Löst A x = b exakt (fraktionsfreies Eliminieren nach Bareiss mit ganzen Zahlen, Rücksubstitution mit Brüchen); gibt x als float-Array.
    Wirft ZeroDivisionError bei singulärem A (z. B. lambda = 0 mit doppelten Punkten)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    fr = [[F(float(A[i, j])) for j in range(n)] + [F(float(b[i]))] for i in range(n)]
    den = 1
    for row in fr:
        for v in row:
            if v.denominator > den:
                den = v.denominator
    M = [[int(v * den) for v in row] for row in fr]
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = next((r for r in range(k + 1, n) if M[r][k] != 0), None)
            if sw is None:
                raise ZeroDivisionError("singulär")
            M[k], M[sw] = M[sw], M[k]
        pk = M[k][k]
        rk = M[k]
        for i in range(k + 1, n):
            ri = M[i]
            f = ri[k]
            for j in range(k + 1, n + 1):
                ri[j] = (ri[j] * pk - f * rk[j]) // prev
            ri[k] = 0
        prev = pk
    if M[n - 1][n - 1] == 0:
        raise ZeroDivisionError("singulär")
    x = [F(0)] * n
    for i in range(n - 1, -1, -1):
        s = F(M[i][n])
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        if M[i][i] == 0:
            raise ZeroDivisionError("singulär")
        x[i] = s / M[i][i]
    return np.array([float(v) for v in x])


# ---------------------------------------------------------------- Brüche: Gauß-Jordan
def fsolve(A, b):
    """Gauß-Jordan in Brüchen: löst A x = b (b ein Vektor oder eine Matrix als Liste von Zeilen)."""
    n = len(A)
    cols = len(b[0]) if isinstance(b[0], (list, tuple)) else 1
    M = [[F(A[i][j]) for j in range(n)] + ([F(v) for v in b[i]] if cols > 1 else [F(b[i])]) for i in range(n)]
    for k in range(n):
        p = next(r for r in range(k, n) if M[r][k] != 0)
        M[k], M[p] = M[p], M[k]
        piv = M[k][k]
        M[k] = [v / piv for v in M[k]]
        for r in range(n):
            if r != k and M[r][k] != 0:
                f = M[r][k]
                M[r] = [a - f * c for a, c in zip(M[r], M[k])]
    if cols > 1:
        return [row[n:] for row in M]
    return [row[n] for row in M]


def finv(A):
    n = len(A)
    return fsolve(A, [[F(int(i == j)) for j in range(n)] for i in range(n)])


def matvec(A, v):
    return [sum(F(a) * F(x) for a, x in zip(row, v)) for row in A]


def dot(u, v):
    return sum(F(a) * F(b) for a, b in zip(u, v))


def det(A):
    """Determinante in Brüchen (Laplace-Entwicklung, für kleine Matrizen)."""
    n = len(A)
    if n == 1:
        return F(A[0][0])
    return sum((-1) ** j * F(A[0][j]) * det([row[:j] + row[j + 1:] for row in A[1:]]) for j in range(n))


# ---------------------------------------------------------------- Handbeispiel
HAND_X = (F(-1), F(0), F(1))
HAND_Y = (F(1), F(0), F(1))


def kpoly(x, z):
    """Polynomkern vom Grad 2: (x z + 1)^2."""
    return (F(x) * F(z) + 1) ** 2


def hand_gram(xs=HAND_X):
    return [[kpoly(a, b) for b in xs] for a in xs]


def hand_krr(xs, ys, lam, xstar):
    """Duale Form in Brüchen: Mittel f(x*) und Posterior-Varianz der Funktion an x*, plus alpha."""
    K = hand_gram(xs)
    A = [[K[i][j] + (F(lam) if i == j else 0) for j in range(len(xs))] for i in range(len(xs))]
    al = fsolve(A, list(ys))
    ks = [kpoly(xstar, b) for b in xs]
    v = fsolve(A, ks)
    return dot(ks, al), kpoly(xstar, xstar) - dot(ks, v), al


def hand_example(lam=F(1), xstar=F(2)):
    """Das ganze Handbeispiel in Brüchen: K, det K, det(K + lam I), alpha, f(x*), Varianz(x*), Leave-one-out (geschlossen und durch echtes Herausnehmen),
    Hat-Diagonale, Spur, PRESS, GCV, y'alpha, log p(y) (Gleitkomma aus den exakten Brüchen), profilierter Skalenfaktor."""
    lam, xstar = F(lam), F(xstar)
    n = len(HAND_X)
    K = hand_gram()
    A = [[K[i][j] + (lam if i == j else 0) for j in range(n)] for i in range(n)]
    Ainv = finv(A)
    al = fsolve(A, list(HAND_Y))
    f_star, v_star, _ = hand_krr(HAND_X, HAND_Y, lam, xstar)
    e_closed = [al[i] / Ainv[i][i] for i in range(n)]
    v_closed = [1 / Ainv[i][i] for i in range(n)]
    e_real, v_real = [], []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        f_i, v_i, _ = hand_krr([HAND_X[j] for j in keep], [HAND_Y[j] for j in keep], lam, HAND_X[i])
        e_real.append(HAND_Y[i] - f_i)
        v_real.append(v_i + lam)
    H = [[F(int(i == j)) - lam * Ainv[i][j] for j in range(n)] for i in range(n)]
    trace = sum(H[i][i] for i in range(n))
    resid = [HAND_Y[i] - dot(K[i], al) for i in range(n)]
    press = sum(e * e for e in e_closed) / n
    gcv = (sum(r * r for r in resid) / n) / (1 - trace / n) ** 2
    quad = dot(HAND_Y, al)
    detA = det(A)
    logp = -0.5 * float(quad) - 0.5 * math.log(float(detA)) - 0.5 * n * math.log(2.0 * math.pi)
    return {
        "K": K, "det_K": det(K), "det_A": detA, "Ainv": Ainv, "alpha": al, "f_star": f_star, "var_star": v_star, "prior_var": kpoly(xstar, xstar),
        "loo_e": e_closed, "loo_var": v_closed, "loo_e_real": e_real, "loo_var_real": v_real, "hat_diag": [H[i][i] for i in range(n)], "trace": trace,
        "press": press, "gcv": gcv, "quad": quad, "log_p": logp, "s2_hat": quad / n, "fitted": matvec(K, al),
    }


def hand_pm1(lam=F(1)):
    """Dieselben drei Punkte mit den Etiketten (+1, -1, +1): alpha in Brüchen (alle ungleich null) und Werte an den Daten."""
    K = hand_gram()
    A = [[K[i][j] + (F(lam) if i == j else 0) for j in range(3)] for i in range(3)]
    al = fsolve(A, [F(1), F(-1), F(1)])
    return al, matvec(K, al)
