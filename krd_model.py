"""Kern der Demo: Gram-Matrix, vier Rechenwege zu (K + lambda I) alpha = y, Posterior, Leave-one-out, Wahlregeln (Leave-one-out, GCV, Evidenz).

Alles hier ist numpy/scipy und in kleine Einheiten zerlegt (DEMO-PLAYBOOK §1b): jede Funktion hat Ein- und Ausgaben, keinen versteckten Zustand.
Konvention der Linie: Gauß-Kern k(x, x') = exp(-||x - x'||^2 / (2 sigma^2)) mit sigma = Bandbreite (Längenmaß, nicht gamma); lambda = Regularisierung von Kernel
Ridge; die Gleichung (K + lambda I) alpha = y und das GP-Mittel mit Prior-Varianz c und Rauschvarianz sigma_n^2 sind gleich, wenn lambda = sigma_n^2 / c.
Quellen der Gleichungen: Rasmussen/Williams 2006 (Gl. 2.23 bis 2.26 Posterior, 2.30 und 5.8 Randwahrscheinlichkeit, 5.10 bis 5.12 Leave-one-out, 6.19 und 6.20 Ridge)."""

import math
import warnings

import numpy as np
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve

import krd_constants as C


# ---------------------------------------------------------------- Gram-Matrix
def sqdist_exact(X, Y):
    """Quadrierte Abstände aus Differenzen (ohne Auslöschung), für kleine Dimensionen und schlecht konditionierte Fälle."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    D = X[:, None, :] - Y[None, :, :]
    return (D ** 2).sum(-1)


def sqdist_fast(X, Y):
    """Quadrierte Abstände über das Skalarprodukt (schnell, für viele Spalten; Rundung auf >= 0 begrenzt)."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    d = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    return np.maximum(d, 0.0)


def gram(X, Y, sigma, fast=False):
    """Gauß-Kern exp(-r^2 / (2 sigma^2)); sigma ist die Bandbreite (Längenmaß), gamma = 1 / (2 sigma^2) wird nur genannt."""
    d2 = sqdist_fast(X, Y) if fast else sqdist_exact(X, Y)
    return np.exp(-d2 / (2.0 * sigma ** 2))


# ---------------------------------------------------------------- vier Rechenwege
def solve_cholesky(A, y):
    """Weg 1: Cholesky-Zerlegung von A = K + lambda I. Wirft numpy.linalg.LinAlgError, wenn A numerisch nicht positiv definit ist."""
    try:
        return cho_solve(cho_factor(A), y)
    except Exception as ex:  # scipy meldet LinAlgError; hier einheitlich weiterreichen
        raise np.linalg.LinAlgError(str(ex)) from ex


def solve_lu(A, y):
    """Weg 2: LU-Zerlegung (numpy.linalg.solve)."""
    return np.linalg.solve(A, y)


def eig_parts(K, y, lam):
    """Eigenzerlegung von K (Eigenwerte auf >= 0 begrenzt): alpha = V diag(1/(L + lambda)) V' y und Diagonale der Hat-Matrix H = K (K + lambda I)^-1."""
    L, V = np.linalg.eigh(K)
    L = np.maximum(L, 0.0)
    c = V.T @ y
    alpha = V @ (c / (L + lam))
    Hd = (V * V) @ (L / (L + lam))
    return alpha, Hd


def solve_eig(K, y, lam):
    """Weg 3: Eigenzerlegung."""
    return eig_parts(K, y, lam)[0]


def solve_minimize(K, y, lam, maxiter=5000):
    """Weg 4: numerische Minimierung des regularisierten Risikos sum (y - K a)^2 + lambda a'K a mit analytischem Gradienten (kein Gleichungslöser).
    Rückgabe (alpha, Info) mit Info = {'success', 'message', 'nit'}; nur diesen Weg meldet scipy, wenn er abbricht."""
    n = len(y)

    def obj(a):
        return float(np.sum((y - K @ a) ** 2) + lam * a @ K @ a)

    def grd(a):
        return -2.0 * K @ (y - K @ a) + 2.0 * lam * K @ a

    r = optimize.minimize(obj, np.zeros(n), jac=grd, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-18, "gtol": 1e-12, "maxcor": 50})
    msg = (r.message.decode() if isinstance(r.message, bytes) else str(r.message)).strip().rstrip(":").strip()
    return r.x, {"success": bool(r.success), "message": msg, "nit": int(r.nit)}


def run_way(name, K, y, lam, maxiter=5000):
    """Rechnet einen Weg und sammelt, was der Anwender sieht: (alpha oder None, Meldung oder None).
    Meldung = Text der Ausnahme, einer Warnung oder (nur Minimierung) des Abbruchstatus; None heißt: der Löser hat nichts gemeldet."""
    A = K + lam * np.eye(len(y))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            if name == "cholesky":
                alpha, note = solve_cholesky(A, y), None
            elif name == "lu":
                alpha, note = solve_lu(A, y), None
            elif name == "eig":
                alpha, note = solve_eig(K, y, lam), None
            elif name == "min":
                alpha, info = solve_minimize(K, y, lam, maxiter)
                note = None if info["success"] else info["message"]
            else:
                raise ValueError(name)
        except np.linalg.LinAlgError as ex:
            return None, f"Ausnahme: {str(ex)[:60]}"
    if note is None and wl:
        note = f"Warnung: {str(wl[0].message)[:60]}"
    return alpha, note


# ---------------------------------------------------------------- Vorhersage und Posterior
def predict_mean(Ks, alpha, ym=0.0):
    """Vorhersage f(x*) = k*' alpha (plus Trainingsmittel ym bei zentriertem Ziel)."""
    return Ks @ alpha + ym


def posterior(K, Ks, y, lam, s2=1.0, center=True):
    """GP-Posterior der FUNKTION f an den Testpunkten (Rasmussen/Williams Gl. 2.23 bis 2.26): Prior s2 * Gauß-Kern, Rauschvarianz lam * s2.
    Mittel k*' (K + lam I)^-1 y (bei center: Trainingsmittel vorher abgezogen und danach wieder addiert), Varianz s2 (1 - k*' (K + lam I)^-1 k*).
    Eigenzerlegung, deshalb auch bei kleinem lam ohne Ausnahme. Rückgabe (Mittel, Varianz); die Varianz ist nach unten bei 1e-300 begrenzt."""
    ym = float(np.mean(y)) if center else 0.0
    L, V = np.linalg.eigh(K)
    L = np.maximum(L, 0.0)
    W = Ks @ V
    m = W @ ((V.T @ (y - ym)) / (L + lam)) + ym
    var = s2 * (1.0 - np.sum(W * W / (L + lam), axis=1))
    return m, np.maximum(var, 1e-300)


def prior_variance(kss_diag=1.0, s2=1.0):
    """Prior-Varianz der Funktion an einem Punkt (Gauß-Kern: k(x, x) = 1)."""
    return s2 * kss_diag


# ---------------------------------------------------------------- Leave-one-out
def loo_closed(K, y, lam):
    """Leave-one-out ohne n-mal Rechnen (Rasmussen/Williams Gl. 5.10 bis 5.12 und Ridge-Residuum (y_i - f_i) / (1 - H_ii)).
    Rückgabe (Residuen e_i, Mittel y_i - e_i, Varianz der Vorhersage von y_i = 1 / [(K + lam I)^-1]_ii in Einheiten des Prior)."""
    L, V = np.linalg.eigh(K)
    L = np.maximum(L, 0.0)
    inv_diag = (V * V) @ (1.0 / (L + lam))             # Diagonale von (K + lam I)^-1
    alpha = V @ ((V.T @ y) / (L + lam))
    e = alpha / inv_diag
    return e, y - e, 1.0 / inv_diag


def loo_by_removal(K, y, lam):
    """Echtes Herausnehmen: n Zerlegungen der Größe n - 1 (Referenz für loo_closed). Rückgabe wie loo_closed."""
    n = len(y)
    e = np.zeros(n)
    mu = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):
        keep = np.delete(np.arange(n), i)
        cf = cho_factor(K[np.ix_(keep, keep)] + lam * np.eye(n - 1))
        ks = K[i, keep]
        mu[i] = ks @ cho_solve(cf, y[keep])
        var[i] = K[i, i] + lam - ks @ cho_solve(cf, ks)
        e[i] = y[i] - mu[i]
    return e, mu, var


def flops_chol(m):
    """Multiplikationen und Additionen einer Cholesky-Zerlegung der Größe m, in den Schleifen mitgezählt (Skalarprodukt der Länge j kostet 2 j)."""
    f = 0
    for j in range(m):
        f += 2 * j
        for _ in range(j + 1, m):
            f += 2 * j
    return f


def flops_tri_solve(m):
    """Vorwärts- oder Rückwärtssubstitution der Größe m (ohne Division)."""
    return sum(2 * i for i in range(m))


def flops_tri_inverse(m):
    """Dreiecksinverse der Größe m (für die Diagonale von K^-1)."""
    f = 0
    for j in range(m):
        for i in range(j + 1, m):
            f += 2 * (i - j)
    return f


def flops_loo_removal(n):
    """Echtes Herausnehmen: n mal (Cholesky der Größe n - 1 plus zwei Dreieckslösungen)."""
    return n * (flops_chol(n - 1) + 2 * flops_tri_solve(n - 1))


def flops_loo_closed(n):
    """Geschlossen: eine Cholesky-Zerlegung der Größe n plus Dreiecksinverse plus Quadratsummen der Spalten plus zwei Dreieckslösungen."""
    return flops_chol(n) + flops_tri_inverse(n) + n * (n + 1) + 2 * flops_tri_solve(n)


# ---------------------------------------------------------------- Wahl von (sigma, lambda)
def cell_table(X, y, sigmas, lams, Xte=None, fte=None):
    """Je Zelle (sigma, lambda): LOO-Fehler (PRESS / n), GCV, profilierte log-Evidenz, geschätzter Skalenfaktor s2, Spur H und, falls Testpunkte gegeben,
    der Überschussfehler gegen die rauschfreie Wahrheit fte. y wird um den Trainingsmittelwert zentriert. Eine Eigenzerlegung je sigma, alle lambda danach billig."""
    n = len(y)
    ym = float(np.mean(y))
    yc = y - ym
    shape = (len(sigmas), len(lams))
    res = {k: np.zeros(shape) for k in ("press", "gcv", "logev", "test", "sf2", "trh")}
    for i, sg in enumerate(sigmas):
        K = gram(X, X, sg)
        w, V = np.linalg.eigh(K)
        w = np.maximum(w, 0.0)
        c = V.T @ yc
        Kt = gram(Xte, X, sg) if Xte is not None else None
        V2 = V * V
        for j, lm in enumerate(lams):
            a = V @ (c / (w + lm))
            Hd = V2 @ (w / (w + lm))
            r = yc - K @ a
            with np.errstate(divide="ignore", invalid="ignore"):
                e = r / (1.0 - Hd)
            trh = float(Hd.sum())
            q = float(np.sum(c * c / (w + lm)))
            s2 = max(q / n, 1e-300)
            res["press"][i, j] = float(np.mean(e * e))
            res["gcv"][i, j] = float(np.mean(r * r) / (1.0 - trh / n) ** 2)
            res["trh"][i, j] = trh
            res["sf2"][i, j] = s2
            res["logev"][i, j] = -0.5 * n * math.log(2.0 * math.pi * s2) - 0.5 * float(np.sum(np.log(w + lm))) - 0.5 * n
            if Kt is not None:
                res["test"][i, j] = float(np.mean((Kt @ a + ym - fte) ** 2))
    return res


def pick_cell(table, rule):
    """Wahl der Zelle nach Regel: 'loo' (kleinster PRESS), 'gcv' (kleinster GCV), 'evidence' (größte profilierte log-Evidenz). Rückgabe (i, j)."""
    if rule == "loo":
        return tuple(int(v) for v in np.unravel_index(np.argmin(table["press"]), table["press"].shape))
    if rule == "gcv":
        return tuple(int(v) for v in np.unravel_index(np.argmin(table["gcv"]), table["gcv"].shape))
    if rule == "evidence":
        return tuple(int(v) for v in np.unravel_index(np.argmax(table["logev"]), table["logev"].shape))
    raise ValueError(rule)


def logev_and_grad(theta, D2, yc):
    """Profilierte log-Evidenz (Skalenfaktor s2 = y'A^-1 y / n; Rasmussen/Williams Gl. 2.30 und 5.8) und ihr Gradient nach (log sigma, log lambda); A = K + lambda I."""
    n = len(yc)
    sg, lm = math.exp(theta[0]), math.exp(theta[1])
    K = np.exp(-D2 / (2.0 * sg * sg))
    A = K + lm * np.eye(n)
    cf = cho_factor(A)
    al = cho_solve(cf, yc)
    q = float(yc @ al)
    s2 = q / n
    logdet = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
    le = -0.5 * n * math.log(2.0 * math.pi * s2) - 0.5 * logdet - 0.5 * n
    Ai = cho_solve(cf, np.eye(n))
    dK = K * D2 / (sg * sg)                                   # dK / d log sigma
    g_sig = 0.5 * n * float(al @ dK @ al) / q - 0.5 * float(np.sum(Ai * dK))
    g_lam = 0.5 * n * lm * float(al @ al) / q - 0.5 * lm * float(np.trace(Ai))
    return le, np.array([g_sig, g_lam]), s2


def pick_gradient(X, y):
    """Evidenz per Gradientenverfahren (L-BFGS-B auf (log sigma, log lambda), sechs Startpunkte, Grenzen sigma 0,1 bis 6,4 und lambda 1e-6 bis 10).
    Rückgabe (sigma, lambda, s2, log-Evidenz, am_Rand, Zahl_verschiedener_Optima)."""
    yc = y - y.mean()
    D2 = sqdist_exact(X, X)
    bnds = [(math.log(C.GRAD_SIGMA[0]), math.log(C.GRAD_SIGMA[1])), (math.log(C.GRAD_LAM[0]), math.log(C.GRAD_LAM[1]))]

    def fun(th):
        try:
            le, g, _ = logev_and_grad(th, D2, yc)
        except np.linalg.LinAlgError:
            return 1e30, np.zeros(2)
        return -le, -g

    best, vals = None, []
    for s0, l0 in C.GRAD_STARTS:
        r = optimize.minimize(fun, np.array([math.log(s0), math.log(l0)]), jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
        vals.append(-r.fun)
        if best is None or -r.fun > best[0]:
            best = (-r.fun, r.x)
    le, th = best
    _, _, s2 = logev_and_grad(th, D2, yc)
    edge = bool(any(abs(th[k] - bnds[k][b]) < 1e-6 for k in range(2) for b in range(2)))
    distinct = len({round(v, 2) for v in vals})
    return math.exp(th[0]), math.exp(th[1]), s2, le, edge, distinct


def evidence_at(X, y, sigma, lam):
    """Profilierte log-Evidenz und Skalenfaktor s2 an einer Stelle (sigma, lambda), Eigenzerlegung (robust)."""
    n = len(y)
    yc = y - y.mean()
    L, V = np.linalg.eigh(gram(X, X, sigma))
    L = np.maximum(L, 0.0)
    c = V.T @ yc
    q = float(np.sum(c * c / (L + lam)))
    s2 = max(q / n, 1e-300)
    return -0.5 * n * math.log(2.0 * math.pi * s2) - 0.5 * float(np.sum(np.log(L + lam))) - 0.5 * n, s2


def hat_trace(K, lam):
    """Spur der Hat-Matrix H = K (K + lambda I)^-1 = Summe lambda_i / (lambda_i + lambda): effektive Parameterzahl (Rasmussen/Williams §2.6)."""
    L = np.maximum(np.linalg.eigvalsh(K), 0.0)
    return float((L / (L + lam)).sum())


# ---------------------------------------------------------------- lineare Ridge (Gegner in Experiment 4)
def linear_ridge_loo(Z, yc, lams=C.LIN_LAMS):
    """Lineare Ridge (zentrierte Merkmale Z, zentriertes Ziel yc), lambda per geschlossenem Leave-one-out gewählt. Rückgabe (Gewichte w, gewähltes lambda)."""
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    best = (np.inf, None)
    for lm in lams:
        shrink = sv ** 2 / (sv ** 2 + lm)
        Hd = (U * U) @ shrink
        f = U @ (shrink * (U.T @ yc))
        e = (yc - f) / (1.0 - Hd)
        sc = float(np.mean(e * e))
        if sc < best[0]:
            best = (sc, lm)
    lm = best[1]
    w = Vt.T @ ((sv / (sv ** 2 + lm)) * (U.T @ yc))
    return w, lm


# ---------------------------------------------------------------- Kernel Ridge mit Wahl per Leave-one-out auf einem Raster (Experiment 4)
def krr_grid_loo(Z, yc, sigmas, lams, fast=False):
    """Kernel Ridge auf zentriertem Ziel yc; (sigma, lambda) aus sigmas x lams per geschlossenem Leave-one-out (kleinster PRESS) gewählt.
    Rückgabe (sigma, lambda, alpha, press). Eine Eigenzerlegung je sigma."""
    best = (np.inf, None)
    for sg in sigmas:
        K = gram(Z, Z, sg, fast=fast)
        w, V = np.linalg.eigh(K)
        w = np.maximum(w, 0.0)
        c = V.T @ yc
        V2 = V * V
        for lm in lams:
            a = V @ (c / (w + lm))
            Hd = V2 @ (w / (w + lm))
            r = yc - K @ a
            with np.errstate(divide="ignore", invalid="ignore"):
                e = r / (1.0 - Hd)
            sc = float(np.mean(e * e))
            if sc < best[0]:
                best = (sc, (sg, lm, a))
    sg, lm, a = best[1]
    return sg, lm, a, best[0]


def median_distance(Z, cap=800):
    """Median der paarweisen Abstände der Zeilen von Z (bei mehr als cap Zeilen cap Zeilen nach fester Mischung): Median-Heuristik sigma_Median."""
    if len(Z) > cap:
        from krd_rng import SplitMix64
        idx = list(range(len(Z)))
        SplitMix64(0).shuffle(idx)
        Z = Z[idx[:cap]]
    D = np.sqrt(sqdist_fast(Z, Z))
    iu = np.triu_indices(len(Z), 1)
    return float(np.median(D[iu]))


def standardize_pair(Xtr, Xte):
    """Merkmale mit den Momenten des Trainings standardisieren (Spalten ohne Streuung bleiben 0)."""
    m, s = Xtr.mean(0), Xtr.std(0)
    s = np.where(s < 1e-12, 1.0, s)
    return (Xtr - m) / s, (Xte - m) / s
