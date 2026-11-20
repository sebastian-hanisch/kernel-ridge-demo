"""Orakel-Tests (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12): jede Kernroutine gegen einen ANDEREN Rechenweg.

(a) Gewichtsraum gegen Dualform (Polynomkern Grad 3, Merkmalsdimension 10), (b) numerische Minimierung gegen die Gleichung (nur bei gut gestellter Matrix, Band 1e-5),
(c) echtes Herausnehmen gegen die geschlossene Leave-one-out-Formel (Residuum, Mittel, Varianz, 20 Instanzen), (d) Evidenz gegen scipy-Dichte und Gauß-Hermite (n <= 3),
(e) Spur H gegen Eigenwerte, (f) Bareiss in ganzen Zahlen gegen Brüche, (g) scikit-learn als Fremdreferenz (dieselbe Gleichung in anderen Programmen, kein Beweis),
(h) Kalibrierungs-Orakel (Wahrheit aus dem Prior: Abdeckung 0,95 und z^2 gleich 1 als Band mit Standardfehler), (j) Gradient der Evidenz gegen Differenzenquotient."""

import math
from itertools import combinations_with_replacement

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import krd_constants as C
import krd_evaluation as EV
import krd_exact as E
import krd_model as M
import krd_scenario as S


def _inst(n, seed, sigma=1.0):
    X, y, _ = S.make_plan(n, seed)
    return X, y - y.mean(), M.gram(X, X, sigma)


# ---- (a) Gewichtsraum -------------------------------------------------------------------------------------------------------------------------
def _poly3_features(X):
    """Explizite Merkmale des Polynomkerns (x.x' + 1)^3 in 2 Dimensionen: alle Monome vom Grad <= 3 mit Multinomialgewicht, Dimension 10."""
    cols = []
    for deg in range(4):
        for idx in combinations_with_replacement(range(2), deg):
            mono = np.prod([X[:, i] for i in idx], axis=0) if idx else np.ones(len(X))
            counts = [idx.count(0), idx.count(1)]
            coef = math.factorial(3) / (math.factorial(3 - deg) * math.prod(math.factorial(c) for c in counts))   # 3! / ((3-deg)! n0! n1!) * 1^(3-deg)
            cols.append(math.sqrt(coef) * mono)
    return np.column_stack(cols)


def test_gewichtsraum_gleich_dualform_bei_polynomkern_grad_3():
    X, y, _ = S.make_plan(30, 1)
    Xs, _ = S.make_test(50, 2)
    X, Xs = X / 2.0, Xs / 2.0
    lam = 0.3
    Phi, Phis = _poly3_features(X), _poly3_features(Xs)
    assert Phi.shape[1] == 10
    assert np.abs(Phi @ Phi.T - (X @ X.T + 1.0) ** 3).max() < 1e-10             # die Merkmale erzeugen den Kern
    K = (X @ X.T + 1.0) ** 3
    ks = (Xs @ X.T + 1.0) ** 3
    dual = ks @ np.linalg.solve(K + lam * np.eye(30), y)
    w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(10), Phi.T @ y)
    assert np.abs(Phis @ w - dual).max() < 1e-8
    var_dual = np.einsum("ij,ij->i", Phis, Phis) - np.einsum("ij,ji->i", ks, np.linalg.solve(K + lam * np.eye(30), ks.T))
    var_w = lam * np.einsum("ij,jk,ik->i", Phis, np.linalg.inv(Phi.T @ Phi + lam * np.eye(10)), Phis)
    assert np.abs(var_dual - var_w).max() < 1e-8


# ---- (b) Minimierung --------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_minimierung_gleich_gleichung_bei_gut_gestellter_matrix(seed):
    X, y, K = _inst(30, seed)
    lam = 0.1                                                                  # cond klein; bei sigma = 4, lambda = 1e-10 ist die Minimierung nur auf 1e1 genau
    a_min, _ = M.solve_minimize(K, y, lam, 20000)
    a_eq = np.linalg.solve(K + lam * np.eye(30), y)
    assert np.abs(K @ (a_min - a_eq)).max() < 1e-5


# ---- (c) Leave-one-out ------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(20))
def test_loo_geschlossen_gleich_echtes_herausnehmen(seed):
    rng = np.random.default_rng(seed)             # nur im Test: Parameter der Instanz; die Daten selbst kommen aus SplitMix64
    n = int(rng.integers(6, 31))
    sg, lm = float(np.exp(rng.uniform(np.log(0.3), np.log(3)))), float(np.exp(rng.uniform(np.log(1e-3), 0)))
    X, y, K = _inst(n, 100 + seed, sg)
    e_c, mu_c, v_c = M.loo_closed(K, y, lm)
    e_r, mu_r, v_r = M.loo_by_removal(K, y, lm)
    sc = max(1.0, np.abs(e_r).max())
    assert np.abs(e_c - e_r).max() / sc < 1e-7
    assert np.abs(mu_c - mu_r).max() / sc < 1e-7
    assert np.abs(v_c - v_r).max() / max(1.0, np.abs(v_r).max()) < 1e-7


# ---- (d) Evidenz ------------------------------------------------------------------------------------------------------------------------------
def test_evidenz_gleich_scipy_dichte_und_skalenfaktor_ist_optimal():
    X, y, _ = S.make_plan(40, 5400)
    yc = y - y.mean()
    sg, lm = 1.3, 0.05
    le, s2 = M.evidence_at(X, y, sg, lm)
    K = M.gram(X, X, sg)
    assert abs(multivariate_normal(mean=np.zeros(40), cov=s2 * (K + lm * np.eye(40))).logpdf(yc) - le) < 1e-8
    for fac in (0.5, 0.9, 1.1, 2.0):
        assert multivariate_normal(mean=np.zeros(40), cov=fac * s2 * (K + lm * np.eye(40))).logpdf(yc) < le


def test_evidenz_gleich_gauss_hermite_bei_n_gleich_2():
    X = np.array([[0.0, 0.0], [0.7, -0.3]])
    y = np.array([0.4, -0.2])
    K = M.gram(X, X, 1.0)
    lam = 0.5
    logp = multivariate_normal(mean=np.zeros(2), cov=K + lam * np.eye(2)).logpdf(y)          # p(y) = Integral p(y|f) p(f) df
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / math.sqrt(2 * math.pi)
    g = np.meshgrid(nodes, nodes, indexing="ij")
    Z = np.stack([a.ravel() for a in g], axis=1)
    W = np.outer(weights, weights).ravel()
    F = Z @ np.linalg.cholesky(K).T
    lik = np.exp(-0.5 * ((y[None, :] - F) ** 2).sum(1) / lam) / (2 * math.pi * lam)
    assert abs(math.log(float((W * lik).sum())) - logp) < 1e-8


# ---- (e) Spur H -------------------------------------------------------------------------------------------------------------------------------
def test_spur_h_gleich_summe_eigenwerte_und_hat_diagonale_aus_inverse():
    X, y, K = _inst(25, 3)
    lam = 0.2
    H = K @ np.linalg.inv(K + lam * np.eye(25))
    assert abs(M.hat_trace(K, lam) - np.trace(H)) < 1e-10
    assert np.abs(M.eig_parts(K, y, lam)[1] - np.diag(H)).max() < 1e-10


# ---- (f) Bareiss -----------------------------------------------------------------------------------------------------------------------------
def test_bareiss_gleich_bruchrechnung_und_loest_das_system_exakt():
    X, y, K = _inst(8, 9)
    A = K + 1e-6 * np.eye(8)
    x = E.exact_solve(A, y)
    x_fr = E.fsolve([[A[i][j] for j in range(8)] for i in range(8)], [float(v) for v in y])
    assert np.abs(x - np.array([float(v) for v in x_fr])).max() < 1e-15 * max(1.0, np.abs(x).max())
    assert np.abs(A @ x - y).max() < 1e-12


# ---- (g) scikit-learn als Fremdreferenz ------------------------------------------------------------------------------------------------------
def test_sklearn_kernelridge_und_gp_geben_dieselbe_vorhersage_wie_die_gleichung():
    """Fremdreferenz, ausdrücklich dieselbe Gleichung in anderen Programmen (kein unabhängiger Beweis)."""
    pytest.importorskip("sklearn")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.kernel_ridge import KernelRidge
    X, y, K = _inst(30, 7)
    Xs, _ = S.make_test(60, 8)
    Ks = M.gram(Xs, X, 1.0)
    lam = 0.1
    mine = Ks @ np.linalg.solve(K + lam * np.eye(30), y)
    assert np.abs(KernelRidge(alpha=lam, kernel="rbf", gamma=0.5).fit(X, y).predict(Xs) - mine).max() < 1e-9
    gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed"), alpha=lam, optimizer=None).fit(X, y)
    m, sd = gp.predict(Xs, return_std=True)
    assert np.abs(m - mine).max() < 1e-9
    _, var = M.posterior(K, Ks, y, lam, 1.0, center=False)
    assert np.abs(sd ** 2 - var).max() < 1e-6                      # Varianz der Funktion (ohne Rauschen) gegen die Streuung von sklearn


# ---- (h) Kalibrierung ------------------------------------------------------------------------------------------------------------------------
def test_wahrheit_aus_dem_prior_gibt_abdeckung_095_und_z2_gleich_1():
    res = [EV.prior_oracle(30000 + s) for s in range(100)]
    cov = np.array([r["coverage"] for r in res])
    z2 = np.array([r["z2"] for r in res])
    assert abs(cov.mean() - 0.95) < 4.5 * cov.std(ddof=1) / math.sqrt(100)
    assert abs(z2.mean() - 1.0) < 4.5 * z2.std(ddof=1) / math.sqrt(100)


# ---- (j) Gradient der Evidenz -----------------------------------------------------------------------------------------------------------------
def test_gradient_der_evidenz_gleich_differenzenquotient_und_optimum_nicht_niedriger_als_raster():
    X, y, _ = S.make_plan(40, 5400)
    yc = y - y.mean()
    D2 = M.sqdist_exact(X, X)
    for th in ([0.0, math.log(0.1)], [math.log(0.4), math.log(1e-3)], [math.log(2.5), math.log(1e-2)]):
        th = np.array(th)
        g = M.logev_and_grad(th, D2, yc)[1]
        gn = np.array([(M.logev_and_grad(th + e, D2, yc)[0] - M.logev_and_grad(th - e, D2, yc)[0]) / 2e-5 for e in (np.array([1e-5, 0.0]), np.array([0.0, 1e-5]))])
        assert np.max(np.abs(g - gn) / (1 + np.abs(gn))) < 1e-6
    for seed in range(5):
        X, y, _ = S.make_plan(60, 5400 + seed)
        t = M.cell_table(X, y, C.SIGMAS_A, C.LAMS_A)
        *_, le, _, _ = M.pick_gradient(X, y)
        assert le >= float(t["logev"].max()) - 1e-6
