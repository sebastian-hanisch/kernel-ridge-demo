"""Handbeispiel von Hand (drei Punkte, Polynomkern Grad 2, alles in Brüchen) und seine unabhängigen Rechenwege.

Die Werte sind von Hand gerechnet (K = [[4,1,0],[1,1,1],[0,1,4]], det K = 8, det(K + I) = 40, alpha aus 5a + b = 1 und b = -a), nicht aus dem Code abgeschrieben."""

import math
from fractions import Fraction as F

import numpy as np
import pytest
from scipy import integrate, optimize
from scipy.stats import multivariate_normal

import krd_exact as E


def test_gram_und_determinanten(hand):
    assert hand["K"] == [[4, 1, 0], [1, 1, 1], [0, 1, 4]]
    assert hand["det_K"] == 8 and hand["det_A"] == 40


def test_inverse_und_alpha(hand):
    assert hand["Ainv"] == [[F(9, 40), F(-5, 40), F(1, 40)], [F(-5, 40), F(25, 40), F(-5, 40)], [F(1, 40), F(-5, 40), F(9, 40)]]
    assert hand["alpha"] == [F(1, 4), F(-1, 4), F(1, 4)]
    assert all(a != 0 for a in hand["alpha"])            # keine Stützvektoren, allgemeine Lage


def test_vorhersage_und_varianz_an_zwei_testpunkten(hand):
    assert (hand["f_star"], hand["var_star"], hand["prior_var"]) == (F(9, 4), F(319, 40), 25)
    h = E.hand_example(1, F(1, 2))
    assert (h["f_star"], h["var_star"], h["prior_var"]) == (F(3, 8), F(61, 160), F(25, 16))
    assert hand["fitted"] == [F(3, 4), F(1, 4), F(3, 4)]   # Schrumpfung an den Daten: y - lambda alpha


def test_gewichtsraum_ohne_gram_matrix_gibt_dieselben_brueche():
    """Unabhängiger Weg: Merkmale (1, x, x^2) mit Prior-Gewichtsmatrix diag(1, 2, 1) (psi' S psi' = (x x' + 1)^2), keine Gram-Matrix."""
    S = [F(1), F(2), F(1)]
    Psi = [[F(1), x, x * x] for x in E.HAND_X]
    PtP = [[sum(Psi[r][i] * Psi[r][j] for r in range(3)) for j in range(3)] for i in range(3)]
    Pty = [sum(Psi[r][i] * E.HAND_Y[r] for r in range(3)) for i in range(3)]
    for x, f_exp, v_exp in ((F(2), F(9, 4), F(319, 40)), (F(1, 2), F(3, 8), F(61, 160))):
        psi = [F(1), x, x * x]
        w = E.fsolve([[PtP[i][j] + (F(1) / S[i] if i == j else 0) for j in range(3)] for i in range(3)], Pty)
        assert E.dot(psi, w) == f_exp
        vv = E.fsolve([[PtP[i][j] + (1 / S[i] if i == j else 0) for j in range(3)] for i in range(3)], psi)
        assert E.dot(psi, vv) == v_exp


def test_numerische_minimierung_gibt_alpha_auf_1e7(hand):
    Kf = np.array([[float(v) for v in r] for r in hand["K"]])
    yf = np.array([float(v) for v in E.HAND_Y])
    r = optimize.minimize(lambda a: float(np.sum((yf - Kf @ a) ** 2) + a @ Kf @ a), np.zeros(3), jac=lambda a: -2 * Kf @ (yf - Kf @ a) + 2 * Kf @ a,
                          method="L-BFGS-B", options={"ftol": 1e-18, "gtol": 1e-13, "maxiter": 1000})
    assert np.abs(r.x - np.array([0.25, -0.25, 0.25])).max() < 1e-6      # Referenz nie ungenauer als die Toleranz (Orakel-Playbook §2)


def test_sklearn_ist_fremdreferenz_fuer_dieselbe_gleichung():
    pytest.importorskip("sklearn")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct
    from sklearn.kernel_ridge import KernelRidge
    Xa = np.array([[-1.0], [0.0], [1.0]])
    yf = np.array([1.0, 0.0, 1.0])
    xs = np.array([[2.0], [0.5]])
    kr = KernelRidge(alpha=1.0, kernel="polynomial", degree=2, gamma=1.0, coef0=1.0).fit(Xa, yf)
    assert np.abs(kr.predict(xs) - np.array([2.25, 0.375])).max() < 1e-12
    gp = GaussianProcessRegressor(kernel=DotProduct(sigma_0=1.0) ** 2, alpha=1.0, optimizer=None).fit(Xa, yf)
    m, sd = gp.predict(xs, return_std=True)
    assert np.abs(m - np.array([2.25, 0.375])).max() < 1e-9
    assert np.abs(sd - np.sqrt([319 / 40, 61 / 160])).max() < 1e-8


def test_leave_one_out_geschlossen_und_durch_herausnehmen_gleiche_brueche(hand):
    assert hand["loo_e"] == [F(10, 9), F(-2, 5), F(10, 9)]
    assert hand["loo_var"] == [F(40, 9), F(8, 5), F(40, 9)]
    assert hand["loo_e_real"] == hand["loo_e"] and hand["loo_var_real"] == hand["loo_var"]
    # Beispiel von Hand für Punkt 1: f_-1(x_1) = -1/9, e_1 = 10/9, Varianz 5 - 5/9 = 40/9
    f1, v1, _ = E.hand_krr([E.HAND_X[1], E.HAND_X[2]], [E.HAND_Y[1], E.HAND_Y[2]], 1, E.HAND_X[0])
    assert f1 == F(-1, 9) and v1 + 1 == F(40, 9)


def test_hat_spur_press_gcv(hand):
    assert hand["hat_diag"] == [F(31, 40), F(15, 40), F(31, 40)] and hand["trace"] == F(77, 40)
    ev = np.linalg.eigvalsh(np.array([[float(v) for v in r] for r in hand["K"]]))
    assert abs(float((ev / (ev + 1.0)).sum()) - 77 / 40) < 1e-12     # Spur H = Summe lambda_i / (lambda_i + 1), Eigenzerlegung als anderer Weg
    assert hand["press"] == F(5324, 6075) and hand["gcv"] == F(900, 1849)
    assert abs(float(hand["gcv"]) - float(hand["press"])) > 0.3      # n = 3: H_ii ~ Spur / n gilt nicht; keine Aussage über größere n


def test_randwahrscheinlichkeit_gegen_dichte_und_integration(hand):
    assert hand["quad"] == F(1, 2) and hand["s2_hat"] == F(1, 6)
    logp = -0.25 - 0.5 * math.log(40.0) - 1.5 * math.log(2 * math.pi)
    assert abs(hand["log_p"] - logp) < 1e-12 and abs(hand["log_p"] + 4.8512) < 5e-4
    Kf = np.array([[float(v) for v in r] for r in hand["K"]])
    yf = np.array([1.0, 0.0, 1.0])
    assert abs(multivariate_normal(mean=np.zeros(3), cov=Kf + np.eye(3)).logpdf(yf) - hand["log_p"]) < 1e-10
    Lc = np.linalg.cholesky(Kf)
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)       # Gauß-Hermite, 80^3 = 512 000 Auswertungen
    weights = weights / math.sqrt(2 * math.pi)
    g = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    Z = np.stack([a.ravel() for a in g], axis=1)
    W = (weights[:, None, None] * weights[None, :, None] * weights[None, None, :]).ravel()
    lik = np.exp(-0.5 * ((yf[None, :] - Z @ Lc.T) ** 2).sum(1)) / (2 * math.pi) ** 1.5
    assert abs(math.log(float((W * lik).sum())) - hand["log_p"]) < 1e-9
    p1 = integrate.quad(lambda f: math.exp(-0.5 * (1 - f) ** 2) / math.sqrt(2 * math.pi) * math.exp(-f * f / 8.0) / math.sqrt(2 * math.pi * 4.0), -40, 40, epsabs=1e-14, epsrel=1e-12)[0]
    assert abs(math.log(p1) + 1.8237) < 5e-4                       # n = 1: N(1; 0, 5)
    lpf = lambda s: multivariate_normal(mean=np.zeros(3), cov=s * (Kf + np.eye(3))).logpdf(yf)  # noqa: E731
    assert all(lpf(1 / 6) > lpf(s) for s in (1 / 12, 1 / 8, 1 / 4, 1 / 3, 1.0))


def test_konvention_mu_gleich_lambda_durch_n_und_skala_c():
    K = E.hand_gram()
    Kf = np.array([[float(v) for v in r] for r in K])
    yf = np.array([1.0, 0.0, 1.0])
    mu = 1.0 / 3
    r = optimize.minimize(lambda a: float(np.mean((yf - Kf @ a) ** 2) + mu * a @ Kf @ a), np.zeros(3), jac=lambda a: -2.0 / 3 * Kf @ (yf - Kf @ a) + 2 * mu * Kf @ a,
                          method="L-BFGS-B", options={"ftol": 1e-18, "gtol": 1e-13, "maxiter": 1000})
    assert np.abs(r.x - np.array([0.25, -0.25, 0.25])).max() < 1e-6
    c, s2 = F(4), F(36, 100)
    A2 = [[c * K[i][j] + (s2 if i == j else 0) for j in range(3)] for i in range(3)]
    ks = [E.kpoly(F(2), b) for b in E.HAND_X]
    f_gp = E.dot([c * k for k in ks], E.fsolve(A2, list(E.HAND_Y)))
    assert f_gp == E.hand_krr(E.HAND_X, E.HAND_Y, s2 / c, F(2))[0]
    assert E.hand_krr(E.HAND_X, E.HAND_Y, s2, F(2))[0] != f_gp      # dasselbe Zahlenpaar ohne c: anderes Mittel


def test_pm1_etiketten_alpha_und_harte_marge_von_hand():
    al, fit = E.hand_pm1()
    assert al == [F(3, 8), F(-7, 8), F(3, 8)] and fit == [F(5, 8), F(-1, 8), F(5, 8)]
    K = E.hand_gram()
    c, b = [F(1), F(-2), F(1)], F(-1)                                # harte Marge von Hand: alle drei Bedingungen aktiv, alpha = (1, 2, 1)
    assert [s + b for s in E.matvec(K, c)] == [F(1), F(-1), F(1)] and sum(c) == 0
    sk = pytest.importorskip("sklearn.svm")
    sv = sk.SVC(kernel="poly", degree=2, gamma=1.0, coef0=1.0, C=1e4).fit(np.array([[-1.0], [0.0], [1.0]]), [1, -1, 1])
    dual = np.zeros(3)
    dual[sv.support_] = sv.dual_coef_[0]
    assert len(sv.support_) == 3 and np.allclose(dual, [1.0, -2.0, 1.0], atol=2e-3) and abs(float(sv.intercept_[0]) + 1.0) < 2e-3
