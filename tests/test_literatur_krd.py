"""Literatur (Stufe W: Rasmussen/Williams 2006, ganzes Buch offen lesbar): nur Gleichungsidentitäten, keine Zahlenwerte aus Tabellen.

Gl. 2.23 gleich Gl. 6.20 mit lambda = sigma_n^2 / c (Mittel von GP und Kernel Ridge); Gl. 5.10 bis 5.12 (Leave-one-out aus K^-1) gleich echtes Herausnehmen. Nicht gelesen oder nicht lesbar
(Saunders/Gammerman/Vovk 1998, Kimeldorf/Wahba 1971, Wahba 1990, Stein 1999, Schölkopf/Smola): daraus gibt es keinen Test („Quelle nicht eingesehen heißt: kein Test“)."""

import numpy as np

import krd_model as M
import krd_scenario as S


def test_gl_2_23_gleich_gl_6_20_mit_lambda_gleich_rauschvarianz_durch_c():
    X, y, _ = S.make_plan(20, 31)
    Xs, _ = S.make_test(15, 32)
    K, Ks = M.gram(X, X, 1.3), M.gram(Xs, X, 1.3)
    c, s2n = 2.5, 0.4
    gp = c * Ks @ np.linalg.solve(c * K + s2n * np.eye(20), y)                  # Gl. 2.23 mit Prior-Varianz c
    ridge = Ks @ np.linalg.solve(K + (s2n / c) * np.eye(20), y)                  # Gl. 6.20
    assert np.abs(gp - ridge).max() < 1e-10


def test_gl_5_10_bis_5_12_gleich_echtes_herausnehmen():
    X, y, _ = S.make_plan(18, 33)
    yc = y - y.mean()
    K = M.gram(X, X, 1.0)
    lam = 0.2
    Kinv = np.linalg.inv(K + lam * np.eye(18))
    mu_511 = yc - (Kinv @ yc) / np.diag(Kinv)                                    # Mittel, Gl. 5.12
    var_511 = 1.0 / np.diag(Kinv)                                                # Varianz, Gl. 5.12
    e, mu, var = M.loo_by_removal(K, yc, lam)
    assert np.abs(mu_511 - mu).max() < 1e-9 and np.abs(var_511 - var).max() < 1e-9
    assert np.abs(e - (Kinv @ yc) / np.diag(Kinv)).max() < 1e-9                 # Residuum aus Gl. 5.10
