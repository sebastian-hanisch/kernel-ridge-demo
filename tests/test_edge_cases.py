"""Randfälle (DEMO-PLAYBOOK §12): n = 1, n = 2, y = 0, doppelte Punkte, lambda = 0, Regler an Minimum und Maximum, Rand des Rasters, Testpunkt weit weg."""

import numpy as np
import pytest

import krd_constants as C
import krd_evaluation as EV
import krd_exact as E
import krd_model as M
import krd_scenario as S


def test_n_gleich_1_und_n_gleich_2():
    for n in (1, 2):
        X, y, _ = S.make_plan(n, 5)
        K = M.gram(X, X, 1.0)
        a = M.solve_eig(K, y, 0.1)
        assert a.shape == (n,) and np.isfinite(a).all()
        assert np.allclose(a, np.linalg.solve(K + 0.1 * np.eye(n), y))
        e, mu, v = M.loo_closed(K, y, 0.1)
        assert np.isfinite(e).all() and (v > 0).all()
    X, y, _ = S.make_plan(2, 5)
    K = M.gram(X, X, 1.0)
    e_c = M.loo_closed(K, y, 0.1)[0]
    e_r = M.loo_by_removal(K, y, 0.1)[0]
    assert np.allclose(e_c, e_r)


def test_y_gleich_null_gibt_alpha_null_keine_stuetzvektoren_nicht_fuer_jedes_y():
    X, _, _ = S.make_plan(20, 6)
    K = M.gram(X, X, 1.0)
    a = np.linalg.solve(K + 0.1 * np.eye(20), np.zeros(20))
    assert EV.zero_count(a) == 20                                        # alle Koeffizienten null: die Aussage „keine Nullen“ gilt nur in allgemeiner Lage
    r = EV.ridge_pm1(50, 3)
    assert r["zeros"] == 0


def test_doppelte_punkte_lambda_rettet_und_lambda_null_wird_gemeldet():
    X = np.array([[0.5, 0.5], [0.5, 0.5], [1.0, -1.0]])
    y = np.array([1.0, 1.2, 0.3])
    K = M.gram(X, X, 1.0)
    a, note = M.run_way("cholesky", K, y, 0.1)
    assert a is not None and note is None
    a0, note0 = M.run_way("cholesky", K, y, 0.0)                         # K singulär: die Meldung statt einer Zahl
    assert a0 is None and note0 and note0.startswith("Ausnahme")
    with pytest.raises(ZeroDivisionError):
        E.exact_solve(K, y)                                              # Bareiss mit lambda = 0 und singulärer Matrix


def test_sehr_kleines_lambda_und_grosses_sigma_liefern_alle_wege_oder_meldungen():
    res = EV.wege_run(10, 6.0, 1e-10, 1, "v1", C.WAY_KEYS)
    assert res["cond"] > 1e8 and set(res["ways"]) == set(C.WAY_KEYS)


def test_regler_an_minimum_und_maximum_laufen_ohne_ausnahme():
    for n in (C.N_W[0], C.N_W[1]):
        for sigma in (C.SIGMA[0], C.SIGMA[1]):
            for le in (C.LAM_EXP[0], C.LAM_EXP[1]):
                r = EV.wege_run(n, sigma, 10.0 ** le, 3, "rau", C.WAY_KEYS)
                assert r["cond"] >= 1.0 and all(r["ways"][w]["note"] is None or isinstance(r["ways"][w]["note"], str) for w in C.WAY_KEYS)


def test_leave_one_out_bei_n_gleich_10_und_n_gleich_60_ohne_ausnahme():
    for n in (C.N_L[0], C.N_L[1]):
        chk = EV.loo_check(n, 1.0, 0.1, 4)
        assert chk["factorizations_removal"] == n and chk["dev_residual"] < 1e-7
        fl = EV.flaeche_run(n, 4)
        assert set(fl["picks"]) == {"loo", "gcv", "ev_grid", "ev_grad"}


def test_wahl_am_rand_des_rasters_wird_erkannt():
    X, y, _ = S.make_plan(20, 11)
    t = M.cell_table(X, y, C.SIGMAS_A, C.LAMS_A)
    i, j = M.pick_cell(t, "gcv")
    assert 0 <= i < len(C.SIGMAS_A) and 0 <= j < len(C.LAMS_A)
    sg, lm, _, _, edge, _ = M.pick_gradient(X, y)
    assert C.GRAD_SIGMA[0] <= sg <= C.GRAD_SIGMA[1] + 1e-9 and C.GRAD_LAM[0] <= lm <= C.GRAD_LAM[1] + 1e-9 and isinstance(edge, bool)


def test_testpunkt_weit_weg_varianz_gleich_prior_varianz():
    X, y, _ = S.make_plan(30, 12)
    K = M.gram(X, X, 1.0)
    far = np.array([[40.0, 40.0]])
    m, v = M.posterior(K, M.gram(far, X, 1.0), y, 0.1, 2.5, center=False)
    assert abs(m[0]) < 1e-12 and abs(v[0] - 2.5) < 1e-12


def test_keine_wege_gewaehlt_gibt_hinweis_statt_zahl():
    kind, text = EV.verdict_wege(10.0, {}, {}, None, C.WAY_LABELS)
    assert kind == "info" and "Kein direkter Weg" in text


def test_hand_beispiel_testpunkte_im_ganzen_regler_bereich():
    from fractions import Fraction
    for x in C.HAND_X_OPTIONS:
        h = E.hand_example(1, Fraction(str(x)))
        assert 0 < h["var_star"] <= h["prior_var"]
