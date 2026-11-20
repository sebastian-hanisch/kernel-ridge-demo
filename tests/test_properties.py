"""Eigenschaftstests (je 20 Zufallsinstanzen, feste Zufalls-Seeds): Aussagen, die für jede Instanz gelten müssen."""

import numpy as np
import pytest

import krd_constants as C
import krd_model as M
import krd_scenario as S

SEEDS = range(20)


def _inst(seed, n=15, sigma=1.0):
    X, y, _ = S.make_plan(n, 700 + seed)
    return X, y, M.gram(X, X, sigma)


@pytest.mark.parametrize("seed", SEEDS)
def test_skalierung_c_k_mit_rauschvarianz_gleich_lambda(seed):
    """GP-Mittel mit Prior-Varianz c und Rauschvarianz s gleich Kernel Ridge mit lambda = s / c."""
    X, y, K = _inst(seed)
    Xs, _ = S.make_test(10, 900 + seed)
    Ks = M.gram(Xs, X, 1.0)
    c, s = 3.0, 0.7
    gp = (c * Ks) @ np.linalg.solve(c * K + s * np.eye(15), y)
    kr = Ks @ np.linalg.solve(K + (s / c) * np.eye(15), y)
    assert np.abs(gp - kr).max() < 1e-10


@pytest.mark.parametrize("seed", SEEDS)
def test_lambda_gross_gibt_prior_mittel_und_zentriert_das_trainingsmittel(seed):
    X, y, K = _inst(seed)
    Xs = np.array([[15.0, 15.0], [-16.0, 12.0]])                          # weit weg von den Daten
    Ks = M.gram(Xs, X, 1.0)
    m_zero, _ = M.posterior(K, Ks, y, 1e12, 1.0, center=False)
    m_cent, _ = M.posterior(K, Ks, y, 1e12, 1.0, center=True)
    assert np.abs(m_zero).max() < 1e-9 and np.abs(m_cent - y.mean()).max() < 1e-9
    m_far, _ = M.posterior(K, Ks, y, 0.1, 1.0, center=False)
    assert np.abs(m_far).max() < 1e-9                                    # Gauß-Kern: fern der Daten Rückfall auf den Prior 0


@pytest.mark.parametrize("seed", SEEDS)
def test_lambda_klein_interpoliert_bei_gut_gestellter_matrix(seed):
    X, y, K = _inst(seed, n=8, sigma=0.8)
    a = M.solve_eig(K, y, 1e-9)
    assert np.abs(K @ a - y).max() < 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_permutation_der_datenpunkte_aendert_nichts(seed):
    X, y, K = _inst(seed)
    perm = np.random.default_rng(seed).permutation(len(y))
    Xs, _ = S.make_test(10, 1000 + seed)
    m1, v1 = M.posterior(K, M.gram(Xs, X, 1.0), y, 0.1)
    m2, v2 = M.posterior(M.gram(X[perm], X[perm], 1.0), M.gram(Xs, X[perm], 1.0), y[perm], 0.1)
    assert np.abs(m1 - m2).max() < 1e-10 and np.abs(v1 - v2).max() < 1e-10
    e1 = M.loo_closed(K, y - y.mean(), 0.1)[0]
    e2 = M.loo_closed(M.gram(X[perm], X[perm], 1.0), (y - y.mean())[perm], 0.1)[0]
    assert np.abs(e1[perm] - e2).max() < 1e-9                             # Leave-one-out-Residuum unabhängig von der Zeilenreihenfolge


@pytest.mark.parametrize("seed", SEEDS)
def test_posterior_varianz_haengt_nicht_von_y_ab_und_liegt_zwischen_0_und_prior(seed):
    X, y, K = _inst(seed)
    Xs, _ = S.make_test(20, 1100 + seed)
    Ks = M.gram(Xs, X, 1.0)
    _, v1 = M.posterior(K, Ks, y, 0.2)
    _, v2 = M.posterior(K, Ks, 5.0 * y + 3.0, 0.2)
    assert np.abs(v1 - v2).max() < 1e-12
    assert v1.min() > 0 and v1.max() <= 1.0 + 1e-12
    _, v_small = M.posterior(K, Ks, y, 0.02)
    assert np.all(v_small <= v1 + 1e-12)                                  # kleineres lambda: weniger Varianz (an denselben Punkten)


@pytest.mark.parametrize("seed", SEEDS)
def test_hat_matrix_symmetrisch_eigenwerte_in_0_1_und_spur_gleich_effektive_parameterzahl(seed):
    X, y, K = _inst(seed)
    lam = 0.3
    H = K @ np.linalg.inv(K + lam * np.eye(15))
    assert np.abs(H - H.T).max() < 1e-10
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    assert ev.min() > -1e-10 and ev.max() < 1.0
    assert abs(np.trace(H) - M.hat_trace(K, lam)) < 1e-10


@pytest.mark.parametrize("seed", SEEDS)
def test_verschiebung_aller_y_um_d_verschiebt_das_zentrierte_mittel_um_d(seed):
    X, y, K = _inst(seed)
    Xs, _ = S.make_test(10, 1200 + seed)
    Ks = M.gram(Xs, X, 1.0)
    m1, _ = M.posterior(K, Ks, y, 0.1, center=True)
    m2, _ = M.posterior(K, Ks, y + 2.5, 0.1, center=True)
    assert np.abs(m2 - m1 - 2.5).max() < 1e-10


def test_mehr_trainingsdaten_senken_im_mittel_ueber_zufalls_seeds_den_testfehler():
    Xt, ft = S.make_test(300, 5301)
    def err(n):
        out = []
        for s in range(15):
            X, y, _ = S.make_plan(n, 3000 + s)
            K = M.gram(X, X, 1.0)
            m, _ = M.posterior(K, M.gram(Xt, X, 1.0), y, 0.1)
            out.append(np.mean((m - ft) ** 2))
        return np.mean(out)
    assert err(20) > err(60) > err(150)


def test_regler_grenzen_sind_konsistent():
    assert C.N_W[0] <= C.N_W[2] <= C.N_W[1] <= C.BAREISS_LIMIT
    assert C.LAM_EXP[0] <= C.LAM_EXP[2] <= C.LAM_EXP[1] and C.SIGMA[0] <= C.SIGMA[2] <= C.SIGMA[1]
    assert C.N_BAND_DEFAULT in C.N_BAND_OPTIONS and C.HAND_X_DEFAULT in C.HAND_X_OPTIONS
