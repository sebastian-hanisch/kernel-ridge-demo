"""Zufallsquelle (Kopier-Test gegen die Vorlage), Plan-Erzeuger und Container-Erzeuger: eingefrorene Werte und gemessene Eigenschaften als Band.

Der Container-Erzeuger ist nach der Beschreibung der Quelle selbst gebaut (SplitMix64): gleiche Verteilung, keine Bitgleichheit; es gibt deshalb keinen Kopier-Test gegen die Quelle,
sondern Tests gegen die Eigenschaften (Boden, Anteile, Form der Merkmale)."""

import math
from pathlib import Path

import numpy as np
import pytest

import krd_constants as C
import krd_container as CT
import krd_scenario as S
from krd_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


def test_krd_rng_ist_byte_gleiche_kopie_der_vorlage():
    if not TEMPLATE.exists():
        pytest.skip("Vorlage nicht im Nachbarordner (z. B. in der CI): die Kopie wird dort nicht geprüft")
    assert (ROOT / "krd_rng.py").read_bytes() == TEMPLATE.read_bytes()


def test_splitmix64_bekannte_werte():
    r = SplitMix64(0)
    assert r.next_u64() == 0xE220A8397B1DCDAF and r.next_u64() == 0x6E789E6AA1B965F4          # veröffentlichte Referenzfolge von SplitMix64 mit Zufalls-Seed 0
    a, b = SplitMix64(7), SplitMix64(7)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_plan_erzeuger_eingefrorene_erste_werte_und_form():
    X, y, f = S.make_plan(5, 5400)
    assert X.shape == (5, 2) and (X >= -2).all() and (X <= 2).all()
    assert np.allclose(f, S.target_v1(X))
    X2, y2, _ = S.make_plan(5, 5400)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)                 # gleicher Zufalls-Seed, gleiche Daten
    assert abs(X[0, 0] - 0.42939579847062737) < 1e-12 and abs(y[0] - 0.4878464915146144) < 1e-9


def test_plan_erzeuger_eigenschaften_als_band():
    Xt, ft = S.make_test(20000, 5301)
    assert abs(Xt.mean()) < 0.05 and abs(Xt.std() - 4 / math.sqrt(12)) < 0.03              # gleichverteilt in [-2, 2]
    assert abs(S.variance_of_target() - 0.79) < 0.02                                       # Fehler eines konstanten Mittelwerts
    res = []
    for s in range(30):
        X, y, f = S.make_plan(100, 9000 + s)
        res.append(np.mean((y - f) ** 2))
    assert abs(np.mean(res) - C.NOISE_VAR) < 4.5 * np.std(res, ddof=1) / math.sqrt(30)    # Rauschvarianz 0,09


def test_kreis_erzeuger_halbe_flaeche_und_vertauschungsanteil():
    X, y = S.make_circle(20000, 3)
    inside = (S.circle_truth(X) == 1).mean()
    assert abs(inside - 0.5) < 0.02
    assert abs((y != S.circle_truth(X)).mean() - 0.1) < 0.01


def test_container_boden_exakt_und_stichprobenstreuung_als_band():
    assert abs(CT.floor_exact() - 0.48606) < 5e-6
    sq = []
    for s in range(20):
        _, _, _, ss = CT.make_container(4000, 700000 + s)
        sq.append(math.sqrt(np.mean(ss ** 2)))
    assert abs(np.mean(sq) - CT.floor_exact()) < 4.5 * np.std(sq, ddof=1) / math.sqrt(20)


def test_container_anteile_merkmale_und_verteilung():
    X, d, mu, s = CT.make_container(40000, 1)
    share = [(X[:, 0] == a).mean() for a in range(4)]
    assert all(abs(share[a] - C.ART_P[a]) < 0.01 for a in range(4))
    assert set(np.unique(X[:, 1])) == set(range(12)) and abs(((X[:, 0] == 0) & (X[:, 2] == 1)).mean() - 0.35 * 0.25) < 0.01
    assert (X[X[:, 0] == 3, 4] == 0).all() and X[X[:, 0] == 1, 4].min() >= 1.0 and X[X[:, 0] == 2, 4].max() <= 6.0
    assert abs(X[:, 5].mean()) < 0.03 and abs(X[:, 5].std() - 1) < 0.03                    # Rauschmerkmal
    z = (np.log(d) - mu) / s
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03                                # lognormal, bedingte Streuung wie im Orakel
    assert 3.0 < np.median(d) < 6.0                                                        # Median der Verweilzeit in Tagen (Band)
    assert abs(np.std(CT.kunden_effekte())) < 0.5 and len(CT.kunden_effekte()) == 12


def test_container_encode_hat_20_spalten_und_one_hot():
    X, _, _, _ = CT.make_container(50, 2)
    Z = CT.encode(X)
    assert Z.shape == (50, C.N_CONT_COLS) and (Z[:, :4].sum(1) == 1).all() and (Z[:, 4:16].sum(1) == 1).all()


def test_container_orakel_mittel_stimmt_mit_der_formel():
    mu, s = CT.log_median_and_s(0, 0, 1, 1, 0.0)
    assert abs(mu - (math.log(3.5 * math.exp(0.7 + 0.3)) + CT.kunden_effekte()[0] * 1.0)) < 1e-12 and s == 0.55
    mu, s = CT.log_median_and_s(1, 3, 0, 0, 4.0)
    assert abs(mu - (math.log(0.6 + 0.85 * 4.0) + CT.kunden_effekte()[3] * 0.3)) < 1e-12 and s == 0.25
    assert abs(CT.log_median_and_s(3, 5, 0, 1, 0.0)[0] - (math.log(7.0 * math.exp(0.3)) + CT.kunden_effekte()[5])) < 1e-12


def test_prior_ziehung_hat_varianz_eins_als_band():
    vals = []
    for s in range(60):
        _, _, _, ft, _ = S.draw_from_prior(10, 60, 2000 + s)
        vals.append(np.mean(ft ** 2))
    assert abs(np.mean(vals) - 1.0) < 4.5 * np.std(vals, ddof=1) / math.sqrt(60)
