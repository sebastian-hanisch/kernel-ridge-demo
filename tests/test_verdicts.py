"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier; dazu je Preset die erwartete Verdikt-Zeile
(ein Preset ohne Wirkung meldet keine Wirkung)."""

from fractions import Fraction as F

import krd_constants as C
import krd_evaluation as EV
import krd_exact as E

L = C.WAY_LABELS


def test_wege_gut_gestellt_ist_success_und_nennt_zahlen():
    kind, text = EV.verdict_wege(87.0, {"cholesky": 1e-15, "lu": 2e-15, "eig": 5e-15}, {}, 3e-15, L)
    assert kind == "success" and "die 3 direkten Wege liegen höchstens 5,0e-15" in text and "cond(K + λI) = 87,0" in text and "keine Richtigkeit" in text


def test_wege_schwellen_sind_inklusiv_exklusiv_wie_dokumentiert():
    n = {}
    assert EV.verdict_wege(1.0, {"cholesky": 0.99e-8, "lu": 1e-9}, n, None, L)[0] == "success"       # unter 1e-8
    assert EV.verdict_wege(1.0, {"cholesky": 1e-8, "lu": 1e-9}, n, None, L)[0] == "info"              # ab 1e-8
    assert EV.verdict_wege(1.0, {"cholesky": 0.99e-6, "lu": 1e-9}, n, None, L)[0] == "info"
    assert EV.verdict_wege(1.0, {"cholesky": 1e-6, "lu": 1e-9}, n, None, L)[0] == "warning"           # ab 1e-6


def test_wege_schlecht_gestellt_sagt_dass_keiner_meldet_und_nennt_minimierung():
    kind, text = EV.verdict_wege(2.4e11, {"cholesky": 2e-5, "lu": 5e-6, "eig": 3.6e-3, "min": 11.0}, {"min": "ABNORMAL"}, 3.6e-3, L)
    assert kind == "warning" and "bis 3,6e-3" in text and "keiner meldet etwas" in text and "2,4e11" in text
    assert "Die Minimierung ist am ungenauesten (11,0)" in text and "meldet als einzige „ABNORMAL“" in text


def test_wege_meldung_eines_direkten_weges_wird_gezaehlt_und_ausnahme_ist_warning():
    kind, text = EV.verdict_wege(1e5, {"cholesky": 3e-7, "lu": 4e-7}, {"cholesky": "Warnung: x"}, None, L)
    assert kind == "info" and "1 von 2 ohne Meldung" in text
    kind, text = EV.verdict_wege(1e20, {"cholesky": None, "lu": 3e-3}, {"cholesky": "Ausnahme: nicht positiv definit"}, None, L)
    assert kind == "warning" and "Cholesky meldet eine Ausnahme" in text and "bis 3,0e-3" in text


def test_wege_band_nennt_spanne_laeufe_ohne_meldung_und_verhaeltnis():
    band = {"n_seeds": 20, "cond": (2.4e11, 2.6e11, 2.7e11), "err": {"cholesky": (5e-6, 1e-5, 1e-4), "eig": (4e-6, 1e-4, 2.6e-3), "min": (6.0, 15.0, 140.0)},
            "silent_runs": 20, "ratio": (0.63, 1.29, 1.65)}
    kind, text = EV.verdict_wege_band(band)
    assert kind == "warning" and "Über 20 Zufalls-Seeds" in text and "4,0e-6 bis 2,6e-3" in text and "In 20 von 20 Läufen" in text and "0.63 bis 1.65 (Median 1.29)" in text
    band["err"] = {"cholesky": (1e-15, 2e-15, 3e-15)}
    assert EV.verdict_wege_band(band)[0] == "success"


def test_hand_verdikt_innerhalb_und_ausserhalb():
    h = E.hand_example(1, 2)
    kind, text = EV.verdict_hand(h, 2.0)
    assert kind == "info" and "außerhalb der Daten" in text and "f(x*) = 9/4" in text and "319/40" in text and "68,1 %" in text
    text = EV.verdict_hand(E.hand_example(1, F(1, 2)), 0.5)[1]
    assert "innerhalb der Daten" in text and "3/8" in text and "61/160" in text and "75,6 %" in text


def _cell(lo, hi, blo, bhi):
    s = {"n": 30}
    return {"sets": [s, s, s], "mean_band": (lo, hi), "below90_band": (blo, bhi), "z2_median_band": (1.0, 1.1)}


def test_band_verdikt_schwellen_und_zahlen():
    orc = {"mean": 0.9539, "se": 0.0027, "n": 300}
    assert EV.verdict_band(0.95, _cell(0.93, 0.96, 2, 4), orc, "R", "W", 60)[0] == "success"           # genau 0,93: success
    assert EV.verdict_band(0.95, _cell(0.9299, 0.96, 2, 4), orc, "R", "W", 60)[0] == "info"
    assert EV.verdict_band(0.95, _cell(0.90, 0.96, 2, 4), orc, "R", "W", 60)[0] == "info"             # genau 0,90: info
    kind, text = EV.verdict_band(0.9, _cell(0.8999, 0.91, 12, 20), orc, "Leave-one-out", "V1-Funktion", 20)
    assert kind == "warning" and "Diese Ziehung: Abdeckung 90,0 %" in text and "n = 20, Wahl Leave-one-out" in text and "12 bis 20 von 30" in text and "95,4 % ± 0,3 % (300 Zufalls-Seeds" in text


def test_loo_verdikt():
    chk = {"factorizations_removal": 60, "factorizations_closed": 1, "ops_closed": 154700, "ops_removal": 4517040, "dev_residual": 1.8e-13, "dev_var": 1.2e-14}
    kind, text = EV.verdict_loo(chk)
    assert kind == "success" and "1,8e-13" in text and "Zerlegungen: 1 gegen 60" in text and "154 700 gegen 4 517 040" in text and "Faktor 29,2" in text
    assert EV.verdict_loo(dict(chk, dev_residual=1e-7))[0] == "warning" and EV.verdict_loo(dict(chk, dev_residual=0.99e-7))[0] == "success"
    assert EV.verdict_loo(dict(chk, dev_residual=None, dev_var=None))[0] == "warning"


def test_rules_verdikt_nennt_alle_zaehlungen():
    sm = {"n": 30, "hindsight": 0.0457, "loo_min_optimistic": 13,
          "rules": {"loo": {"mean": 0.0557, "above_2": 1}, "gcv": {"mean": 0.0836, "above_2": 5}, "ev_grid": {"mean": 0.0530, "above_2": 0}, "ev_grad": {"mean": 0.0491, "above_2": 0}}}
    text = EV.verdict_rules(sm)[1]
    assert "Leave-one-out 0,0557" in text and "GCV 0,0836" in text and "Rückblick-Zelle 0,0457" in text and "in 1 / 5 / 0 / 0 von 30" in text and "in 13 von 30" in text


def test_kern_verdikt_relation_aus_differenz_und_standardfehler():
    assert EV._relation(-0.05, 0.02) == "lineare Ridge im Mittel vorn" and EV._relation(0.05, 0.02) == "Kern im Mittel vorn"
    assert EV._relation(0.04, 0.02).startswith("gleichauf") and EV._relation(-0.04, 0.02).startswith("gleichauf")       # genau 2 Standardfehler: gleichauf
    rows = [{"n": 100, "n_seeds": 20, "wins": 7, "diff": -0.0165, "se": 0.0066, "gain_share": -0.029}, {"n": 1000, "n_seeds": 20, "wins": 19, "diff": 0.0073, "se": 0.0009, "gain_share": 0.014}]
    plan = {"n": 30, "kernel_wins": 30, "linear_mean": 0.4796, "rules": {"loo": {"mean": 0.0557}}}
    kind, text = EV.verdict_kern(C.VEHICLE_OPTIONS[1], {"n": 100, "rmse_krr": 0.59, "rmse_lin": 0.58}, rows, plan, 0.486)
    assert kind == "info" and "n = 100: Kern vorn in 7 von 20 Zufalls-Seeds, lineare Ridge im Mittel vorn" in text and "n = 1 000: Kern vorn in 19 von 20" in text
    assert "1,4 %" in text and "Boden 0,486" in text and "30 von 30" in text and "0,0557 gegen 0,4796" in text and "RMSE Kern 0,590" in text
    assert "Überschussfehler Kern" in EV.verdict_kern(C.VEHICLE_OPTIONS[0], {"n": 100, "err_krr": 0.06, "err_lin": 0.48, "err_const": 0.8}, rows, plan, 0.486)[1]


def test_stuetzvektoren_verdikt():
    text = EV.verdict_zero(0, 100, 0.608, 0.48, 0.69, 0, 5000, 50)[1]
    assert "0 Nullen in 100 Koeffizienten" in text and "0 von 5 000 in 50 Instanzen" in text and "60,8 %" in text and "48 % bis 69 %" in text and "y = 0" in text


def test_zahlenformate():
    assert EV.fmt(0.0557, 3) == "0,0557" and EV.fmt(2.6e-3) == "2,6e-3" and EV.fmt(87.0) == "87,0" and EV.fmt(0) == "0"
    assert EV.pct(0.9468) == "94,7 %" and EV.pct(0.608, 0) == "61 %"


def test_je_preset_die_erwartete_verdikt_zeile():
    """Gut gestellt meldet keine Abweichung der Wege; Schlecht gestellt meldet sie ohne Meldung der Löser; Handbeispiel liegt außerhalb der Daten."""
    ok = C.PRESETS["Gut gestellt (σ = 1)"]
    r = EV.wege_run(C.N_W[2], ok.get("sigma", C.SIGMA[2]), 10.0 ** ok.get("lam_exp", C.LAM_EXP[2]), C.SEED_W[2])
    errs = {w: r["ways"][w]["err"] for w in C.WAY_KEYS}
    notes = {w: r["ways"][w]["note"] for w in C.WAY_KEYS}
    kind, text = EV.verdict_wege(r["cond"], errs, notes, r["spread"], L)
    assert kind == "success" and "Gut gestellt" in text
    bad = C.PRESETS["Schlecht gestellt (σ = 4, λ = 1e-10)"]
    r = EV.wege_run(C.N_W[2], bad["sigma"], 10.0 ** bad["lam_exp"], C.SEED_W[2])
    kind, text = EV.verdict_wege(r["cond"], {w: r["ways"][w]["err"] for w in C.WAY_KEYS}, {w: r["ways"][w]["note"] for w in C.WAY_KEYS}, r["spread"], L)
    assert kind == "warning" and "keiner meldet etwas" in text
    assert "außerhalb der Daten" in EV.verdict_hand(E.hand_example(1, F(str(C.PRESETS["Handbeispiel (3 Punkte)"]["hand_x"]))), C.PRESETS["Handbeispiel (3 Punkte)"]["hand_x"])[1]
