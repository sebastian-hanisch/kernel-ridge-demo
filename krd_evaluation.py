"""Auswertung: die Läufe der vier Experimente (alle als reine Funktionen), die vorgerechneten Messungen (JSON) und die Verdikte als reine Funktionen.

Die Läufe hier benutzt auch die Messwerkstatt `tools/krd_make_*.py`: App und Messung rechnen mit demselben Code. Jede Zahl, die ein Verdikt nennt, kommt aus einer Funktion
in diesem Modul und steht in `tests/test_verdicts.py` oder `tests/test_claims.py`."""

import json
import math
import os
from functools import lru_cache

import numpy as np

import krd_constants as C
import krd_container as CT
import krd_exact as E
import krd_model as M
import krd_scenario as S

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- vorgerechnete Messungen
@lru_cache(maxsize=None)
def load_data(name: str) -> dict:
    """Eine vorgerechnete Messung (krd_<name>.json) laden; sie wird von tools/krd_make_<name>.py erzeugt und mit `--check` geprüft."""
    with open(os.path.join(HERE, C.DATA_FILES[name]), encoding="utf-8") as fh:
        return json.load(fh)


def fmt(x: float, digits: int = 1) -> str:
    """Zahl im Anzeigestil: Komma als Dezimalzeichen; kleine und große Beträge in Zehnerpotenzschreibweise (2 Stellen)."""
    if x == 0:
        return "0"
    if abs(x) < 1e-2 or abs(x) >= 1e4:
        m, e = f"{x:.{digits}e}".split("e")
        return f"{m.replace('.', ',')}e{int(e)}"
    return f"{x:.{digits + 1}f}".replace(".", ",") if abs(x) < 10 else f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma (x in 0 bis 1)."""
    return f"{100 * x:.{digits}f}".replace(".", ",") + " %"


def band3(values) -> tuple[float, float, float]:
    """(kleinster, Median, größter) Wert."""
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    return float(v.min()), float(np.median(v)), float(v.max())


# ---------------------------------------------------------------- Experiment 1: Rechenwege gegen die exakte Lösung
def wege_run(n: int, sigma: float, lam: float, seed: int, target: str = "v1", ways=C.WAY_KEYS, center: bool = True, n_test: int = C.N_TEST_WAYS,
             exact: bool = True, maxiter: int = C.MIN_MAXITER) -> dict:
    """Ein Zufalls-Seed: Daten der Plan-Regression, Gram-Matrix, jeder gewählte Weg gegen die exakte Lösung des Gleitkomma-Systems (Bareiss; nur bis n <= BAREISS_LIMIT).
    Maß: größter Betrag der Abweichung der Vorhersage über n_test Testpunkte. Rückgabe: cond, je Weg {err, note}, Abstand der direkten Wege untereinander (ohne Minimierung)."""
    X, y, _ = S.make_plan(n, seed, target)
    Xs, _ = S.make_test(n_test, seed + 1_000_000, target)
    yc = y - y.mean() if center else y
    K = M.gram(X, X, sigma)
    Ks = M.gram(Xs, X, sigma)
    A = K + lam * np.eye(n)
    cond = float(np.linalg.cond(A))
    f_ex = None
    if exact and n <= C.BAREISS_LIMIT:
        try:
            f_ex = Ks @ E.exact_solve(A, yc)
        except ZeroDivisionError:
            f_ex = None
    preds, out = {}, {}
    for w in ways:
        alpha, note = M.run_way(w, K, yc, lam, maxiter)
        preds[w] = None if alpha is None else Ks @ alpha
        out[w] = {"err": None if (alpha is None or f_ex is None) else float(np.abs(preds[w] - f_ex).max()), "note": note}
    direct = [preds[w] for w in ways if w != "min" and preds[w] is not None]
    spread = max((float(np.abs(a - b).max()) for a in direct for b in direct), default=0.0) if len(direct) >= 2 else None
    return {"cond": cond, "ways": out, "spread": spread, "exact": f_ex is not None, "silent_direct": sum(1 for w in ways if w != "min" and out[w]["note"] is None),
            "n_direct": sum(1 for w in ways if w != "min")}


def wege_band(n: int, sigma: float, lam: float, seed: int, n_seeds: int, target: str = "v1", ways=C.WAY_KEYS, center: bool = True) -> dict:
    """Mehrere aufeinanderfolgende Zufalls-Seeds (seed, seed + 1, ...): Band (kleinster / Median / größter Fehler) je Weg, Band der Kondition,
    Läufe ohne jede Meldung der direkten Wege und Verhältnis (Abstand der direkten Wege / größter Fehler eines direkten Weges)."""
    runs = [wege_run(n, sigma, lam, seed + s, target, ways, center) for s in range(n_seeds)]
    errs = {w: [r["ways"][w]["err"] for r in runs] for w in ways}
    ratios = []
    for r in runs:
        direct = [r["ways"][w]["err"] for w in ways if w != "min" and r["ways"][w]["err"] is not None]
        if r["spread"] is not None and direct and max(direct) > 0:
            ratios.append(r["spread"] / max(direct))
    return {
        "n_seeds": n_seeds, "cond": band3([r["cond"] for r in runs]),
        "err": {w: (band3([e for e in errs[w] if e is not None]) if any(e is not None for e in errs[w]) else None) for w in ways},
        "silent_runs": sum(1 for r in runs if r["silent_direct"] == r["n_direct"]), "min_notes": sum(1 for r in runs if "min" in ways and r["ways"]["min"]["note"]),
        "ratio": band3(ratios) if ratios else None,
    }


# ---------------------------------------------------------------- Experiment 2: das Band des GP
def _chosen(table, X, y, rule: str):
    """(sigma, lambda, s2) nach Regel; 'fixed' ist die feste Zelle, 'ev_grad' das Gradientenverfahren."""
    if rule == "fixed":
        sg, lm = C.FIXED_CELL
        _, s2 = M.evidence_at(X, y, sg, lm)
        return sg, lm, s2
    if rule == "ev_grad":
        sg, lm, s2, _, _, _ = M.pick_gradient(X, y)
        return sg, lm, s2
    i, j = M.pick_cell(table, {"ev_grid": "evidence", "loo": "loo"}[rule])
    return C.SIGMAS_A[i], C.LAMS_A[j], float(table["sf2"][i, j])


SLICE_X1 = np.linspace(-2.0, 2.0, 81)
SLICE = np.column_stack([SLICE_X1, np.zeros(81)])


def band_run(truth: str, n: int, seed: int, rule: str, center: bool = True, with_slice: bool = True) -> dict:
    """Ein Zufalls-Seed des Band-Experiments: Daten (Wahrheit V1, raue Funktion oder aus dem Prior gezogen), Wahl von (sigma, lambda, s2) nach Regel, GP-Posterior der FUNKTION
    an den Testpunkten. Abdeckung = Anteil der Testpunkte, an denen die Wahrheit im 95-%-Band liegt (|z| <= 1,96); z2 = Mittel von z^2 (Soll 1).
    Testpunkte: 1 000 (V1, raue Funktion) bzw. 100 (Prior, wie die Orakel-Messung). Zusätzlich ein Schnitt bei x2 = 0 für die Zeichnung."""
    if truth == "prior":
        X, y, Xt, ft, f_sl = S.draw_from_prior(n, 100, seed, X_extra=SLICE if with_slice else None)
    else:
        X, y, _ = S.make_plan(n, seed, truth)
        Xt, ft = S.make_test(C.N_TEST, 5301, truth)
        f_sl = S.TARGETS[truth](SLICE) if with_slice else None
    ym = y.mean() if center else 0.0
    table = M.cell_table(X, y, C.SIGMAS_A, C.LAMS_A) if rule in ("ev_grid", "loo") else None
    sg, lm, s2 = _chosen(table, X, y, rule)
    K = M.gram(X, X, sg)
    m, v = M.posterior(K, M.gram(Xt, X, sg), y, lm, s2, center)
    z = (m - ft) / np.sqrt(v)
    res = {"sigma": sg, "lam": lm, "s2": s2, "noise_var": lm * s2, "coverage": float(np.mean(np.abs(z) <= 1.96)), "z2": float(np.mean(z * z)), "rmse": float(np.sqrt(np.mean((m - ft) ** 2))),
           "mean_sd": float(np.sqrt(v).mean()), "X": X, "y": y, "ym": ym}
    if with_slice:
        ms, vs = M.posterior(K, M.gram(SLICE, X, sg), y, lm, s2, center)
        sd = np.sqrt(vs)
        res.update({"slice_x": SLICE_X1, "slice_truth": f_sl, "slice_mean": ms, "slice_lo": ms - 1.96 * sd, "slice_hi": ms + 1.96 * sd})
    return res


def prior_oracle(seed: int, n: int = 30, n_test: int = 100) -> dict:
    """Kalibrierungs-Orakel: die Wahrheit stammt aus dem Prior (Gauß-Kern sigma = 1, Prior-Varianz 1, Rauschvarianz 0,09), die Hyperparameter sind bekannt (lambda = 0,09,
    Prior-Mittel 0, keine Zentrierung). Dann muss das 95-%-Band im Mittel über Zufalls-Seeds 95 % der Funktionswerte enthalten und z^2 im Mittel 1 sein."""
    X, y, Xt, ft, _ = S.draw_from_prior(n, n_test, seed)
    m, v = M.posterior(M.gram(X, X, 1.0), M.gram(Xt, X, 1.0), y, C.NOISE_VAR, 1.0, center=False)
    z = (m - ft) / np.sqrt(v)
    return {"coverage": float(np.mean(np.abs(z) <= 1.96)), "z2": float(np.mean(z * z))}


def wege_frozen(setting_key: str) -> dict:
    """Vorgerechnetes Band über 20 Zufalls-Seeds (krd_wege.json) für eine Einstellung 'sigma|lambda' (z. B. '4|1e-10'): Band der Kondition, Band des größten Fehlers
    je Weg (kleinster / Median / größter), Band über die fünf direkten Wege zusammen, Läufe ohne jede Meldung, Läufe mit Meldung der Minimierung, Band des Verhältnisses
    (Abstand der Wege untereinander / größter Fehler)."""
    per = load_data("wege")["band"]["settings"][setting_key]["per_seed"]
    ways = ("cholesky", "lu", "eig", "sk_kr", "sk_gp")
    errs = {w: [p["err"][w] for p in per if p["err"][w] is not None] for w in ("cholesky", "lu", "eig", "min", "sk_kr", "sk_gp")}
    allv = [p["err"][w] for p in per for w in ways if p["err"][w] is not None]
    return {"n_seeds": len(per), "cond": band3([p["cond"] for p in per]), "err": {w: band3(v) for w, v in errs.items()}, "direct_range": (min(allv), max(allv)),
            "silent_runs": sum(1 for p in per if p["silent"]), "min_notes": sum(1 for p in per if p["min_note"]), "ratio": band3([p["ratio"] for p in per if p["ratio"] is not None])}


def far_means(X, y, sigma: float, lam: float, point=(6.0, 6.0)) -> dict:
    """Zentrierung als Konvention: Mittel an einem Punkt weit außerhalb der Daten, mit Prior-Mittel 0 (fällt auf 0) und mit abgezogenem Trainingsmittel (fällt auf das Trainingsmittel)."""
    K = M.gram(X, X, sigma)
    ks = M.gram(np.array([point]), X, sigma)
    return {"prior_zero": float(M.posterior(K, ks, y, lam, 1.0, center=False)[0][0]), "centered": float(M.posterior(K, ks, y, lam, 1.0, center=True)[0][0]), "train_mean": float(np.mean(y))}


def band_summary(values) -> dict:
    """Verteilung der Abdeckung über Zufalls-Seeds: Mittel, Standardfehler, 10-%-Perzentil, Anzahl unter 0,90, Anzahl Zufalls-Seeds."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0, "p10": float(np.percentile(v, 10)), "below90": int((v < 0.90).sum()), "n": len(v)}


def band_cell(truth: str, n: int, rule: str) -> dict:
    """Vorgerechnete Verteilung der Abdeckung (alle Sätze von Zufalls-Seeds aus krd_band.json zusammen) für eine Zelle (Wahrheit, n, Regel);
    je Satz eine eigene Zusammenfassung, dazu das Band über die Sätze (kleinster bis größter Mittelwert, Anteil unter 0,90)."""
    data = load_data("band")
    sets = [band_summary(s["cells"][f"{truth}|{n}|{rule}"]["coverage"]) for s in data["sets"]]
    means = [s["mean"] for s in sets]
    below = [s["below90"] for s in sets]
    z2 = [float(np.median(s["cells"][f"{truth}|{n}|{rule}"]["z2"])) for s in data["sets"]]
    return {"sets": sets, "mean_band": (min(means), max(means)), "below90_band": (min(below), max(below)), "z2_median_band": (min(z2), max(z2))}


# ---------------------------------------------------------------- Experiment 3: Leave-one-out ohne n-mal Rechnen, Wahl von (sigma, lambda)
def loo_check(n: int, sigma: float, lam: float, seed: int, target: str = "v1", center: bool = True) -> dict:
    """Leave-one-out geschlossen gegen echtes Herausnehmen an (sigma, lambda): größte Abweichung der Residuen (relativ zu max(1, größtes Residuum)), Zählung der
    Zerlegungen (1 gegen n) und der Gleitkommaoperationen (mitgezählt: Multiplikationen und Additionen)."""
    X, y, _ = S.make_plan(n, seed, target)
    yc = y - y.mean() if center else y
    K = M.gram(X, X, sigma)
    e_c, mu_c, v_c = M.loo_closed(K, yc, lam)
    try:
        e_r, mu_r, v_r = M.loo_by_removal(K, yc, lam)
        sc = max(1.0, float(np.abs(e_r).max()))
        dev = float(np.abs(e_c - e_r).max()) / sc
        dev_var = float(np.abs(v_c - v_r).max()) / max(1.0, float(np.abs(v_r).max()))
    except np.linalg.LinAlgError:
        dev, dev_var = None, None
    return {"dev_residual": dev, "dev_var": dev_var, "factorizations_closed": 1, "factorizations_removal": n, "ops_closed": M.flops_loo_closed(n), "ops_removal": M.flops_loo_removal(n),
            "press": float(np.mean(e_c ** 2)), "residuals": e_c}


def flaeche_run(n: int, seed: int, target: str = "v1", center: bool = True) -> dict:
    """Fläche über Raster A (7 x 6) für einen Zufalls-Seed: Tabellen (PRESS, GCV, Evidenz, Testfehler), Wahl je Regel (Leave-one-out, GCV, Evidenz auf dem Raster,
    Evidenz per Gradient) und Rückblick-Zelle (beste Zelle am Test, nicht ausführbar)."""
    X, y, _ = S.make_plan(n, seed, target)
    Xt, ft = S.make_test(C.N_TEST, 5301, target)
    t = M.cell_table(X, y, C.SIGMAS_A, C.LAMS_A, Xt, ft)
    picks = {"loo": M.pick_cell(t, "loo"), "gcv": M.pick_cell(t, "gcv"), "ev_grid": M.pick_cell(t, "evidence")}
    tests = {k: float(t["test"][p]) for k, p in picks.items()}
    sg, lm, s2, _, edge, _ = M.pick_gradient(X, y)
    K = M.gram(X, X, sg)
    m, _ = M.posterior(K, M.gram(Xt, X, sg), y, lm, s2, center=True)
    tests["ev_grad"] = float(np.mean((m - ft) ** 2))
    best = np.unravel_index(np.argmin(t["test"]), t["test"].shape)
    picks["ev_grad"] = (sg, lm)
    return {"table": t, "picks": picks, "tests": tests, "best": (int(best[0]), int(best[1])), "best_test": float(t["test"].min()), "grad_edge": edge}


def pick_values(pick) -> tuple[float, float]:
    """(sigma, lambda) einer Wahl: Rasterzelle (Indizes) oder Werte des Gradientenverfahrens."""
    i, j = pick
    if isinstance(i, int) and isinstance(j, int):
        return C.SIGMAS_A[i], C.LAMS_A[j]
    return float(i), float(j)


def nearest_cell(sigma: float, lam: float) -> tuple[int, int]:
    """Rasterzelle von Raster A, die (sigma, lambda) auf logarithmischer Skala am nächsten liegt (nur für die Markierung in der Zeichnung)."""
    i = min(range(len(C.SIGMAS_A)), key=lambda k: abs(math.log(C.SIGMAS_A[k]) - math.log(sigma)))
    j = min(range(len(C.LAMS_A)), key=lambda k: abs(math.log(C.LAMS_A[k]) - math.log(lam)))
    return i, j


def flaeche_summary(set_name: str = "A") -> dict:
    """Zusammenfassung eines Satzes von Zufalls-Seeds aus krd_flaeche.json: je Regel Mittel des Überschussfehlers der gewählten Zelle, Zufalls-Seeds mit Verhältnis zur
    Rückblick-Zelle über 1,5 und über 2 (letzteres „gescheitert“ wie in der parameterwahl-demo), größtes Verhältnis; dazu Rückblick-Zelle, LOO-Minimum als Schätzer
    (PRESS_min minus Rauschvarianz) gegen den tatsächlichen Fehler, lineare Ridge und Siege des Kerns (Wahl per Leave-one-out) gegen linear."""
    blk = load_data("flaeche")["sets"][set_name]
    rows = blk["rows"]
    best = np.array([r["best_test"] for r in rows])
    out = {"n": len(rows), "hindsight": float(best.mean()), "rules": {}}
    for rule in ("loo", "gcv", "ev_grid", "ev_grad"):
        t = np.array([r["tests"][rule] for r in rows])
        ratio = t / best
        out["rules"][rule] = {"mean": float(t.mean()), "above_1_5": int((ratio > 1.5).sum()), "above_2": int((ratio > 2.0).sum()), "max_ratio": float(ratio.max())}
    est = np.array([r["press_min"] - C.NOISE_VAR for r in rows])
    act = np.array([r["tests"]["loo"] for r in rows])
    out["loo_min_estimate"] = float(est.mean())
    out["loo_min_actual"] = float(act.mean())
    out["loo_min_optimistic"] = int((est < act).sum())
    lin = np.array([r["err_lin"] for r in rows])
    out["linear_mean"] = float(lin.mean())
    out["kernel_wins"] = int((act < lin).sum())
    tb = np.array([r["test_grid_b"] for r in rows])
    out["ev_grid_b_mean"] = float(tb.mean())
    out["ev_grid_b_above_2"] = int((tb / np.array([r["best_test_b"] for r in rows]) > 2.0).sum())
    out["grad_edge"] = int(sum(r["grad_edge"] for r in rows))
    return out


# ---------------------------------------------------------------- Experiment 4: Kern gegen linear
def container_run(n: int, seed: int) -> dict:
    """Regression der log-Verweilzeit (Container-Erzeuger): Kernel Ridge (Skalenfaktor f und lambda per geschlossenem Leave-one-out auf dem Raster CONT_FACTORS x CONT_LAMS)
    gegen lineare Ridge (lambda per Leave-one-out). Trainingsdaten Zufalls-Seed 1000 + seed, 4 000 frische Testcontainer (Zufalls-Seed 700000 + seed). RMSE in log-Tagen."""
    Xe, d, _, _ = CT.make_container(n, 1000 + seed)
    Xt, dt, _, st = CT.make_container(C.N_TEST_CONT, 700000 + seed)
    Ztr, Zte = M.standardize_pair(CT.encode(Xe), CT.encode(Xt))
    ltr, lte = np.log(d), np.log(dt)
    yb = float(ltr.mean())
    yc = ltr - yb
    sig_med = M.median_distance(Ztr)
    sg, lm, a, _ = M.krr_grid_loo(Ztr, yc, [f * sig_med for f in C.CONT_FACTORS], C.CONT_LAMS, fast=True)
    pred = M.gram(Zte, Ztr, sg, fast=True) @ a + yb
    w, lm_lin = M.linear_ridge_loo(Ztr, yc, C.LIN_LAMS)
    pred_lin = Zte @ w + yb
    return {"rmse_krr": float(np.sqrt(np.mean((pred - lte) ** 2))), "rmse_lin": float(np.sqrt(np.mean((pred_lin - lte) ** 2))), "floor_sample": float(np.sqrt(np.mean(st ** 2))),
            "rmse_const": float(np.sqrt(np.mean((yb - lte) ** 2))), "factor": sg / sig_med, "lam": lm, "lam_lin": lm_lin}


def lernkurve_summary(n: int) -> dict:
    """Zusammenfassung der vorgerechneten Lernkurve (krd_lernkurve.json) für n Trainingscontainer: Mittel der RMSE beider Verfahren, Siege des Kerns (gepaart),
    Differenz (linear minus Kern) mit Standardfehler, Anteil der Differenz am RMSE der linearen Ridge, konstanter Mittelwert, Stichprobenmittel des Bodens."""
    runs = load_data("lernkurve")["runs"][str(n)]
    k = np.array([r["rmse_krr"] for r in runs])
    li = np.array([r["rmse_lin"] for r in runs])
    diff = li - k
    return {"n_seeds": len(runs), "rmse_krr": float(k.mean()), "rmse_lin": float(li.mean()), "wins": int((k < li).sum()), "diff": float(diff.mean()),
            "se": float(diff.std(ddof=1) / math.sqrt(len(diff))), "gain_share": float(diff.mean() / li.mean()), "const": float(np.mean([r["rmse_const"] for r in runs])),
            "floor_sample": float(np.mean([r["floor_sample"] for r in runs]))}


def plan_run(n: int, seed: int, target: str = "v1") -> dict:
    """Plan-Regression: Kernel Ridge (sigma, lambda per geschlossenem Leave-one-out auf Raster A) gegen lineare Ridge (lambda per Leave-one-out); Überschussfehler
    (mittlerer quadratischer Fehler gegen die rauschfreie Wahrheit auf 1 000 Testpunkten); konstanter Mittelwert als Bezug."""
    X, y, _ = S.make_plan(n, seed, target)
    Xt, ft = S.make_test(C.N_TEST, 5301, target)
    ym = float(y.mean())
    yc = y - ym
    sg, lm, a, _ = M.krr_grid_loo(X, yc, C.SIGMAS_A, C.LAMS_A)
    pred = M.gram(Xt, X, sg) @ a + ym
    w, lm_lin = M.linear_ridge_loo(X - X.mean(0), yc, C.LIN_LAMS_PLAN)
    pred_lin = (Xt - X.mean(0)) @ w + ym
    return {"err_krr": float(np.mean((pred - ft) ** 2)), "err_lin": float(np.mean((pred_lin - ft) ** 2)), "err_const": float(np.mean((ym - ft) ** 2)), "sigma": sg, "lam": lm, "lam_lin": lm_lin}


# ---------------------------------------------------------------- keine Stützvektoren
def zero_count(alpha, tol: float = 1e-12) -> int:
    """Anzahl der Koeffizienten mit |alpha_i| <= tol * größter Betrag (Rauschgrenze; Nichtnullen zählen braucht eine Grenze)."""
    a = np.abs(np.asarray(alpha, dtype=float))
    return int((a <= tol * a.max()).sum()) if a.size and a.max() > 0 else int(a.size)


def ridge_pm1(n: int, seed: int, sigma: float = 0.4, lam: float = 0.3) -> dict:
    """Kernel Ridge auf ±1-Labels des Kreises (Plan-Erzeuger, 10 % vertauschte Labels): Zahl der Nullen in alpha."""
    X, y = S.make_circle(n, seed)
    K = M.gram(X, X, sigma)
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    return {"zeros": zero_count(alpha), "alpha": alpha, "X": X, "y": y}


# ---------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text); die App ruft st.<Art>(Text))
GOOD_ERR = 1e-8       # direkte Wege gelten als übereinstimmend mit der exakten Lösung, wenn der größte Fehler darunter liegt
BAD_ERR = 1e-6        # ab diesem größten Fehler (Skala der Vorhersage, größter Betrag, etwa 1 bis 2) gilt die Abweichung als deutlich
BAND_GOOD = 0.93      # Mittel der Abdeckung in jedem Satz mindestens so hoch: Band „brauchbar“
BAND_BAD = 0.90       # Mittel der Abdeckung in mindestens einem Satz darunter: Band „zu eng“


def verdict_wege(cond: float, errors: dict, notes: dict, spread, labels: dict) -> tuple[str, str]:
    """Verdikt der Wege-Anzeige für einen Zufalls-Seed. `errors[w]` = größter Fehler gegen die exakte Lösung (None: kein Ergebnis oder keine Referenz),
    `notes[w]` = Meldung des Lösers (None: nichts gemeldet). Die Minimierung ("min") zählt nicht zu den direkten Wegen."""
    direct = [w for w in errors if w != "min"]
    shown = [errors[w] for w in direct if errors[w] is not None]
    raised = [labels[w] for w in direct if errors[w] is None and notes.get(w)]
    if not shown:
        if raised:
            return "warning", f"{', '.join(raised)} meldet eine Ausnahme und liefert kein Ergebnis (cond(K + λI) = {fmt(cond)}); kein direkter Weg ist übrig, der mit der exakten Lösung verglichen werden kann."
        return "info", "Kein direkter Weg gewählt: Cholesky, LU oder Eigenzerlegung ankreuzen, um sie mit der exakten Lösung zu vergleichen."
    worst = max(shown)
    silent = sum(1 for w in direct if errors[w] is not None and notes.get(w) is None)
    tail = ""
    if "min" in errors and errors["min"] is not None:
        tail = f" Die Minimierung ist am ungenauesten ({fmt(errors['min'])})" + (f" und meldet als einzige „{notes['min']}“." if notes.get("min") else ".")
    sp = f" Abstand der direkten Wege untereinander {fmt(spread)}." if spread is not None else ""
    if raised:
        return "warning", f"{', '.join(raised)} meldet eine Ausnahme (cond(K + λI) = {fmt(cond)}); die übrigen direkten Wege weichen bis {fmt(worst)} von der exakten Lösung ab.{sp}{tail}"
    if worst < GOOD_ERR:
        return "success", (f"Gut gestellt: die {len(shown)} direkten Wege liegen höchstens {fmt(worst)} neben der exakten Lösung (cond(K + λI) = {fmt(cond)}).{sp} "
                           f"Das ist Gleichheit der Gleichung, und sie hält hier; Gleichheit der Wege bleibt aber keine Richtigkeit.{tail}")
    quiet = "keiner meldet etwas" if silent == len(shown) else f"{silent} von {len(shown)} ohne Meldung"
    kind = "warning" if worst >= BAD_ERR else "info"
    return kind, (f"Die {len(shown)} direkten Wege weichen bis {fmt(worst)} von der exakten Lösung ab (cond(K + λI) = {fmt(cond)}), {quiet}.{sp} "
                  f"Gleichheit der Wege ist Gleichheit der Gleichung, nicht Richtigkeit.{tail}")


def verdict_wege_band(band: dict) -> tuple[str, str]:
    """Verdikt für mehrere Zufalls-Seeds (Ergebnis von wege_band): Band des größten Fehlers der direkten Wege und Läufe ohne jede Meldung."""
    direct = [w for w, b in band["err"].items() if w != "min" and b is not None]
    if not direct:
        return "info", "Kein direkter Weg gewählt: Cholesky, LU oder Eigenzerlegung ankreuzen."
    lo = min(band["err"][w][0] for w in direct)
    hi = max(band["err"][w][2] for w in direct)
    n = band["n_seeds"]
    quiet = f"In {band['silent_runs']} von {n} Läufen meldete keiner der direkten Wege etwas."
    ratio = ""
    if band.get("ratio"):
        ratio = f" Abstand der Wege untereinander geteilt durch den größten Fehler: {band['ratio'][0]:.2f} bis {band['ratio'][2]:.2f} (Median {band['ratio'][1]:.2f})."
    kind = "success" if hi < GOOD_ERR else ("warning" if hi >= BAD_ERR else "info")
    return kind, f"Über {n} Zufalls-Seeds: größter Fehler der direkten Wege {fmt(lo)} bis {fmt(hi)} (cond(K + λI) {fmt(band['cond'][0])} bis {fmt(band['cond'][2])}). {quiet}{ratio}"


def verdict_hand(h: dict, x_star: float) -> tuple[str, str]:
    """Verdikt des Handbeispiels an x*: Mittel und Varianz in Brüchen, abgebauter Anteil der Prior-Varianz; Lage von x* zu den Daten (-1 bis 1)."""
    f, v, p = h["f_star"], h["var_star"], h["prior_var"]
    share = 1 - float(v) / float(p)
    where = "innerhalb der Daten (x von -1 bis 1)" if -1.0 <= x_star <= 1.0 else "außerhalb der Daten (x von -1 bis 1)"
    return "info", (f"x* = {fmt(x_star)} liegt {where}: Mittel f(x*) = {f}, Varianz der Funktion {v} ({fmt(float(v), 2)}) gegen Prior-Varianz {p} ({fmt(float(p), 2)}); "
                    f"es werden {pct(share)} der Prior-Varianz abgebaut. Kernel Ridge mit λ = 1 und das GP-Mittel mit Prior-Varianz 1 und Rauschvarianz 1 geben dasselbe Mittel; die Varianz gibt es nur beim GP.")


def verdict_band(coverage: float, cell: dict, oracle: dict, rule_label: str, truth_label: str, n: int) -> tuple[str, str]:
    """Verdikt des Band-Experiments: Abdeckung dieser Ziehung, Verteilung über die vorgerechneten Zufalls-Seeds (Sätze), Orakel mit bekannten Hyperparametern."""
    lo, hi = cell["mean_band"]
    blo, bhi = cell["below90_band"]
    n_sets = len(cell["sets"])
    n_seeds = cell["sets"][0]["n"]
    kind = "success" if lo >= BAND_GOOD else ("warning" if lo < BAND_BAD else "info")
    where = ("Die Abdeckung gilt hier innerhalb der Daten und für diese Wahl von σ und λ; außerhalb der Daten (feste Hyperparameter) ist ein anderer Ort und Gegenstand "
             "der Kernwahl-Demo der Reihe, kein Widerspruch.")
    return kind, (f"Diese Ziehung: Abdeckung {pct(coverage)}. {truth_label}, n = {n}, Wahl {rule_label}: im Mittel über {n_seeds} Zufalls-Seeds {pct(lo)} bis {pct(hi)} "
                  f"(kleinster und größter von {n_sets} Sätzen), {blo} bis {bhi} von {n_seeds} Zufalls-Seeds unter 90 %. Wahrheit aus dem Prior mit bekannten Hyperparametern: "
                  f"{pct(oracle['mean'])} ± {pct(oracle['se'])} ({oracle['n']} Zufalls-Seeds, Soll 95 %). Das Band meldet nicht von selbst, dass der Prior nicht passt. {where}")


def verdict_loo(chk: dict) -> tuple[str, str]:
    """Verdikt: Leave-one-out geschlossen gegen echtes Herausnehmen, mit Zählung der Zerlegungen und der Gleitkommaoperationen."""
    n = chk["factorizations_removal"]
    ops = (f"Zerlegungen: {chk['factorizations_closed']} gegen {n}; Gleitkommaoperationen (mitgezählt): {chk['ops_closed']:,} gegen {chk['ops_removal']:,} "
           f"(Faktor {chk['ops_removal'] / chk['ops_closed']:.1f})").replace(",", " ").replace(".", ",")
    if chk["dev_residual"] is None:
        return "warning", f"Echtes Herausnehmen scheitert hier (die Gram-Matrix ist numerisch nicht positiv definit); die geschlossene Formel liefert trotzdem Zahlen. {ops}."
    kind = "success" if chk["dev_residual"] < 1e-7 else "warning"
    return kind, f"Geschlossen gleich herausgenommen: größte Abweichung der Residuen {fmt(chk['dev_residual'])} (relativ), der Varianzen {fmt(chk['dev_var'])}. {ops}."


def verdict_rules(sm: dict, set_label: str = "Satz A") -> tuple[str, str]:
    """Verdikt der Wahlregeln (Zusammenfassung eines Satzes von Zufalls-Seeds, n = 60): Mittel des Überschussfehlers, Fehlgriffe über das Doppelte, LOO-Minimum optimistisch."""
    r = sm["rules"]
    n = sm["n"]
    return "info", (f"Gewählt (n = 60, {set_label}, {n} Zufalls-Seeds, Überschussfehler): Leave-one-out {fmt(r['loo']['mean'], 3)}, GCV {fmt(r['gcv']['mean'], 3)}, Evidenz auf dem Raster "
                    f"{fmt(r['ev_grid']['mean'], 3)}, Evidenz per Gradient {fmt(r['ev_grad']['mean'], 3)}, Rückblick-Zelle {fmt(sm['hindsight'], 3)}; über das Doppelte der Rückblick-Zelle in "
                    f"{r['loo']['above_2']} / {r['gcv']['above_2']} / {r['ev_grid']['above_2']} / {r['ev_grad']['above_2']} von {n} Zufalls-Seeds. Das Leave-one-out-Minimum unterschätzt den Fehler "
                    f"in {sm['loo_min_optimistic']} von {n} Zufalls-Seeds. Die Rangfolge gilt im Mittel; zwischen den Sätzen von Zufalls-Seeds streut sie (siehe README).")


def _relation(diff: float, se: float) -> str:
    if diff < -2.0 * se:
        return "lineare Ridge im Mittel vorn"
    if diff > 2.0 * se:
        return "Kern im Mittel vorn"
    return "gleichauf (Differenz innerhalb von 2 Standardfehlern)"


def verdict_kern(vehicle: str, live: dict, rows: list, plan: dict, floor: float) -> tuple[str, str]:
    """Verdikt von Experiment 4. `rows`: lernkurve_summary je n (Container, mit Schlüssel n); `plan`: flaeche_summary (Plan-Regression, n = 60); `live`: die Ziehung dieser Seite."""
    cont = "; ".join(f"n = {r['n']:,}".replace(",", " ") + f": Kern vorn in {r['wins']} von {r['n_seeds']} Zufalls-Seeds, {_relation(r['diff'], r['se'])}" for r in rows)
    gain = rows[-1]
    plan_txt = f"Plan-Regression (n = 60, {plan['n']} Zufalls-Seeds): Kern vorn in {plan['kernel_wins']} von {plan['n']}, Überschussfehler {fmt(plan['rules']['loo']['mean'], 3)} gegen {fmt(plan['linear_mean'], 3)} bei linearer Ridge."
    if vehicle == C.VEHICLE_OPTIONS[1]:
        here = f"Diese Ziehung (n = {live['n']}): RMSE Kern {fmt(live['rmse_krr'], 3)}, linear {fmt(live['rmse_lin'], 3)}, Boden {fmt(floor, 3)} (log-Tage)."
    else:
        here = f"Diese Ziehung (n = {live['n']}): Überschussfehler Kern {fmt(live['err_krr'], 3)}, linear {fmt(live['err_lin'], 3)}, konstanter Mittelwert {fmt(live['err_const'], 3)}."
    n_gain = f"{gain['n']:,}".replace(",", " ")
    return "info", (f"Container-Erzeuger (20 Zufalls-Seeds, gepaart): {cont}; bei n = {n_gain} sind es {pct(gain['gain_share'])} des RMSE der linearen Ridge. Der Boden {fmt(floor, 3)} (exakt) "
                    f"ist die Untergrenze des RMSE im Erwartungswert. {plan_txt} {here}")


def verdict_zero(zeros: int, n: int, sv_mean: float, sv_lo: float, sv_hi: float, total_zero: int, total_coeff: int, n_inst: int) -> tuple[str, str]:
    """Verdikt „keine Stützvektoren“: Nullen in alpha (diese Instanz und alle vorgerechneten) gegen den Stützvektoranteil von SVC."""
    total = f"{total_coeff:,}".replace(",", " ")
    return "info", (f"Kernel Ridge auf den ±1-Labels dieser Instanz: {zeros} Nullen in {n} Koeffizienten (vorgerechnet: {total_zero} von {total} in {n_inst} Instanzen, Rauschgrenze 1e-12 "
                    f"des größten Betrags). SVC mit demselben Kern: Stützvektoranteil im Mittel {pct(sv_mean)} ({pct(sv_lo, 0)} bis {pct(sv_hi, 0)}). Gilt für allgemeine Lage; für y = 0 ist alpha = 0.")
