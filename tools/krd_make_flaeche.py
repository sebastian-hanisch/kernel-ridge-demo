"""Messung zu Experiment 3 und 4 (Plan-Regression): Wahl von (sigma, lambda) durch Leave-one-out, GCV und Evidenz (Raster und Gradient) auf der V1-Fläche, n = 60;
Leave-one-out geschlossen gegen echtes Herausnehmen (100 Zufallsinstanzen); Operationszählung; lineare Ridge als Gegner.

Schreibt krd_flaeche.json mit drei Sätzen von Zufalls-Seeds (Satz A: 30 Zufalls-Seeds 5400 + s, Sätze B und C: je 40 frische Zufalls-Seeds). Zähleinheiten: Zufalls-Seeds,
Instanzen, Testpunkte (1 000), Gleitkommaoperationen (mitgezählt). Maß: Überschussfehler = mittlerer quadratischer Fehler gegen die rauschfreie Wahrheit (konstanter Mittelwert
etwa 0,79, Rauschvarianz 0,09). Keine Sekunden als Befund. Aufruf: python tools/krd_make_flaeche.py [--check]"""

import math
import sys
from multiprocessing import Pool

import _krd_common as T

import numpy as np

import krd_constants as C
import krd_evaluation as EV
import krd_model as M
import krd_scenario as S
from krd_rng import SplitMix64

N = 60
SETS = (("A", 5400, 30), ("B", 19000, 40), ("C", 25000, 40))
RULES = ("loo", "gcv", "ev_grid", "ev_grad")


def one_seed(args):
    n, seed = args
    r = EV.flaeche_run(n, seed)
    t = r["table"]
    X, y, _ = S.make_plan(n, seed)
    Xt, ft = S.make_test(C.N_TEST, 5301)
    i, j = r["picks"]["loo"]
    press_min = float(t["press"][i, j])
    # Raster B: Evidenz auf dem zweiten Raster
    tb = M.cell_table(X, y, C.SIGMAS_B, C.LAMS_B, Xt, ft)
    ib, jb = M.pick_cell(tb, "evidence")
    # lineare Ridge
    ym = float(y.mean())
    w, _ = M.linear_ridge_loo(X - X.mean(0), y - ym, C.LIN_LAMS_PLAN)
    err_lin = float(np.mean(((Xt - X.mean(0)) @ w + ym - ft) ** 2))
    return {"tests": r["tests"], "best_test": r["best_test"], "press_min": press_min, "test_grid_b": float(tb["test"][ib, jb]), "best_test_b": float(tb["test"].min()),
            "err_lin": err_lin, "grad_edge": bool(r["grad_edge"]), "ev_pick": [C.SIGMAS_A[r["picks"]["ev_grid"][0]], C.LAMS_A[r["picks"]["ev_grid"][1]]],
            "grad_pick": [float(r["picks"]["ev_grad"][0]), float(r["picks"]["ev_grad"][1])], "table_test": t["test"].tolist()}


def loo_instances(n_inst=100):
    """100 Zufallsinstanzen (n = 6 bis 30, sigma 0,3 bis 3, lambda 1e-3 bis 1): geschlossen gegen echtes Herausnehmen (relative Abweichung)."""
    rng = SplitMix64(14200)
    devs, dev_vars, ns = [], [], []
    for t in range(n_inst):
        n = 6 + rng.below(25)
        sg = math.exp(math.log(0.3) + (math.log(3.0) - math.log(0.3)) * rng.random())
        lm = math.exp(math.log(1e-3) + (math.log(1.0) - math.log(1e-3)) * rng.random())
        r = EV.loo_check(n, sg, lm, 14300 + t)
        devs.append(r["dev_residual"])
        dev_vars.append(r["dev_var"])
        ns.append(n)
    return {"n_instances": n_inst, "n_min": min(ns), "n_max": max(ns), "dev_residual_max": max(devs), "dev_residual_median": float(np.median(devs)),
            "dev_var_max": max(dev_vars), "dev_var_median": float(np.median(dev_vars))}


def build():
    out = {"n": N, "rules": list(RULES), "sets": {}}
    with Pool(4) as pool:
        for name, seed0, count in SETS:
            rows = pool.map(one_seed, [(N, seed0 + s) for s in range(count)])
            out["sets"][name] = {"seed0": seed0, "count": count, "rows": rows}
    out["loo_check"] = loo_instances()
    out["ops"] = {str(n): {"closed": M.flops_loo_closed(n), "removal": M.flops_loo_removal(n), "chol_alone": M.flops_chol(n)} for n in (20, 40, 80, 160)}
    out["target_variance"] = S.variance_of_target("v1")
    return out


def check():
    data = T.read_json("flaeche")
    bad = 0
    old_rows = data["sets"]["A"]["rows"]
    for s in range(2):
        r = one_seed((N, 5400 + s))
        old = old_rows[s]
        ok = all(abs(r["tests"][k] / old["tests"][k] - 1) < 0.05 for k in RULES) and abs(r["err_lin"] / old["err_lin"] - 1) < 0.05
        print(f"Zufalls-Seed {5400 + s}: Testfehler {[round(r['tests'][k], 4) for k in RULES]} gegen {[round(old['tests'][k], 4) for k in RULES]}: {'ok' if ok else 'ABWEICHUNG'}")
        bad += 0 if ok else 1
    print("CHECK", "bestanden" if bad == 0 else f"{bad} Abweichungen")
    return bad


def main():
    if T.wants_check():
        sys.exit(1 if check() else 0)
    T.write_json("flaeche", build())


if __name__ == "__main__":
    main()
