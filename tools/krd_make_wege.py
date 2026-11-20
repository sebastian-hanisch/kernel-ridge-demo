"""Messung zu Experiment 1: Rechenwege gegen die exakte Lösung (Bareiss) bei n = 30, Gauß-Kern.

Schreibt krd_wege.json: (a) Tabelle über lambda für sigma = 1 und sigma = 4 bei einem Zufalls-Seed (Standard 14101), sechs Wege (Cholesky, LU, Eigenzerlegung, Minimierung,
sklearn KernelRidge, sklearn GaussianProcessRegressor; die beiden sklearn-Wege nur hier und im Test, nie in der Live-App); (b) Band über 20 Zufalls-Seeds für vier Einstellungen.
Zähleinheit: Zufalls-Seeds; Maß: größter Betrag der Abweichung der Vorhersage über 200 Testpunkte gegen die exakte Lösung des Gleitkomma-Systems. Keine Sekunden als Befund.
Aufruf: python tools/krd_make_wege.py [--check]   (--check rechnet drei Zufalls-Seeds einer Einstellung neu und vergleicht mit der Datei)."""

import sys
import warnings

import _krd_common as T

import numpy as np

import krd_constants as C
import krd_exact as E
import krd_model as M
import krd_scenario as S

N = 30
TABLE_SEED = 14101
BAND_SEED0 = 16000
BAND_SEEDS = 20
LAMS = (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
BAND_SETTINGS = ((1.0, 1e-10), (1.0, 1e-6), (4.0, 1e-10), (4.0, 1e-6))
WAYS6 = ("cholesky", "lu", "eig", "min", "sk_kr", "sk_gp")


def six_ways(seed: int, sigma: float, lam: float):
    """Alle sechs Wege für einen Zufalls-Seed. Rückgabe (cond, Fehler je Weg, Meldung je Weg, Abstand der direkten Wege, größter Fehler der direkten Wege)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.kernel_ridge import KernelRidge

    X, y, _ = S.make_plan(N, seed)
    Xs, _ = S.make_test(C.N_TEST_WAYS, seed + 1_000_000)
    yc = y - y.mean()
    K = M.gram(X, X, sigma)
    Ks = M.gram(Xs, X, sigma)
    A = K + lam * np.eye(N)
    f_ex = Ks @ E.exact_solve(A, yc)
    preds, notes = {}, {}
    for w in ("cholesky", "lu", "eig", "min"):
        a, note = M.run_way(w, K, yc, lam, C.MIN_MAXITER)
        preds[w] = None if a is None else Ks @ a
        notes[w] = note
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        preds["sk_kr"] = KernelRidge(alpha=lam, kernel="rbf", gamma=1.0 / (2.0 * sigma ** 2)).fit(X, yc).predict(Xs)
        notes["sk_kr"] = f"Warnung: {str(wl[0].message)[:60]}" if wl else None
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0, "fixed") * RBF(sigma, "fixed"), alpha=lam, optimizer=None).fit(X, yc)
        preds["sk_gp"] = gp.predict(Xs)
        notes["sk_gp"] = f"Warnung: {str(wl[0].message)[:60]}" if wl else None
    err = {w: (None if preds[w] is None else float(np.abs(preds[w] - f_ex).max())) for w in WAYS6}
    direct = [preds[w] for w in WAYS6 if w != "min" and preds[w] is not None]
    spread = max(float(np.abs(a - b).max()) for a in direct for b in direct)
    worst = max(err[w] for w in WAYS6 if w != "min" and err[w] is not None)
    return float(np.linalg.cond(A)), err, notes, spread, worst


def build_table():
    out = {"seed": TABLE_SEED, "n": N, "lams": list(LAMS), "sigmas": {}}
    for sigma in (1.0, 4.0):
        conds, errs, notes, spreads = [], {w: [] for w in WAYS6}, {w: [] for w in WAYS6}, []
        for lam in LAMS:
            cond, err, note, spread, _ = six_ways(TABLE_SEED, sigma, lam)
            conds.append(cond)
            spreads.append(spread)
            for w in WAYS6:
                errs[w].append(err[w])
                notes[w].append(note[w])
        out["sigmas"][f"{sigma:g}"] = {"cond": conds, "err": errs, "notes": notes, "spread": spreads}
    return out


def build_band():
    out = {"seed0": BAND_SEED0, "n_seeds": BAND_SEEDS, "n": N, "settings": {}}
    for sigma, lam in BAND_SETTINGS:
        per = []
        for s in range(BAND_SEEDS):
            cond, err, note, spread, worst = six_ways(BAND_SEED0 + s, sigma, lam)
            silent = all(note[w] is None for w in WAYS6 if w != "min")
            per.append({"cond": cond, "err": err, "silent": silent, "min_note": note["min"], "ratio": (spread / worst) if worst > 0 else None})
        out["settings"][f"{sigma:g}|{lam:g}"] = {"sigma": sigma, "lam": lam, "per_seed": per}
    return out


def check():
    data = T.read_json("wege")
    bad = 0
    key = "4|1e-10"
    for s in range(3):
        cond, err, _, _, _ = six_ways(BAND_SEED0 + s, 4.0, 1e-10)
        old = data["band"]["settings"][key]["per_seed"][s]
        ok = abs(cond / old["cond"] - 1) < 0.05 and all(abs(np.log10(err[w] / old["err"][w])) < 1.0 for w in WAYS6 if w != "min" and err[w] is not None)
        print(f"Zufalls-Seed {BAND_SEED0 + s}: cond {cond:.2e} gegen {old['cond']:.2e}; Fehler {'ok' if ok else 'ABWEICHUNG'}")
        bad += 0 if ok else 1
    print("CHECK", "bestanden" if bad == 0 else f"{bad} Abweichungen")
    return bad


def main():
    if T.wants_check():
        sys.exit(1 if check() else 0)
    T.write_json("wege", {"table": build_table(), "band": build_band()})


if __name__ == "__main__":
    main()
