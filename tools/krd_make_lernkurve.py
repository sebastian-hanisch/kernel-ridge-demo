"""Messung zu Experiment 4: Kernel Ridge gegen lineare Ridge auf dem Container-Erzeuger (Regression der log-Verweilzeit), n = 100, 300, 1 000.

Schreibt krd_lernkurve.json: 20 Zufalls-Seeds je n, gepaart (dieselben Trainings- und Testdaten für beide Verfahren), 4 000 frische Testcontainer je Zufalls-Seed; Kernel Ridge mit
Skalenfaktor f und lambda per geschlossenem Leave-one-out auf dem Raster, lineare Ridge ebenso. Der Boden ist exakt sqrt(Summe p_Art s_Art^2) (Orakel des Erzeugers); das
Stichprobenmittel der Zufalls-Seeds steht daneben. Zähleinheiten: Zufalls-Seeds (gepaart), Testpunkte. Maß: RMSE in log-Tagen. Keine Sekunden als Befund.
Aufruf: python tools/krd_make_lernkurve.py [--check]"""

import sys
from multiprocessing import Pool

import _krd_common as T

import krd_evaluation as EV

NS = (100, 300, 1000)
N_SEEDS = 20


def one(args):
    n, s = args
    return args, EV.container_run(n, s)


def build():
    tasks = [(n, s) for n in NS for s in range(N_SEEDS)]
    with Pool(4) as pool:
        res = pool.map(one, tasks, chunksize=2)
    out = {"n_seeds": N_SEEDS, "ns": list(NS), "runs": {str(n): [] for n in NS}}
    for (n, s), r in res:
        out["runs"][str(n)].append(r)
    return out


def check():
    data = T.read_json("lernkurve")
    bad = 0
    for s in range(2):
        r = EV.container_run(100, s)
        old = data["runs"]["100"][s]
        ok = abs(r["rmse_krr"] - old["rmse_krr"]) < 5e-3 and abs(r["rmse_lin"] - old["rmse_lin"]) < 5e-3
        print(f"n = 100, Zufalls-Seed {s}: RMSE Kern {r['rmse_krr']:.4f} gegen {old['rmse_krr']:.4f}, linear {r['rmse_lin']:.4f} gegen {old['rmse_lin']:.4f}: {'ok' if ok else 'ABWEICHUNG'}")
        bad += 0 if ok else 1
    print("CHECK", "bestanden" if bad == 0 else f"{bad} Abweichungen")
    return bad


def main():
    if T.wants_check():
        sys.exit(1 if check() else 0)
    T.write_json("lernkurve", build())


if __name__ == "__main__":
    main()
