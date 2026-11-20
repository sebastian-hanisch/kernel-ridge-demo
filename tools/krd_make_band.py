"""Messung zu Experiment 2: Abdeckung des 95-%-Bandes der FUNKTION (nicht der verrauschten Beobachtung).

Schreibt krd_band.json: drei Sätze zu je 30 Zufalls-Seeds (Zufalls-Seed = Satzbasis + 1000 n + s) für jede Zelle (Wahrheit: V1-Funktion, raue Funktion, aus dem Prior gezogen) x
(n = 20, 60, 200) x (Regel: Evidenz auf dem Raster 7 x 6, Evidenz per Gradient, Leave-one-out, feste Zelle); dazu das Kalibrierungs-Orakel (Wahrheit aus dem Prior, Hyperparameter
bekannt, 300 Zufalls-Seeds). Zähleinheiten: Zufalls-Seeds, Testpunkte (1 000, beim Prior 100). Keine Sekunden als Befund.
Aufruf: python tools/krd_make_band.py [--check]"""

import sys
from multiprocessing import Pool

import _krd_common as T


import krd_constants as C
import krd_evaluation as EV

SET_BASES = (14500, 17000, 21000)
N_SEEDS = C.BAND_SEEDS
TRUTHS = ("v1", "rau", "prior")
NS = C.N_BAND_OPTIONS
RULES = ("ev_grid", "ev_grad", "loo", "fixed")
ORACLE_SEEDS = 300
ORACLE_SEED0 = 14400


def one(args):
    truth, n, seed = args
    out = {}
    for rule in RULES:
        r = EV.band_run(truth, n, seed, rule, with_slice=False)
        out[rule] = (r["coverage"], r["z2"], r["sigma"], r["lam"])
    return args, out


def oracle(s):
    return EV.prior_oracle(ORACLE_SEED0 + s)


def build():
    tasks = [(truth, n, base + 1000 * n + s) for base in SET_BASES for truth in TRUTHS for n in NS for s in range(N_SEEDS)]
    with Pool(4) as pool:
        results = pool.map(one, tasks, chunksize=3)
        orc = pool.map(oracle, range(ORACLE_SEEDS), chunksize=10)
    sets = [{"seed_base": base, "cells": {}} for base in SET_BASES]
    for (truth, n, seed), out in results:
        k = SET_BASES.index(next(b for b in SET_BASES if 0 <= seed - b - 1000 * n < N_SEEDS))
        for rule in RULES:
            cell = sets[k]["cells"].setdefault(f"{truth}|{n}|{rule}", {"coverage": [], "z2": [], "sigma": [], "lam": []})
            cov, z2, sg, lm = out[rule]
            cell["coverage"].append(cov)
            cell["z2"].append(z2)
            cell["sigma"].append(sg)
            cell["lam"].append(lm)
    return {"n_seeds": N_SEEDS, "sets": sets, "oracle": {"n": 30, "n_test": 100, "seed0": ORACLE_SEED0, "coverage": [o["coverage"] for o in orc], "z2": [o["z2"] for o in orc]}}


def check():
    data = T.read_json("band")
    bad = 0
    for truth, n, s in (("v1", 60, 0), ("rau", 20, 1)):
        seed = SET_BASES[0] + 1000 * n + s
        _, out = one((truth, n, seed))
        for rule in RULES:
            old = data["sets"][0]["cells"][f"{truth}|{n}|{rule}"]["coverage"][s]
            ok = abs(out[rule][0] - old) < 0.03
            print(f"{truth} n = {n} Zufalls-Seed {seed} {rule}: Abdeckung {out[rule][0]:.3f} gegen {old:.3f}: {'ok' if ok else 'ABWEICHUNG'}")
            bad += 0 if ok else 1
    print("CHECK", "bestanden" if bad == 0 else f"{bad} Abweichungen")
    return bad


def main():
    if T.wants_check():
        sys.exit(1 if check() else 0)
    T.write_json("band", build())


if __name__ == "__main__":
    main()
