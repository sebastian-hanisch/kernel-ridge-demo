"""Messung zu Befund H: keine Stützvektoren. Kernel Ridge auf ±1-Etiketten (Kreis mit halber Fläche, 10 % vertauschte Etiketten) hat keine Nullen in alpha;
SVC (scikit-learn, nur in dieser Messwerkstatt und im Test) mit demselben Kern hat Stützvektoren.

Schreibt krd_stuetz.json: 50 Instanzen (n = 100, Zufalls-Seeds 14600 + s), Gauß-Kern sigma = 0,4, Kernel Ridge lambda = 0,3, SVC C = 1; feste, nicht abgestimmte Einstellungen
(kein Verfahrensvergleich). Zähleinheit: Instanzen, Koeffizienten (Nullen mit Rauschgrenze 1e-12 des größten Betrags), Testpunkte (4 000, rauschfreie Etiketten).
Aufruf: python tools/krd_make_stuetz.py [--check]"""

import sys

import _krd_common as T

import numpy as np

import krd_evaluation as EV
import krd_model as M
import krd_scenario as S

N_INST = 50
N = 100
SEED0 = 14600
SIGMA, LAM, C_SVC = 0.4, 0.3, 1.0


def one(s: int):
    from sklearn.svm import SVC
    X, y = S.make_circle(N, SEED0 + s)
    Xt, _ = S.make_test(4000, 800000 + s)
    Xt = Xt / 2.0                                                    # Testpunkte in [-1, 1]^2
    yt = S.circle_truth(Xt)
    K = M.gram(X, X, SIGMA)
    alpha = np.linalg.solve(K + LAM * np.eye(N), y)
    err_r = float(np.mean(np.sign(M.gram(Xt, X, SIGMA) @ alpha) != yt))
    sv = SVC(kernel="rbf", gamma=1.0 / (2.0 * SIGMA ** 2), C=C_SVC).fit(X, y)
    err_s = float(np.mean(sv.predict(Xt) != yt))
    return {"zeros": EV.zero_count(alpha), "sv_share": len(sv.support_) / N, "err_ridge": err_r, "err_svc": err_s}


def check():
    data = T.read_json("stuetz")
    bad = 0
    for s in range(3):
        r = one(s)
        old = data["per_instance"][s]
        ok = r["zeros"] == old["zeros"] and abs(r["sv_share"] - old["sv_share"]) < 1e-9 and abs(r["err_ridge"] - old["err_ridge"]) < 5e-3
        print(f"Instanz {s}: Nullen {r['zeros']} gegen {old['zeros']}, Stützvektoranteil {r['sv_share']} gegen {old['sv_share']}: {'ok' if ok else 'ABWEICHUNG'}")
        bad += 0 if ok else 1
    print("CHECK", "bestanden" if bad == 0 else f"{bad} Abweichungen")
    return bad


def main():
    if T.wants_check():
        sys.exit(1 if check() else 0)
    per = [one(s) for s in range(N_INST)]
    T.write_json("stuetz", {"n": N, "n_instances": N_INST, "seed0": SEED0, "sigma": SIGMA, "lam": LAM, "C": C_SVC, "per_instance": per})


if __name__ == "__main__":
    main()
