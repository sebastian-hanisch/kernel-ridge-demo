"""Plan-Erzeuger: ausgedachte Zielfunktionen über dem Plan [-2, 2]^2, Punkte, Rauschen, ±1-Labels des Kreises und Ziehungen aus dem GP-Prior.

Alles ist ein Modell (kein Messdatum). Zufallsquelle: SplitMix64 (krd_rng.py); Normalzahlen als `NormalDist().inv_cdf(random())` (schnell; in den Rändern über math.log,
deshalb sind Gleitkommagrößen daraus im Test Bänder und keine exakten Werte)."""

import math
from statistics import NormalDist

import numpy as np

import krd_constants as C
from krd_rng import SplitMix64

_INV_CDF = NormalDist().inv_cdf
_U_MIN = 2.0 ** -54


def gauss(rng: SplitMix64) -> float:
    """Standardnormalzahl über die Inverse der Verteilungsfunktion (eine gleichverteilte Zahl je Normalzahl)."""
    u = rng.random()
    return _INV_CDF(min(max(u, _U_MIN), 1.0 - _U_MIN))


def target_v1(X):
    """Glatte Zielfunktion der Vormessung V1: sin(1,5 x1) + 0,5 cos(2,5 x2) + 0,3 x1 x2 über [-2, 2]^2."""
    X = np.atleast_2d(X)
    return np.sin(1.5 * X[:, 0]) + 0.5 * np.cos(2.5 * X[:, 1]) + 0.3 * X[:, 0] * X[:, 1]


def target_rough(X):
    """Raue Zielfunktion mit Knicken: |x1| + 0,5 |x2| + 0,3 sin(3 x1)."""
    X = np.atleast_2d(X)
    return np.abs(X[:, 0]) + 0.5 * np.abs(X[:, 1]) + 0.3 * np.sin(3.0 * X[:, 0])


TARGETS = {"v1": target_v1, "rau": target_rough}


def uniform_points(rng: SplitMix64, n: int, lo: float = C.REGION[0], hi: float = C.REGION[1]):
    """n Punkte gleichverteilt in [lo, hi]^2 (zwei Ziehungen je Punkt)."""
    return np.array([[lo + (hi - lo) * rng.random(), lo + (hi - lo) * rng.random()] for _ in range(n)]).reshape(n, 2)


def make_plan(n: int, seed: int, target: str = "v1", noise_sd: float = C.NOISE_SD):
    """Plan-Regression: n Trainingspunkte, y = Zielfunktion + Rauschen (sd noise_sd). Rückgabe (X, y, f) mit f = rauschfreie Wahrheit an den Trainingspunkten."""
    rng = SplitMix64(seed)
    X = uniform_points(rng, n)
    f = TARGETS[target](X)
    y = f + noise_sd * np.array([gauss(rng) for _ in range(n)])
    return X, y, f


def make_test(n_test: int, seed: int, target: str = "v1"):
    """Testpunkte und rauschfreie Wahrheit."""
    X = uniform_points(SplitMix64(seed), n_test)
    return X, TARGETS[target](X)


def variance_of_target(target: str = "v1", n: int = 200000, seed: int = 1) -> float:
    """Varianz der Zielfunktion über dem Plan (Fehler eines konstanten Mittelwerts), aus n Punkten."""
    X = uniform_points(SplitMix64(seed), n)
    return float(TARGETS[target](X).var())


def draw_from_prior(n: int, n_test: int, seed: int, sigma: float = 1.0, noise_var: float = C.NOISE_VAR, X_extra=None):
    """Wahrheit aus dem GP-Prior (Gauß-Kern sigma, Prior-Varianz 1) an n + n_test (+ extra) Punkten gezogen; Beobachtungen y = f + N(0, noise_var).
    Rückgabe (X_train, y, X_test, f_test, f_extra); f_extra ist die Funktion an den Zusatzpunkten X_extra (oder None). Die Ziehung benutzt die Eigenzerlegung
    der Gram-Matrix (negative Eigenwerte auf 0 gesetzt)."""
    from krd_model import gram
    rng = SplitMix64(seed)
    Xa = uniform_points(rng, n + n_test)
    if X_extra is not None:
        Xa = np.vstack([Xa, np.atleast_2d(X_extra)])
    m = len(Xa)
    w, V = np.linalg.eigh(gram(Xa, Xa, sigma))
    z = np.array([gauss(rng) for _ in range(m)])
    f = V @ (np.sqrt(np.maximum(w, 0.0)) * z)
    y = f[:n] + math.sqrt(noise_var) * np.array([gauss(rng) for _ in range(n)])
    f_extra = f[n + n_test:] if X_extra is not None else None
    return Xa[:n], y, Xa[n:n + n_test], f[n:n + n_test], f_extra


CIRCLE_RADIUS = math.sqrt(2.0 / math.pi)       # Kreis mit halber Fläche in [-1, 1]^2


def make_circle(n: int, seed: int, flip: float = 0.1):
    """±1-Labels: innerhalb des Kreises mit halber Fläche +1, sonst -1; jedes Label mit Wahrscheinlichkeit flip vertauscht. Rückgabe (X, y_pm1)."""
    rng = SplitMix64(seed)
    X = uniform_points(rng, n, -1.0, 1.0)
    inside = (np.sqrt((X ** 2).sum(1)) <= CIRCLE_RADIUS).astype(int)
    swap = np.array([rng.random() < flip for _ in range(n)])
    y = np.where(swap, 1 - inside, inside) * 2 - 1
    return X, y.astype(float)


def circle_truth(X):
    """Wahre ±1-Labels des Kreises (ohne Vertauschung)."""
    return (np.sqrt((np.atleast_2d(X) ** 2).sum(1)) <= CIRCLE_RADIUS).astype(int) * 2 - 1
