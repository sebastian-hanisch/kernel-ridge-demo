"""Container-Erzeuger: Regression der log-Verweilzeit als Gegenbefund neben dem Plan.

Nach der Beschreibung von `make_history` der `verweilzeit-demo` (lognormal, heteroskedastisch, Art, Kunde, Zollhalt, Wochenende, Vorlauf, ein Rauschmerkmal) in
dieser Demo selbst gebaut, mit SplitMix64 gezogen und NICHT kopiert: gleiche Verteilung, keine Bitgleichheit mit der Quelle (die Quelle zieht mit numpy). Die Konstanten
stehen in krd_constants.py; geprüft wird gegen gemessene Eigenschaften (Boden, Anteile, Form der Merkmale), nicht gegen die Quelle. Ein Modell, keine Messdaten."""

import math

import numpy as np

import krd_constants as C
from krd_rng import SplitMix64
from krd_scenario import gauss

_UK = None


def kunden_effekte():
    """Kundeneffekte auf die log-Verweilzeit: 12 Normalzahlen mit Streuung U_SD aus dem festen Zufalls-Seed KUNDEN_SEED (plattformfeste Normalzahl der Vorlage)."""
    global _UK
    if _UK is None:
        rng = SplitMix64(C.KUNDEN_SEED)
        _UK = tuple(C.U_SD * rng.normal() for _ in range(C.K_KUNDEN))
    return _UK


def log_median_and_s(art: int, kunde: int, zoll: int, we: int, vorlauf: float):
    """Exakte bedingte Verteilung der log-Verweilzeit: N(mu, s^2) (Orakel des Erzeugers)."""
    if art == 0:
        med = 3.5 * math.exp(0.7 * zoll + 0.30 * we)
    elif art == 1:
        med = 0.6 + 0.85 * vorlauf
    elif art == 2:
        med = 0.3 + 0.8 * vorlauf
    else:
        med = 7.0 * math.exp(0.30 * we)
    return math.log(med) + kunden_effekte()[kunde] * C.U_SCALE[art], C.S_BASE[art]


def make_container(n: int, seed: int):
    """n abgeschlossene Container: Merkmale (art, kunde, zoll, wochenende, vorlauf, gewicht), Verweilzeit in Tagen, Orakel (mu, s) der log-Verweilzeit.
    Die Verweilzeit wird nach den Merkmalen gezogen; `gewicht` ist ein reines Rauschmerkmal."""
    rng = SplitMix64(seed)
    feats, dwell, mus, ss = [], [], [], []
    for _ in range(n):
        art = rng.choice(C.ART_P)
        kunde = rng.below(C.K_KUNDEN)
        zoll = int(rng.random() < C.P_ZOLL and art == 0)
        tday = rng.random() * 7000.0
        we = int((int(tday) % 7) >= 5)
        if art == 1:
            vorlauf = 1.0 + 9.0 * rng.random()
        elif art == 2:
            vorlauf = 0.5 + 5.5 * rng.random()
        else:
            vorlauf = 0.0
        gewicht = gauss(rng)
        mu, s = log_median_and_s(art, kunde, zoll, we, vorlauf)
        d = math.exp(mu + s * gauss(rng))
        feats.append((art, kunde, zoll, we, vorlauf, gewicht))
        dwell.append(d)
        mus.append(mu)
        ss.append(s)
    return np.array(feats, dtype=float), np.array(dwell), np.array(mus), np.array(ss)


def encode(X):
    """Merkmale -> 20 Spalten: Art (4, One-Hot), Kunde (12, One-Hot), Zollhalt, Wochenende, Vorlauf / 10, Gewicht."""
    art = X[:, 0].astype(int)
    kd = X[:, 1].astype(int)
    n = len(X)
    oa = np.zeros((n, 4))
    oa[np.arange(n), art] = 1.0
    ok = np.zeros((n, C.K_KUNDEN))
    ok[np.arange(n), kd] = 1.0
    return np.column_stack([oa, ok, X[:, 2], X[:, 3], X[:, 4] / 10.0, X[:, 5]])


def floor_exact() -> float:
    """Boden des RMSE der log-Verweilzeit: Wurzel des Mittels der bedingten Varianz = sqrt(Summe p_Art * s_Art^2) (Orakel des Erzeugers)."""
    return math.sqrt(sum(p * s * s for p, s in zip(C.ART_P, C.S_BASE)))
