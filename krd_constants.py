"""Konstanten: Regler-Grenzen, feste Annahmen, Raster, Erzeuger-Parameter, Farben und Presets.

Alle Daten der Demo sind Erzeuger (Modelle), keine Messdaten. Zufallsquelle: SplitMix64 (krd_rng.py)."""

# ---- Modell: Zielfunktion der Plan-Regression (ausgedacht, glatt) ------------------------------------------------------------------------------
REGION = (-2.0, 2.0)               # Punkte gleichverteilt in [-2, 2]^2 (in der App als Liefergebiet um ein Depot in der Mitte gelesen)
NOISE_SD = 0.3                     # Rauschen der Beobachtungen
NOISE_VAR = NOISE_SD ** 2          # 0,09
N_TEST = 1000                      # Testpunkte für Überschussfehler und Abdeckung
N_TEST_WAYS = 200                  # Testpunkte der Wege-Anzeige (Experiment 1)

# ---- Raster der Wahl von (sigma, lambda) (Raster A = Plan-Raster 7 x 6, Raster B = zweites Raster 8 x 8) -----------------------------------------
SIGMAS_A = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)
LAMS_A = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
SIGMAS_B = (0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
LAMS_B = (1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)
# Evidenz per Gradientenverfahren: Grenzen = Rand von Raster A, sechs Startpunkte
GRAD_SIGMA = (SIGMAS_A[0], SIGMAS_A[-1])
GRAD_LAM = (LAMS_A[0], LAMS_A[-1])
GRAD_STARTS = tuple((s, lm) for s in (0.4, 1.6, 3.2) for lm in (1e-3, 1e-1))
FIXED_CELL = (1.6, 1e-2)           # beste feste Zelle der V1-Fläche (Regel „fest“ in Experiment 2)
LIN_LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0)     # lineare Ridge, Container
LIN_LAMS_PLAN = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)   # lineare Ridge, Plan-Regression

# ---- Container-Erzeuger (nach der Beschreibung von make_history der verweilzeit-demo; hier selbst mit SplitMix64 gezogen) -----------------------
ART_P = (0.35, 0.30, 0.25, 0.10)
K_KUNDEN = 12
KUNDEN_SEED = 12345
U_SD = 0.30
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
P_ZOLL = 0.25
N_CONT_COLS = 20                   # art 4, kunde 12 (One-Hot), zoll, wochenende, vorlauf/10, gewicht
CONT_FACTORS = (0.58, 1.0, 1.83, 3.16)   # Skalenfaktor f: sigma = f * sigma_Median (gamma = 3, 1, 0,3, 0,1 mal gamma_Median)
CONT_LAMS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
N_TEST_CONT = 4000

# ---- Regler (Bereich, Standard) ------------------------------------------------------------------------------------------------------------
SEED_MAX = 999999
N_W = (10, 30, 30)                 # Experiment 1: Punkte n (min, max, Standard); Bareiss-Referenz live bis n <= 40
SIGMA = (0.3, 6.0, 1.0)
LAM_EXP = (-10, 0, -1)             # lambda = 10^exp
SEED_W = (0, SEED_MAX, 14101)
N_SEEDS_W = (1, 20, 1)
N_BAND_OPTIONS = (20, 60, 200)
N_BAND_DEFAULT = 200
SEED_B = (0, SEED_MAX, 7)
N_L = (10, 60, 60)
SEED_L = (0, SEED_MAX, 5400)
N_C = (100, 300, 100)              # Experiment 4, live; n = 1 000 nur vorgerechnet
SEED_C = (0, SEED_MAX, 1)
HAND_X_OPTIONS = (-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0)
HAND_X_DEFAULT = 0.5
BAND_SEEDS = 30                    # Zufalls-Seeds je Zelle der vorgerechneten Band-Verteilung
BAREISS_LIMIT = 40                 # Bareiss-Referenz live nur bis zu dieser Größe

TRUTH_OPTIONS = ("V1-Funktion (glatt)", "Raue Funktion (Knicke)", "Aus dem Prior gezogen")
TRUTH_KEYS = {"V1-Funktion (glatt)": "v1", "Raue Funktion (Knicke)": "rau", "Aus dem Prior gezogen": "prior"}
RULE_OPTIONS = ("Evidenz (Raster 7 × 6)", "Evidenz (Gradient)", "Leave-one-out", "fest (σ = 1,6; λ = 0,01)")
RULE_KEYS = {"Evidenz (Raster 7 × 6)": "ev_grid", "Evidenz (Gradient)": "ev_grad", "Leave-one-out": "loo", "fest (σ = 1,6; λ = 0,01)": "fixed"}
RULE3_OPTIONS = ("Leave-one-out", "GCV", "Evidenz (Raster 7 × 6)", "Evidenz (Gradient)")
RULE3_KEYS = {"Leave-one-out": "loo", "GCV": "gcv", "Evidenz (Raster 7 × 6)": "ev_grid", "Evidenz (Gradient)": "ev_grad"}
TARGET_OPTIONS = ("V1-Funktion (glatt)", "Raue Funktion (Knicke)")
VEHICLE_OPTIONS = ("Plan-Regression (2D)", "Container-Erzeuger")

WAY_KEYS = ("cholesky", "lu", "eig", "min")
WAY_LABELS = {"cholesky": "Cholesky", "lu": "LU (numpy.solve)", "eig": "Eigenzerlegung", "min": "Minimierung (L-BFGS-B)"}
MIN_MAXITER = 5000                 # Obergrenze der Iterationen der Minimierung in der Live-App (gemessen: wo sie abbricht, steht ABNORMAL)

# ---- Dateien der vorgerechneten Messungen ---------------------------------------------------------------------------------------------------
DATA_FILES = {"wege": "krd_wege.json", "flaeche": "krd_flaeche.json", "band": "krd_band.json", "lernkurve": "krd_lernkurve.json", "stuetz": "krd_stuetz.json"}

COLOR_TRUTH = "#4c78a8"
COLOR_MEAN = "#2F6B65"
COLOR_BAND = "rgba(47,107,101,0.20)"
COLOR_POINTS = "#c77700"
COLOR_BAD = "#b04a3b"
COLOR_GOOD = "#2e7d4f"
COLOR_GRAY = "rgba(150,150,150,0.8)"

PRESETS = {
    "Gut gestellt (σ = 1)": {},
    "Schlecht gestellt (σ = 4, λ = 1e-10)": {"sigma": 4.0, "lam_exp": -10},
    "Band bei n = 60": {"n_band": 60},
    "Band bei n = 20": {"n_band": 20},
    "Container n = 300": {"vehicle": "Container-Erzeuger", "n_c": 300},
    "Handbeispiel (3 Punkte)": {"hand_x": 2.0},
}
PRESET_HELP = {
    "Gut gestellt (σ = 1)": "Gauß-Kern mit σ = 1 und λ = 0,1 auf 30 Punkten der Plan-Regression: alle direkten Wege stimmen mit der exakten Lösung auf Rundungsniveau überein, die Minimierung auf etwa 1e-8 (im Einzelfall schlechter).",
    "Schlecht gestellt (σ = 4, λ = 1e-10)": "σ = 4 und λ = 1e-10: die Gram-Matrix ist numerisch fast singulär. Die direkten Wege weichen von der exakten Lösung ab, und keiner meldet etwas.",
    "Band bei n = 60": "Das 95-%-Band des GP bei 60 Trainingspunkten, Wahl von σ und λ durch die Evidenz: Abdeckung der Funktion über 30 Zufalls-Seeds.",
    "Band bei n = 20": "Dasselbe Band bei nur 20 Trainingspunkten: deutlich zu eng, besonders bei Wahl durch Leave-one-out.",
    "Container n = 300": "Kern gegen lineare Ridge auf dem Container-Erzeuger mit 300 Trainingscontainern: der Kern bringt hier wenig.",
    "Handbeispiel (3 Punkte)": "Drei Punkte, Polynomkern Grad 2, alles in Brüchen: Testpunkt x* = 2 liegt außerhalb der Daten, die Varianz bleibt nahe der Prior-Varianz.",
}
