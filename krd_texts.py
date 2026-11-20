"""Texte der App (Einleitung, Erklärungen, Annahmen, Formeln). Zahlen stehen hier nur, wenn sie aus der Konstantendatei oder aus den vorgerechneten Messungen kommen
(die App setzt sie dort ein); jede Zahl der README steht in tests/test_claims.py."""

TITLE = "Kernel Ridge und das GP-Mittel: dieselbe Gleichung, zwei Aussagen"

INTRO = """
**Kernel Ridge** und das **Mittel eines Gauß-Prozesses** (GP) lösen dieselbe Gleichung (K + λI) α = y, wenn λ = σ_n²/c ist (Rauschvarianz durch Prior-Varianz). Diese Demo
prüft an zwei ausgedachten Modellen (eine glatte Zielfunktion über einem Plan, ein Erzeugermodell für Container-Verweilzeiten), **ob die Rechenwege wirklich dasselbe liefern
und was sie melden**, **was der GP zusätzlich liefert (ein 95-%-Band) und ob es stimmt**, **wie man σ und λ wählt, ohne n-mal zu rechnen**, und **wann der Kern gegenüber einem
linearen Modell etwas bringt**. Alle Daten sind Erzeuger (Modelle), keine Messdaten; „Kernel“ steht hier für die Methode, „Kern“ für die Kernfunktion.

Weiter unten: vier Experimente, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

HOW = """
1. **Kernel Ridge** minimiert die Summe der quadratischen Abweichungen plus λ mal die Norm der Funktion; mit f = Σ α_j k(·, x_j) ist das ein lineares Gleichungssystem
   (K + λI) α = y, und die Vorhersage ist f(x*) = k*ᵀ α.
2. **Der GP** mit Prior-Mittel 0, Prior-Varianz c und Rauschvarianz σ_n² hat dasselbe Mittel, wenn λ = σ_n²/c ist. **Zusätzlich** liefert er eine Varianz der Funktion
   (Rasmussen/Williams 2006, Gl. 2.24 und Gl. 2.26), also ein Band; Kernel Ridge liefert nur einen Schätzer.
3. **Gleichheit der Gleichung ist keine Richtigkeit der Rechnung**: bei schlecht gestellter Matrix weichen die Rechenwege von der exakten Lösung ab, ohne etwas zu melden.
"""

Q1 = "❓ Ist das wirklich dieselbe Rechnung, und was melden die Rechenwege?"
Q2 = "❓ Was gibt es beim GP zusätzlich, und stimmt es?"
Q3 = "❓ Wie wählt man σ und λ, ohne n-mal zu rechnen?"
Q4 = "❓ Wann bringt der Kern gegenüber linear etwas?"

CAP_LIVE_VS_STUDY = ("**Live-Modell ≠ eingefrorene Messung:** die Seite rechnet live vier Wege selbst (Cholesky, LU, Eigenzerlegung, Minimierung) gegen die exakte Lösung (Bareiss, ganze Zahlen). "
                     "Die vorgerechneten Zahlen (Expander unten) enthalten zusätzlich die beiden Wege aus scikit-learn und 20 Zufalls-Seeds; scikit-learn läuft nie in der App. "
                     "Die Abweichung zwischen beiden steht in der README.")

CAP_WEGE = ("Plan-Regression, Gauß-Kern, n Punkte in [-2, 2]², Rauschen sd 0,3, Ziel um das Trainingsmittel zentriert (Schalter in der Seitenleiste). Maß: größter Betrag der Abweichung der "
            "Vorhersage über 200 Testpunkte gegen die **exakte** Lösung des Gleitkomma-Systems; die Skala der Vorhersage (größter Betrag) ist etwa 1 bis 2. Fehlerziffern sind Bänder und hängen an der Plattform.")

CAP_BAND = ("Das Band gehört zur **Funktion** f an den Testpunkten, nicht zur verrauschten Beobachtung. Abdeckung = Anteil der Testpunkte, an denen die Wahrheit im 95-%-Band liegt "
            "(1 000 Testpunkte, beim Prior 100). Mit „Aus dem Prior gezogen“ ist die Wahl der Hyperparameter Teil der Frage, nicht nur der Prior.")

CAP_LOO = ("Leave-one-out geschlossen: Residuum e_i = α_i / [(K + λI)⁻¹]_ii, Varianz 1 / [(K + λI)⁻¹]_ii (Rasmussen/Williams Gl. 5.10 bis Gl. 5.12). Die Fläche zeigt den Überschussfehler "
           "(mittlerer quadratischer Fehler gegen die rauschfreie Wahrheit auf 1 000 Testpunkten) für ein Raster 7 × 6; σ und λ der Seitenleiste gelten für die Prüfung oben.")

CAP_KERN = ("Container-Erzeuger: Regression der log-Verweilzeit; Kernel Ridge mit Skalenfaktor (σ = Skalenfaktor mal Median der paarweisen Abstände der standardisierten Merkmale) und λ "
            "per geschlossenem Leave-one-out gegen lineare Ridge (λ ebenso). Ein Modell, keine Messdaten; in den gelesenen Studien liegen für Verweilzeit und Nachfrage Bäume vorn.")

ASSUMPTIONS = """
| Annahme | Was passiert, wenn sie verletzt ist | Wo gemessen |
|---|---|---|
| **Gauß-Kern allein**, Eingabe in 2 Dimensionen (Container: 20 Spalten) | Andere Kerne (Matérn, periodisch) und der Rückfall auf den Prior außerhalb der Daten sind nicht Gegenstand dieser Demo | Kernwahl-Demo der Reihe |
| **Gut gestellte Matrix** | Bei σ = 4 und λ = 1e-10 weichen die direkten Wege von der exakten Lösung ab (Band über 20 Zufalls-Seeds in Experiment 1), und in keinem dieser Läufe meldet einer etwas. Das ist die milde Form; dass die Zerlegung gelingt und die Lösung falsch ist, zeigt die Bandbreite-und-Kondition-Demo der Reihe | Experiment 1 |
| **Wahrheit ausgedacht, Raster 7 × 6, Zentrierung** | Andere Zielfunktionen, andere Raster und der Verzicht auf Zentrierung (dann fällt das Mittel fern der Daten auf 0 statt auf das Trainingsmittel) ändern Zahlen; gemessen sind nur die hier genannten | Experimente 2 und 3 |
| **Wahl von σ und λ durch Evidenz oder Leave-one-out** | Das Band ist bei Wahl durch Leave-one-out und bei n = 20 enger als versprochen; Leave-one-out und GCV per Gradient sind nicht gemessen, ebenso mehrere Bandbreiten je Merkmal | Experiment 2 und 3 |
| **Container-Erzeuger statt Betriebsdaten** | Gilt für diesen Erzeuger und Ridge-Regression; größere n (etwa 3 000) und andere Lerner (Bäume) sind nicht gemessen; ein Nutzen in Logistik, Lager oder Fertigung ist nicht belegt | Experiment 4 |
| **Nur das Mittel ist gleich** | Die Varianz gibt es nur beim GP; Kriging-Varianten, konforme Bänder und Stützvektor-Regression sind nicht gemessen und nicht angeboten | Experimente 1 und 2 |
"""

LIMITS = """
**Annahmen:** Gauß-Kern, ausgedachte glatte Zielfunktion (und eine raue), Rauschen sd 0,3, Raster 7 × 6 für (σ, λ), Container-Erzeuger nach der Beschreibung einer anderen Demo, selbst gebaut.

**Grenzen:** Alle Zahlen gelten für diese Modelle, Einstellungen und Zufalls-Seeds, mit der Zahl der Zufalls-Seeds neben der Zahl. Es gibt keine reale Datenquelle. Belegt ist der Einsatz von
Kriging als Ersatzmodell einer Simulation; für Verweilzeit und Nachfrage liegen in den gelesenen Studien Bäume vorn. Was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet.

**Bekannte Probleme und Unsicherheiten:** Die Fehlerziffern der Rechenwege (Experiment 1) sind plattformabhängig und stehen nur als Bänder; gemessen ist Windows mit Python 3.14, ein Linux-Lauf
steht aus. Die Minimierung ist auf 5 000 Iterationen begrenzt (die Messung der Planung ließ 20 000 zu). Die Standardfehler der Bänder stammen aus 30 bis 40 Zufalls-Seeds je Satz.
"""

LESEHILFE = """
**Lesehilfe.** Gegeben sind n Beobachtungen (x_i, y_i) mit x_i ∈ ℝ² und y_i ∈ ℝ. Gesucht ist die Vorhersage f(x*) an einem neuen Punkt x* und, beim GP, die Unsicherheit der Funktion dort.

| Symbol | Bedeutung |
|---|---|
| σ | Bandbreite des Gauß-Kerns (Längenmaß); γ = 1/(2σ²) wird nur genannt |
| λ | Regularisierung von Kernel Ridge; beim GP λ = σ_n²/c |
| σ_n² | Rauschvarianz des GP (Varianz der Beobachtungen um die Funktion) |
| c | Prior-Varianz des GP (Skala des Kerns) |
| μ | Regularisierung in der Fassung mit Mittel über n Terme; λ = n μ |
| K, k* | Gram-Matrix K_ij = k(x_i, x_j) und Vektor k*_j = k(x*, x_j) |
| α | Koeffizienten, α = (K + λI)⁻¹ y; Vorhersage f(x*) = k*ᵀ α |
| H | Hat-Matrix K (K + λI)⁻¹, Spur H = effektive Parameterzahl |
| Zentrierung | Trainingsmittel von y abziehen und wieder addieren (Standard hier); ohne sie fällt das Mittel fern der Daten auf 0 |
"""

MATH_A = r"""
**Kern und Gleichung.** Gauß-Kern mit Bandbreite $\sigma > 0$ (und $\lambda \ge 0$, $n \ge 1$):
"""
MATH_KERNEL = r"k(x, x') = \exp\!\left(-\frac{\lVert x - x' \rVert^2}{2\sigma^2}\right)"
MATH_B = r"""
Kernel Ridge minimiert über $f = \sum_j \alpha_j\, k(\cdot, x_j)$ die Summe der quadratischen Abweichungen plus $\lambda \lVert f \rVert_{\mathcal H}^2$; die Lösung ist das lineare System
"""
MATH_ALPHA = r"\alpha = (K + \lambda I)^{-1} y, \qquad f(x_*) = k_*^{\top} \alpha"
MATH_C = r"""
**GP.** Prior $c \cdot k$, Prior-Mittel 0, Rauschen $\mathcal N(0, \sigma_n^2)$ (Rasmussen/Williams 2006, Gl. 2.22 bis Gl. 2.26): das Mittel ist
"""
MATH_GPMEAN = r"\bar f(x_*) = c\, k_*^{\top} (c K + \sigma_n^2 I)^{-1} y"
MATH_D = r"""
und gleich dem von Kernel Ridge, wenn $\lambda = \sigma_n^2 / c$ ist. Die Varianz der Funktion gibt es nur beim GP:
"""
MATH_VAR = r"\operatorname{Var}\, f(x_*) = c \left( k(x_*, x_*) - k_*^{\top} (K + \lambda I)^{-1} k_* \right)"
MATH_E = r"""
**Leave-one-out ohne $n$-mal Rechnen** (Gl. 5.10 bis Gl. 5.12, mit $c = 1$): das Residuum beim Weglassen von Punkt $i$ und die Varianz der Vorhersage von $y_i$ sind
"""
MATH_LOO = r"e_i = \frac{\alpha_i}{[(K + \lambda I)^{-1}]_{ii}}, \qquad \operatorname{Var}_i = \frac{1}{[(K + \lambda I)^{-1}]_{ii}}"
MATH_F = r"""
**Wahlregeln** für $(\sigma, \lambda)$: Leave-one-out (PRESS = Mittel der $e_i^2$), GCV und die profilierte log-Evidenz mit Skalenfaktor $\hat s^2 = y^{\top}(K + \lambda I)^{-1} y / n$:
"""
MATH_GCV = r"\mathrm{GCV} = \frac{\frac{1}{n} \lVert y - H y \rVert^2}{\left(1 - \operatorname{tr} H / n\right)^2}, \qquad H = K (K + \lambda I)^{-1}"
MATH_EVID = r"\log p(y) = -\tfrac{n}{2} \log\!\left(2 \pi \hat s^2\right) - \tfrac{1}{2} \log \lvert K + \lambda I \rvert - \tfrac{n}{2}"
MATH_G = r"""
**Konventionen und Namensfallen.** `KernelRidge` von scikit-learn nennt unser $\lambda$ `alpha`. In der Fassung $\frac{1}{n}\sum_i (y_i - f(x_i))^2 + \mu \lVert f \rVert^2$ ist $\lambda = n\,\mu$; wer $\mu$ als
$\lambda$ nimmt, ändert die Vorhersage um den Faktor $n$. Der Faktor, der $\sigma$ aus der Median-Heuristik skaliert, heißt Skalenfaktor ($\sigma$ = Skalenfaktor mal Median der Abstände); er ist nicht die
Funktion $f$. Die Gram-Matrix des Gauß-Kerns ist positiv semidefinit, $K + \lambda I$ mit $\lambda > 0$ positiv definit. „Kriging“ ist der Name derselben Rechnung in der Geostatistik
(Rasmussen/Williams §2.8); die Varianten einfaches, ordinäres und universelles Kriging sind hier weder gemessen noch angeboten.

**Literatur.** Rasmussen, Williams: *Gaussian Processes for Machine Learning*, MIT Press 2006 (Gl. 2.22 bis 2.26, 2.30, 5.8, 5.10 bis 5.12, 6.19, 6.20; offen lesbar). Girosi, Jones, Poggio 1995
(Vorveröffentlichung, Gl. 7: dieselbe Dualform).
"""
