# Kernel Ridge und das GP-Mittel: dieselbe Gleichung, zwei Aussagen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kernel-ridge-demo.streamlit.app/)** -->

Teil der Reihe **Kernmethoden** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Demo steht zwischen den Demos zur Kernwahl und
zur Bandbreite und Kondition: sie hebt die Annahme der Marge-Demos auf, das Ziel sei eine Klassifikation mit Hinge-Verlust, deren Lösung dünn besetzt ist. Hier ist der Verlust quadratisch, die Lösung
ein **lineares Gleichungssystem** (K + λI) α = y mit lauter von null verschiedenen α, und dieselbe Rechnung gehört zu einem Modell mit **Verteilung** (Gauß-Prozess, GP). Neu ist nicht ein Gewinn gegenüber
der Klassifikation (ob der Wechsel von Hinge- zu quadratischem Verlust bei ±1-Labels die Klassifikation ändert, wird hier nicht gemessen), sondern die Gleichung, ihre Gleichheit mit dem GP-Mittel und das Band innerhalb der Daten.
Die Vehikel sind **Modelle**: eine ausgedachte glatte Zielfunktion über einem Plan („Plan-Regression“) und ein Erzeugermodell für Container-Verweilzeiten.

## Warum dieses Problem

Kernel Ridge und das Mittel eines GP sind **dieselbe Gleichung**, aber nur das Mittel; die Varianz der Funktion gibt es nur beim GP. Vier Kernfragen: (1) Ist das wirklich dieselbe Rechnung, und was melden die
Rechenwege? (2) Was gibt es beim GP zusätzlich, und stimmt es? (3) Wie wählt man σ und λ, ohne n-mal zu rechnen? (4) Wann bringt der Kern gegenüber einem linearen Modell etwas?
Der Hook der Reihe heißt „der Löser meldet nichts“; hier zeigt er sich in seiner **schwachen Form**: sechs Rechenwege stimmen bei gut gestellter Matrix überein, bei σ = 4 und λ = 1e-10 meldet keiner der direkten Wege etwas,
und ihr Abstand untereinander liegt in der Größe ihres Fehlers gegen die exakte Lösung.

## Modell

**Daten** (x_i, y_i), x_i ∈ [-2, 2]², y_i = Zielfunktion + Rauschen (Standardabweichung 0,3, also Varianz 0,09); Zielfunktion sin(1,5 x₁) + 0,5 cos(2,5 x₂) + 0,3 x₁x₂ (glatt) oder |x₁| + 0,5 |x₂| + 0,3 sin(3 x₁) (rau);
Container: log-Verweilzeit aus Art, Kunde, Zollhalt, Wochenende, Vorlauf und einem Rauschmerkmal (20 Spalten, `krd_container.py`). **Gauß-Kern** k(x, x') = exp(−‖x − x'‖² / (2σ²)) mit **σ = Bandbreite (Längenmaß)**; γ = 1/(2σ²) wird nur genannt.
**Kernel Ridge:** α = (K + λI)⁻¹ y, f(x*) = k*ᵀα. **GP** mit Prior c·k, Rauschvarianz σ_n², Prior-Mittel 0: gleiches Mittel, wenn λ = σ_n²/c; die Varianz der Funktion c (k(x*, x*) − k*ᵀ (K + λI)⁻¹ k*) gibt es nur beim GP
(Rasmussen/Williams 2006, Gl. 2.23 bis 2.26, 6.19, 6.20). In `KernelRidge` von scikit-learn ist `alpha` unser λ; in der Fassung (1/n) Σ (y − f)² + μ‖f‖² ist λ = n μ. **Zentrierung ist eine Konvention:** Standard ist das Abziehen des Trainingsmittels
(Schalter in der Seitenleiste); ohne sie fällt das Mittel fern der Daten auf 0. Alle Formeln und die Namenstabelle stehen im 📐-Expander der App.

## Methodik

- **Vier Wege live:** Cholesky, LU (`numpy.linalg.solve`), Eigenzerlegung und numerische Minimierung (L-BFGS-B, höchstens 5 000 Iterationen) rechnen selbst, gegen die **exakte** Lösung des Gleitkomma-Systems (Bareiss mit ganzen Zahlen, bis n = 40). Zählung: **sechs Rechenwege** = fünf direkte (Cholesky, LU, Eigenzerlegung, `KernelRidge`, `GaussianProcessRegressor` aus scikit-learn; dieselbe Formel in verschiedenen Programmen) plus die Minimierung; die exakte Lösung ist Referenz, kein Weg. Die beiden scikit-learn-Wege laufen nur in der Messwerkstatt und in den Tests; kein Modul der App importiert scikit-learn (Test: `tests/test_runtime_imports.py`).
- **Leave-one-out geschlossen:** Residuum α_i / [(K + λI)⁻¹]_ii, Varianz 1 / [(K + λI)⁻¹]_ii, gegen echtes Herausnehmen (n Zerlegungen der Größe n − 1).
- **Wahl von (σ, λ):** Leave-one-out (PRESS), GCV, Evidenz auf einem Raster 7 × 6 (σ 0,1 bis 6,4 in Zweierschritten, λ von 1e-6 bis 10) und Evidenz per Gradientenverfahren (L-BFGS-B auf (log σ, log λ), analytischer Gradient, sechs Startpunkte, Grenzen wie der Rand des Rasters).
- **Zufall:** durchgängig SplitMix64 (`krd_rng.py`, byte-gleiche Kopie der Vorlage, geprüft); Normalzahlen als `NormalDist().inv_cdf(random())`. Kein `numpy.random`. Der Container-Erzeuger ist nach der Beschreibung der `verweilzeit-demo` selbst gebaut (gleiche Verteilung, keine Bitgleichheit).
- **Vorgerechnet** (JSON, Skripte `tools/krd_make_*.py` mit `--check`): Wege-Tabelle und Band über 20 Zufalls-Seeds, Wahlregeln, Abdeckung des Bandes, Lernkurve, Stützvektoranteil. **Live-Modell ≠ Studie:** die App rechnet vier Wege selbst; die vorgerechneten Zahlen enthalten zusätzlich die beiden scikit-learn-Wege, und die Minimierung ist live auf 5 000 Iterationen begrenzt (die Messwerkstatt rechnet ebenfalls mit 5 000).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für diese Modelle, Einstellungen und Zufalls-Seeds; die Zahl der Zufalls-Seeds steht neben der Zahl. Fehlerziffern der Rechenwege sind plattformabhängig (gemessen: Windows, Python 3.14) und stehen als Bänder.

**1. Gleichheit des Mittels.** Am Handbeispiel (drei Punkte, Polynomkern Grad 2, λ = 1) in Brüchen: α = (1/4, −1/4, 1/4), f(2) = 9/4, Varianz der Funktion 319/40 gegen Prior-Varianz 25; GP mit Prior-Varianz 4 und Rauschvarianz 0,36 gleich Kernel Ridge mit λ = 0,09 exakt. Im Gleitkomma (Plan-Regression, n = 30, σ = 1, λ = 1, Zufalls-Seed 14101) liegen die fünf direkten Wege höchstens 2,6e-15 neben der exakten Lösung, die Minimierung 6,2e-9.

**2. Wo die Wege auseinanderlaufen (Zufalls-Seed 14101, n = 30, 200 Testpunkte, Skala der Vorhersage (größter Betrag) etwa 1 bis 2).**

| λ | 1 | 1e-2 | 1e-4 | 1e-6 | 1e-8 | 1e-10 |
|---|---|---|---|---|---|---|
| σ = 4: cond(K + λI) | 2,6e1 | 2,5e3 | 2,5e5 | 2,5e7 | 2,5e9 | 2,5e11 |
| σ = 4: Cholesky | 1,1e-15 | 5,7e-14 | 5,8e-12 | 8,1e-10 | 1,3e-7 | 2,7e-5 |
| σ = 4: LU | 1,0e-15 | 8,2e-14 | 5,9e-12 | 9,3e-10 | 8,4e-8 | 2,4e-5 |
| σ = 4: Eigenzerlegung | 2,2e-15 | 1,4e-13 | 1,6e-11 | 2,7e-9 | 1,9e-7 | 1,1e-4 |
| σ = 4: Abstand der direkten Wege untereinander | 2,6e-15 | 1,6e-13 | 1,7e-11 | 2,8e-9 | 2,8e-7 | 1,1e-4 |
| σ = 4: Minimierung | 8,0e-7 | 1,0e-4 | 1,3e-2 | 5,2e-1 | 8,7e0 | 2,4e1 |

Bei σ = 4 und λ = 1e-10 liegen die Fehler der drei selbst gerechneten Wege zwischen 2,4e-5 und 1,1e-4 und in diesem Zufalls-Seed meldet keiner der fünf direkten Wege etwas. **Band über 20 Zufalls-Seeds** (n = 30, Zufalls-Seeds 16000 bis 16019): σ = 4, λ = 1e-10: Fehler der fünf direkten Wege 4,5e-6 bis 2,6e-3 (cond 2,4e11 bis 2,6e11), die Minimierung 6,3 bis 140 und meldet „ABNORMAL“ in 12 von 20 Läufen; σ = 4, λ = 1e-6: 2,6e-10 bis 3,8e-8; σ = 1, λ = 1e-10: 5,8e-13 bis 2,6e-8 (cond 3,4e4 bis 2,5e7); σ = 1, λ = 1e-6: 8,6e-13 bis 4,9e-9. **In allen 80 Läufen** (4 Einstellungen × 20 Zufalls-Seeds) gab es von den fünf direkten Wegen keine Meldung und keine Ausnahme. Das Verhältnis (Abstand der direkten Wege untereinander) / (größter Fehler eines direkten Weges) liegt über diese 80 Läufe zwischen 0,63 und 1,89: als billige Prüfung („zwei Wege vergleichen“) ist das eine Beobachtung an dieser Plattform und diesen Wegen, kein Satz. Dass die Zerlegung gelingt und die Lösung falsch ist, zeigt die Bandbreite-und-Kondition-Demo der Reihe.

**3. Leave-one-out geschlossen.** Gegen echtes Herausnehmen in 100 Zufallsinstanzen (n = 6 bis 30, σ 0,3 bis 3, λ 1e-3 bis 1): Residuum höchstens 7,1e-13 (relativ), Varianz höchstens 7,5e-15. Zählung: Zerlegungen 1 gegen n; Gleitkommaoperationen (mitgezählt, Multiplikationen und Additionen) bei n = 20: 6 500 gegen 59 280 (Faktor 9,1), bei n = 160: 2 807 200 gegen 222 413 440 (Faktor 79,2). Am Handbeispiel: e = (10/9, −2/5, 10/9), Varianzen (40/9, 8/5, 40/9), echtes Herausnehmen gibt dieselben Brüche.

**4. Wahl von (σ, λ)** (Plan-Regression, V1-Funktion, n = 60, Überschussfehler = mittlerer quadratischer Fehler gegen die rauschfreie Wahrheit auf 1 000 Testpunkten; drei Sätze von Zufalls-Seeds: Satz A 30, Sätze B und C je 40):

| Regel | Satz A: Mittel (über das Doppelte der Rückblick-Zelle) | Satz B | Satz C |
|---|---|---|---|
| Leave-one-out | 0,0557 (1 von 30) | 0,0646 (8 von 40) | 0,0584 (5 von 40) |
| GCV | 0,0836 (5 von 30) | 0,3689 (11 von 40) | 0,3801 (11 von 40) |
| Evidenz auf dem Raster | 0,0530 (0 von 30) | 0,0562 (2 von 40) | 0,0524 (1 von 40) |
| Evidenz per Gradient | 0,0491 (0 von 30) | 0,0552 (1 von 40) | 0,0505 (0 von 40) |
| Rückblick-Zelle (nicht ausführbar) | 0,0457 | 0,0451 | 0,0431 |
| lineare Ridge (λ per Leave-one-out) | 0,4796 | 0,4731 | 0,4765 |

Der Kern (σ und λ per Leave-one-out gewählt) schlägt die lineare Ridge in 30 von 30, 40 von 40 und 40 von 40 Zufalls-Seeds (bei GCV-Wahl in 30, 36 und 39; die Zielfunktion ist nichtlinear, das ist der Fall, in dem der Kern etwas bringt). Das Leave-one-out-Minimum unterschätzt den tatsächlichen Fehler in 13 von 30, 22 von 40 und 26 von 40 Zufalls-Seeds. Die Rangfolge der Mittel (Evidenz vor Leave-one-out vor GCV) gilt in allen drei Sätzen; zwischen den Sätzen streuen die Zahlen deutlich (GCV 0,0836 gegen 0,3801 durch einzelne Fehlgriffe). **Gradient gegen Raster:** gegen das grobe Raster 7 × 6 wählt das Gradientenverfahren im Mittel mit niedrigerem Fehler (0,0491 gegen 0,0530, 0,0552 gegen 0,0562, 0,0505 gegen 0,0524; Unterschiede 0,0039, 0,0011 und 0,0019, kleiner als die Streuung zwischen den Sätzen); gegen ein zweites Raster 8 × 8 (σ 0,3 bis 4, λ 1e-4 bis 3) ist es nicht besser: Raster minus Gradient +0,0006 ± 0,0008, −0,0013 ± 0,0006 und −0,0014 ± 0,0006, in Satz B und C liegt das Raster um etwa zwei Standardfehler vorn (Raster 8 × 8: 0,0497, 0,0539, 0,0491).

**5. Das 95-%-Band der Funktion** (Abdeckung = Anteil der Testpunkte, an denen die Wahrheit im Band liegt; nicht die der verrauschten Beobachtung). **Orakel:** stammt die Wahrheit aus dem Prior und sind die Hyperparameter bekannt (n = 30, 100 Testpunkte), ist das Band kalibriert: Abdeckung 95,4 % ± 0,3 % (300 Zufalls-Seeds, Soll 95 %), mittleres z² 1,002 ± 0,021 (Soll 1). **Hyperparameter aus den Daten** (drei Sätze zu je 30 Zufalls-Seeds, kleinster und größter Satz; in Klammern die Zahl der Zufalls-Seeds je Satz unter 90 % Abdeckung):

| Wahrheit, n | Evidenz (Raster) | Evidenz (Gradient) | Leave-one-out |
|---|---|---|---|
| V1-Funktion, n = 20 | 88,9 % bis 89,5 % (13 bis 14) | 87,5 % bis 89,7 % (11 bis 19) | 84,0 % bis 88,1 % (12 bis 17) |
| V1-Funktion, n = 60 | 93,8 % bis 96,1 % (2 bis 4) | 94,2 % bis 95,2 % (3 bis 7) | 92,0 % bis 92,9 % (9 bis 10) |
| V1-Funktion, n = 200 | 94,1 % bis 95,5 % (3 bis 6) | 94,2 % bis 95,4 % (2 bis 6) | 91,3 % bis 93,6 % (6 bis 11) |
| raue Funktion, n = 60 | 87,4 % bis 88,6 % (13 bis 17) | 90,1 % bis 91,9 % (7 bis 12) | 82,6 % bis 86,8 % (18 bis 21) |
| raue Funktion, n = 200 | 89,6 % bis 92,5 % (5 bis 6) | 89,8 % bis 92,2 % (7 bis 15) | 90,0 % bis 91,0 % (9 bis 11) |
| aus dem Prior, n = 60 | 90,3 % bis 94,2 % (6 bis 13) | 90,6 % bis 94,3 % (3 bis 12) | 88,3 % bis 92,2 % (8 bis 12) |

Lesart: bei Evidenz-Wahl und n ≥ 60 deckt das Band an der glatten Funktion im Mittel 93,8 % bis 96,1 % ab, mit 2 bis 7 von 30 Zufalls-Seeds unter 90 %; zu eng ist es bei n = 20 (84,0 % bis 89,7 %), bei Wahl durch Leave-one-out und an der rauen Funktion mit Evidenz auf dem Raster (n = 60: 87,4 % bis 88,6 %). Die Aussage „brauchbar“ (Mittel der Abdeckung mindestens 93 %) gilt in den Zellen dieser Tabelle nur für Evidenz-Wahl, die glatte Zielfunktion und n ≥ 60, und auch dort nur im Mittel: einzelne Zufalls-Seeds liegen unter 90 %. Das Band selbst hat keinen Warnwert: es zeigt nicht an, ob der Prior passt; die Abdeckung ist nur messbar, wenn die Wahrheit bekannt ist. Es gilt **innerhalb** der Daten; außerhalb der Daten (feste Hyperparameter, Rückfall auf den Prior) ist ein anderer Ort, Gegenstand der Kernwahl-Demo der Reihe.

**6. Container-Erzeuger: der Kern bringt wenig** (20 Zufalls-Seeds gepaart, 4 000 Testcontainer je Zufalls-Seed, RMSE der log-Verweilzeit in log-Tagen; Skalenfaktor und λ per geschlossenem Leave-one-out):

| n | RMSE Kernel Ridge | RMSE lineare Ridge | Kern besser in | Differenz (linear − Kern) ± Standardfehler |
|---|---|---|---|---|
| 100 | 0,5924 | 0,5759 | 7 von 20 | −0,0165 ± 0,0066 |
| 300 | 0,5342 | 0,5345 | 11 von 20 | +0,0003 ± 0,0016 |
| 1 000 | 0,5118 | 0,5191 | 19 von 20 | +0,0073 ± 0,0009 |

Der Boden des RMSE ist exakt 0,4861 (Wurzel des Mittels der bedingten Varianz, Orakel des Erzeugers; Stichprobenmittel der 20 Zufalls-Seeds 0,4852); bei n = 1 000 holt der Kern 1,4 % des RMSE der linearen Ridge (0,0073 von 0,0330 bis zum Boden 0,4861); der Boden selbst, 93,6 % des RMSE der linearen Ridge, ist die unerklärte Streuung des Erzeugers, die auch ein anderes Verfahren nicht unterschreitet. Der konstante Mittelwert liegt bei 0,74. Das gilt für Ridge-Regression auf diesem Erzeuger, nicht für Klassifikation und nicht für andere Lerner; in den gelesenen Studien liegen für Verweilzeit und Nachfrage Bäume vorn.

**7. Keine Stützvektoren.** Kernel Ridge auf ±1-Labels (Kreis mit halber Fläche, 10 % vertauschte Labels, n = 100, σ = 0,4, λ = 0,3, 50 Instanzen): 0 Nullen in 5 000 Koeffizienten (Rauschgrenze 1e-12 des größten Betrags). `SVC` mit demselben Kern und C = 1: Stützvektoranteil im Mittel 60,8 % (48 % bis 69 %). Testfehler gegen den echten Kreis 0,0980 (Kernel Ridge) und 0,1082 (`SVC`) bei festen, nicht abgestimmten Einstellungen: kein Verfahrensvergleich. „Alle α ungleich null“ gilt in allgemeiner Lage, nicht für jedes y: für y = 0 ist α = 0.

**Neumessung gegenüber früher mit numpy gezogenen Daten** (Zufallsstrom und Kundeneffekte sind andere): die Wege-Spanne bei σ = 4, λ = 1e-10 war 5,6e-6 bis 3,3e-3, jetzt 4,5e-6 bis 2,6e-3; die Abdeckung bei Evidenz-Wahl und n = 60 war 93 % bis 95 %, jetzt 93,8 % bis 96,1 %; GCV „gescheitert“ in 7 von 30, jetzt 5 von 30 (Satz A); das Leave-one-out-Minimum unterschätzte den Fehler in 20 von 30, jetzt in 13 von 30 (Satz A), 22 von 40 und 26 von 40. **Container:** früher lag der Kern bei n = 100 in 11 von 20 vorn, bei n = 300 in 16 von 20 und bei n = 1 000 in 20 von 20 (2,6 % des RMSE); jetzt 7, 11 und 19 von 20 (1,4 %). Die zwölf Kundeneffekte dieser Ziehung streuen mit 0,25 statt 0,36 (Soll 0,30), der konstante Mittelwert liegt bei 0,74 statt 0,76; eine Probe mit den früheren Kundeneffekten gab bei n = 300 (6 Zufalls-Seeds) Kern 0,545 gegen linear 0,547 (Messung dieser README, nicht im Test). Die Aussage ändert sich nicht: bei n = 100 liegt die lineare Ridge vorn, bei n = 300 gleichauf, bei n = 1 000 der Kern knapp.

### Zahl → Test

| Zahl oder Aussage | Test |
|---|---|
| Handbeispiel (Brüche, Leave-one-out, Spur, GCV, log p) | `tests/test_handbeispiel_krd.py` (exakt) |
| Tabelle zu σ = 4 und Band über 20 Zufalls-Seeds (Befund 2) | `tests/test_claims.py::test_wege_tabelle_und_band`; Neurechnung einer Zelle: `tools/krd_make_wege.py --check` (Band, Plattform) |
| Leave-one-out gegen Herausnehmen, Operationszählung (Befund 3) | `tests/test_claims.py::test_loo_und_zaehlung`, `tests/test_oracle_kernel_ridge.py` |
| Wahlregeln, drei Sätze (Befund 4) | `tests/test_claims.py::test_wahlregeln_drei_saetze`, `tools/krd_make_flaeche.py --check` |
| Abdeckung des Bandes (Befund 5) | `tests/test_claims.py::test_band_abdeckung`, `tools/krd_make_band.py --check`; Orakel: `tests/test_oracle_kernel_ridge.py` (100 Zufalls-Seeds, Band mit Standardfehler) |
| Container, Boden (Befund 6) | `tests/test_claims.py::test_container_lernkurve`, `tests/test_rng_scenario_container.py` |
| Stützvektoren (Befund 7) | `tests/test_claims.py::test_keine_stuetzvektoren`, `tools/krd_make_stuetz.py --check` |
| Neumessung gegenüber früher, Probe mit früheren Kundeneffekten, Testfehler 0,0980 und 0,1082 als Anzeige | Messung dieser README, nicht im Test (die Zahlen 0,0980 und 0,1082 stehen in `krd_stuetz.json`, der Test prüft sie) |

## Ehrliche Grenzen

- **Nur das Mittel ist gleich.** Die Varianz gibt es nur beim GP; ob das Band stimmt, hängt an der Wahl von σ und λ, an n und an der Wahrheit, nicht an der Gleichung. Bei sehr kleinem n (20) und bei Wahl durch Leave-one-out ist es enger als versprochen, und auch bei Evidenz-Wahl liegt die Abdeckung je nach Zelle in 2 bis 19 von 30 Zufalls-Seeds unter 90 % (an der glatten Funktion mit n ≥ 60: 2 bis 7).
- **Stille der Löser ist hier keine Richtigkeit** (bei σ = 4 und λ = 1e-10 liegt in 20 von 20 Zufalls-Seeds der größte Fehler eines direkten Weges mindestens 1,6e-5, und keiner meldet etwas); der Abstand der Wege untereinander zeigt ihn hier jedoch an (0,63 bis 1,89 des größten Fehlers). Die Abweichungen sind plattformabhängig; ob der Abstand der Wege untereinander für andere Matrizen (andere Kerne, andere Punktlagen) den Fehler anzeigt, ist nicht gemessen.
- **Nur Gauß-Kern, nur zwei Dimensionen** (Container: 20 Spalten), Raster 7 × 6 (und 8 × 8 als zweites Raster), ausgedachte Zielfunktionen; Leave-one-out und GCV per Gradient sowie mehrere Bandbreiten je Merkmal sind nicht gemessen. Die Bareiss-Referenz läuft bis n = 40.
- **Kein Betriebsbezug.** Belegt ist der Einsatz von Kriging als Ersatzmodell einer Simulation; für Verweilzeit und Nachfrage liegen in den gelesenen Studien Bäume vorn. Was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet. Das Live-Modell der App (vier Wege selbst gerechnet) ist nicht das der vorgerechneten Messungen (zusätzlich scikit-learn, 20 bis 40 Zufalls-Seeds); die Abweichung ist die Tabelle oben.
- Ein Linux-Lauf (`demo_linux_check.py`) steht aus.

## Verifikation

- **Handbeispiel von Hand** in Brüchen (K, Determinanten, α, Vorhersage, Varianz, Leave-one-out, Spur, PRESS, GCV, log p, ±1-Labels) gegen den Gewichtsraum ohne Gram-Matrix, numerische Minimierung, scikit-learn (Fremdreferenz, dieselbe Gleichung) und Gauß-Hermite-Integration (`tests/test_handbeispiel_krd.py`).
- **Orakel** (`tests/test_oracle_kernel_ridge.py`): Gewichtsraum gegen Dualform (Polynomkern Grad 3), Minimierung bei gut gestellter Matrix (Band 1e-5), echtes Herausnehmen (20 Instanzen), Evidenz gegen scipy-Dichte und Gauß-Hermite, Spur H, Bareiss gegen Brüche, scikit-learn, Kalibrierung (Wahrheit aus dem Prior, Band mit Standardfehler), Gradient gegen Differenzenquotient.
- **Eigenschaften** (`tests/test_properties.py`, je 20 Zufallsinstanzen), **Randfälle** (`tests/test_edge_cases.py`), **Verdikte** als reine Funktionen (`tests/test_verdicts.py`, je Preset die erwartete Zeile), **Zufall und Erzeuger** (Kopier-Test, eingefrorene Werte, Eigenschaften als Band), **Layout, Presets, Footer, Laufzeit-Importe, AppTest-Rauchtest**.
- Testzahl: 282 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: vier Experimente, Handbeispiel, Annahmen, Formeln |
| `krd_model.py` | Gram-Matrix, vier Rechenwege, Posterior, Leave-one-out, Wahlregeln, Evidenz mit Gradient |
| `krd_exact.py` | Bareiss-Referenz und Handbeispiel in Brüchen |
| `krd_scenario.py`, `krd_container.py`, `krd_rng.py` | Plan-Erzeuger, Container-Erzeuger, SplitMix64 (Kopie der Vorlage) |
| `krd_evaluation.py` | Läufe der Experimente, Zusammenfassungen, Verdikte |
| `krd_constants.py`, `krd_presets.py`, `krd_layout.py`, `krd_visualization.py`, `krd_texts.py` | Konstanten, Presets und Permalink, fester Kopf, Abbildungen, Texte |
| `krd_wege.json`, `krd_flaeche.json`, `krd_band.json`, `krd_lernkurve.json`, `krd_stuetz.json` | vorgerechnete Messungen |
| `tools/krd_make_*.py` | Messwerkstatt (schreibt die JSON-Dateien, `--check` rechnet eine Zelle neu) |
| `tests/` | siehe Verifikation |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml`, `.streamlit/config.toml` | Laufzeit (streamlit, numpy, scipy, plotly), scikit-learn nur für Tests und Messwerkstatt, CI |

## Nicht angeboten / nicht gemessen

Kriging (Name aus der Geostatistik, Rasmussen/Williams §2.8; die Varianten einfaches, ordinäres und universelles Kriging werden hier nur als Name genannt), konforme Bänder um Kernel Ridge, Stützvektor-Regression, andere Kerne und der Rückfall auf den Prior außerhalb der Daten (Kernwahl-Demo der Reihe),
Container mit n ≥ 3 000, Leave-one-out per Gradient, Nutzen im Betrieb. TabPFN und neuronale Netze kommen nicht vor.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Messwerkstatt (schreibt die JSON-Dateien neu, mehrere Minuten): `python tools/krd_make_wege.py`, `krd_make_flaeche.py`, `krd_make_band.py`, `krd_make_lernkurve.py`, `krd_make_stuetz.py`; mit `--check` wird nur eine Zelle neu gerechnet und verglichen.

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html).
