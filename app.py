"""Kernel Ridge und das GP-Mittel: dieselbe Gleichung, zwei Aussagen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vier Experimente an ausgedachten Modellen (Plan-Regression, Container-Erzeuger): Rechenwege gegen die exakte Lösung, 95-%-Band des GP, Leave-one-out und Wahl von (σ, λ),
Kern gegen linear. Aufbau nach DEMO-PLAYBOOK §2 (Presets in der Seitenleiste, fester Kopf, Kernfragen, Annahmen, Formeln).

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import numpy as np
import streamlit as st

import krd_constants as C
import krd_container as CT
import krd_evaluation as EV
import krd_exact as E
import krd_texts as T
import krd_visualization as V
from krd_layout import dunkles_design, kopf_css
from krd_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Kernel Ridge und das GP-Mittel – Sebastian Hanisch", layout="wide")


# ---- zwischengespeicherte Läufe (reine Funktionen aus krd_evaluation) -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _wege(n, sigma, lam, seed, target, ways, center):
    return EV.wege_run(n, sigma, lam, seed, target, ways, center)


@st.cache_data(show_spinner=False)
def _wege_band(n, sigma, lam, seed, n_seeds, target, ways, center):
    return EV.wege_band(n, sigma, lam, seed, n_seeds, target, ways, center)


@st.cache_data(show_spinner=False)
def _band(truth, n, seed, rule, center):
    return EV.band_run(truth, n, seed, rule, center)


@st.cache_data(show_spinner=False)
def _far(truth, n, seed, rule, center):
    r = EV.band_run(truth, n, seed, rule, center, with_slice=False)
    return EV.far_means(r["X"], r["y"], r["sigma"], r["lam"])


@st.cache_data(show_spinner=False)
def _loo(n, sigma, lam, seed, center):
    return EV.loo_check(n, sigma, lam, seed, "v1", center)


@st.cache_data(show_spinner=False)
def _flaeche(n, seed, center):
    return EV.flaeche_run(n, seed, "v1", center)


@st.cache_data(show_spinner=False)
def _container(n, seed):
    return EV.container_run(n, seed)


@st.cache_data(show_spinner=False)
def _plan(n, seed):
    return EV.plan_run(n, seed)


@st.cache_data(show_spinner=False)
def _ridge_pm1(n, seed):
    r = EV.ridge_pm1(n, seed)
    return {"zeros": r["zeros"]}


# ---- fester Kopf, Einleitung -------------------------------------------------------------------------------------------------------------------
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 " + T.TITLE)
st.markdown(T.INTRO)

with st.expander("So funktioniert Kernel Ridge und das GP-Mittel", expanded=True):
    st.markdown(T.HOW)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# ---- Seitenleiste ---------------------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Modell (alle Experimente)**")
    center = st.checkbox("Trainingsmittel abziehen (zentrieren)", key="center", help="Standard: das Mittel der Beobachtungen wird vor der Rechnung abgezogen und danach addiert. Ohne Zentrierung gilt "
                                                                                      "das Prior-Mittel 0: fern der Daten fällt die Vorhersage auf 0 statt auf das Trainingsmittel.")
    st.markdown("**Gram-Matrix (Experimente 1 und 3)**")
    sigma = st.slider("Bandbreite σ", *bounds("sigma"), key="sigma", step=0.1, help="Längenmaß des Gauß-Kerns; γ = 1/(2σ²) wird nur genannt.")
    lam_exp = st.slider("λ = 10^…", *bounds("lam_exp"), key="lam_exp", help="Regularisierung von Kernel Ridge (beim GP: Rauschvarianz durch Prior-Varianz), logarithmisch von 1 bis 1e-10.")
    st.markdown("**Experiment 1: Rechenwege**")
    n_w = st.slider("Punkte n", *bounds("n_w"), key="n_w", help="Trainingspunkte der Plan-Regression; die exakte Referenz (Bareiss) läuft bis n = 40, hier bis n = 30.")
    seed_w = st.number_input("Zufalls-Seed", *bounds("seed_w"), key="seed_w", step=1, help="Zufalls-Seed der Daten; bei mehreren Zufalls-Seeds die folgenden.")
    st.button("🎲 Nächsten Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)
    n_seeds_w = st.slider("Anzahl Zufalls-Seeds", *bounds("n_seeds_w"), key="n_seeds_w", help="1: ein Zufalls-Seed mit allen Zahlen; mehr: Band über aufeinanderfolgende Zufalls-Seeds.")
    target_w = st.selectbox("Zielfunktion", C.TARGET_OPTIONS, key="target_w", help="Glatte oder raue ausgedachte Zielfunktion über dem Plan.")
    st.caption("Wege (Häkchen):")
    way_flags = {w: st.checkbox(C.WAY_LABELS[w], key=f"way_{w}") for w in C.WAY_KEYS}
    st.markdown("**Handbeispiel (Experiment 1)**")
    hand_x = st.select_slider("Testpunkt x*", C.HAND_X_OPTIONS, key="hand_x", help="Drei Punkte x = -1, 0, 1; der Testpunkt liegt innerhalb (-1 bis 1) oder außerhalb der Daten.")
    st.markdown("**Experiment 2: Band**")
    truth = st.selectbox("Wahrheit", C.TRUTH_OPTIONS, key="truth", help="Woher die Funktion stammt, deren Werte das Band einschließen soll.")
    n_band = st.select_slider("Trainingspunkte n", C.N_BAND_OPTIONS, key="n_band", help="Die vorgerechnete Verteilung über 30 Zufalls-Seeds liegt für n = 20, 60 und 200 vor.")
    rule = st.selectbox("Wahl von σ und λ", C.RULE_OPTIONS, key="rule", help="Regel, mit der σ und λ aus den Daten gewählt werden (oder feste Zelle).")
    seed_b = st.number_input("Zufalls-Seed der Einzelansicht", *bounds("seed_b"), key="seed_b", step=1)
    st.markdown("**Experiment 3: Leave-one-out und Wahl**")
    n_l = st.slider("Punkte n (Leave-one-out)", *bounds("n_l"), key="n_l", help="Trainingspunkte der Plan-Regression (V1-Funktion).")
    rule_l = st.selectbox("Markierte Regel", C.RULE3_OPTIONS, key="rule_l", help="Welche Wahl in der Fläche markiert wird.")
    seed_l = st.number_input("Zufalls-Seed der Fläche", *bounds("seed_l"), key="seed_l", step=1)
    st.markdown("**Experiment 4: Kern gegen linear**")
    vehicle = st.selectbox("Erzeuger", C.VEHICLE_OPTIONS, key="vehicle", help="Plan-Regression (nichtlineare Zielfunktion) oder Container-Erzeuger (Gegenbefund).")
    n_c = st.slider("Trainingspunkte n (Experiment 4)", *bounds("n_c"), key="n_c", step=50, help="Live bis 300; n = 1 000 ist nur vorgerechnet.")
    seed_c = st.number_input("Zufalls-Seed der Ziehung", *bounds("seed_c"), key="seed_c", step=1)

sync_query_params({k: st.session_state.get(k, spec.default) for k, spec in SETTING_SPECS.items()})

lam = 10.0 ** int(lam_exp)
ways = tuple(w for w in C.WAY_KEYS if way_flags[w])
target_key = C.TRUTH_KEYS[target_w]

# ================================================================ Experiment 1
st.subheader("🔬 Experiment 1: Dieselbe Gleichung, sechs Rechenwege gegen die exakte Lösung")
st.markdown(f"**{T.Q1}**")
st.caption(T.CAP_WEGE)
st.caption(T.CAP_LIVE_VS_STUDY)

if n_seeds_w == 1:
    res = _wege(int(n_w), float(sigma), lam, int(seed_w), target_key, ways, bool(center))
    errs = {w: res["ways"][w]["err"] for w in ways}
    notes = {w: res["ways"][w]["note"] for w in ways}
    c1, c2, c3 = st.columns(3)
    c1.metric("cond(K + λI)", EV.fmt(res["cond"]))
    c2.metric("Abstand der direkten Wege", "–" if res["spread"] is None else EV.fmt(res["spread"]))
    c3.metric("Meldungen der direkten Wege", f"{sum(1 for w in ways if w != 'min' and notes[w])} von {sum(1 for w in ways if w != 'min')}")
    if ways:
        st.plotly_chart(V.build_wege(errs, C.WAY_LABELS, notes), width="stretch", key="wege_chart")
    kind, text = EV.verdict_wege(res["cond"], errs, notes, res["spread"], C.WAY_LABELS)
else:
    with st.spinner(f"Rechne {int(n_seeds_w)} Zufalls-Seeds (Zählung, keine Zeitangabe) …"):
        band = _wege_band(int(n_w), float(sigma), lam, int(seed_w), int(n_seeds_w), target_key, ways, bool(center))
    rows = [{"Weg": C.WAY_LABELS[w], "kleinster Fehler": EV.fmt(b[0]), "Median": EV.fmt(b[1]), "größter Fehler": EV.fmt(b[2])} for w, b in band["err"].items() if b is not None]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.plotly_chart(V.build_wege({w: b[2] for w, b in band["err"].items() if b is not None}, C.WAY_LABELS), width="stretch", key="wege_chart")
    kind, text = EV.verdict_wege_band(band)
getattr(st, kind)(text)
st.caption("**Hook, schwache Form:** der Löser meldet nichts. Nur die Minimierung meldet („ABNORMAL“), und gerade sie ist die ungenaueste. Dass die Zerlegung gelingt und die Lösung falsch ist, zeigt die Bandbreite-und-Kondition-Demo der Reihe.")

with st.expander("📊 Vorgerechnet: Fehler über λ und Band über 20 Zufalls-Seeds (mit scikit-learn)"):
    wf = EV.load_data("wege")
    st.markdown(f"Ein Zufalls-Seed ({wf['table']['seed']}), n = {wf['table']['n']}: Fehler je Weg über λ. Zusätzlich die beiden Wege aus scikit-learn (`KernelRidge`, `GaussianProcessRegressor`; dieselbe Formel in anderen Programmen, kein unabhängiger Beweis).")
    for sg_key in ("1", "4"):
        st.plotly_chart(V.build_lambda_table(wf["table"]["sigmas"][sg_key], wf["table"]["lams"], f"σ = {sg_key}"), width="stretch", key=f"lam_chart_{sg_key}")
    brow = []
    for key in wf["band"]["settings"]:
        fz = EV.wege_frozen(key)
        s = wf["band"]["settings"][key]
        brow.append({"σ": f"{s['sigma']:g}", "λ": f"{s['lam']:g}", "cond (kleinster bis größter)": f"{EV.fmt(fz['cond'][0])} bis {EV.fmt(fz['cond'][2])}",
                     "fünf direkte Wege: Fehler": f"{EV.fmt(fz['direct_range'][0])} bis {EV.fmt(fz['direct_range'][1])}", "Läufe ohne Meldung": f"{fz['silent_runs']} von {fz['n_seeds']}",
                     "Minimierung meldet": f"{fz['min_notes']} von {fz['n_seeds']}", "Abstand / größter Fehler": f"{fz['ratio'][0]:.2f} bis {fz['ratio'][2]:.2f}"})
    st.dataframe(brow, hide_index=True, width="stretch")
    st.caption(f"Band über {wf['band']['n_seeds']} Zufalls-Seeds ({wf['band']['seed0']} + s), n = {wf['band']['n']}, 200 Testpunkte, Fehler gegen die exakte Lösung.")

st.markdown("#### Von Hand: drei Punkte in Brüchen")
st.caption("Drei Punkte x = (-1, 0, 1) mit y = (1, 0, 1), Polynomkern vom Grad 2, k(x, x') = (x x' + 1)² (ganzzahlige Gram-Matrix, deshalb exakt in Brüchen; der Gauß-Kern wäre nicht rational), λ = 1.")
h = E.hand_example(Fraction(1), Fraction(str(hand_x)))
ch1, ch2 = st.columns([1, 1])
with ch1:
    st.markdown(
        f"- **K** = {[[str(v) for v in row] for row in h['K']]}, det K = {h['det_K']}, det(K + I) = {h['det_A']}\n"
        f"- **α** = (K + I)⁻¹ y = ({', '.join(str(a) for a in h['alpha'])}); alle α ungleich null\n"
        f"- **f(x\\*)** = {h['f_star']}, Varianz der Funktion = {h['var_star']} (Prior-Varianz {h['prior_var']})\n"
        f"- **Leave-one-out** geschlossen: e = ({', '.join(str(a) for a in h['loo_e'])}), Varianzen ({', '.join(str(a) for a in h['loo_var'])}); echtes Herausnehmen gibt "
        f"{'dieselben Brüche' if h['loo_e'] == h['loo_e_real'] and h['loo_var'] == h['loo_var_real'] else 'andere Werte'}\n"
        f"- Spur H = {h['trace']}, PRESS = {h['press']} ({float(h['press']):.4f}), GCV = {h['gcv']} ({float(h['gcv']):.4f})\n"
        f"- log p(y) = {h['log_p']:.4f}, profilierter Skalenfaktor = {h['s2_hat']}".replace(".", ",")
    )
with ch2:
    st.plotly_chart(V.build_hand(E.HAND_X, E.HAND_Y, float(hand_x)), width="stretch", key="hand_chart")
kind, text = EV.verdict_hand(h, float(hand_x))
getattr(st, kind)(text)

st.markdown("#### Keine Stützvektoren")
rp = _ridge_pm1(100, int(seed_w))
zf = EV.load_data("stuetz")
per = zf["per_instance"]
kind, text = EV.verdict_zero(rp["zeros"], 100, sum(p["sv_share"] for p in per) / len(per), min(p["sv_share"] for p in per), max(p["sv_share"] for p in per),
                             sum(p["zeros"] for p in per), len(per) * zf["n"], len(per))
getattr(st, kind)(text)
st.caption("Kreis mit halber Fläche in [-1, 1]², 10 % vertauschte Labels, Gauß-Kern σ = 0,4, Kernel Ridge λ = 0,3, SVC C = 1 (feste Einstellungen, kein Verfahrensvergleich).")

st.markdown("---")

# ================================================================ Experiment 2
st.subheader("🔬 Experiment 2: Das Band, was beim GP dazukommt, und ob es stimmt")
st.markdown(f"**{T.Q2}**")
st.caption(T.CAP_BAND)
truth_key, rule_key = C.TRUTH_KEYS[truth], C.RULE_KEYS[rule]
run = _band(truth_key, int(n_band), int(seed_b), rule_key, bool(center))
b1, b2, b3, b4 = st.columns(4)
b1.metric("Abdeckung dieser Ziehung", EV.pct(run["coverage"]))
b2.metric("gewähltes σ", f"{run['sigma']:.2f}".replace(".", ","))
b3.metric("gewähltes λ", EV.fmt(run["lam"]))
b4.metric("geschätzte Rauschvarianz λ·ŝ²", f"{run['noise_var']:.3f}".replace(".", ","), help="Wahre Rauschvarianz: 0,09.")
st.plotly_chart(V.build_band(run, f"{truth}, n = {int(n_band)}, Wahl: {rule}"), width="stretch", key="band_chart")
cell = EV.band_cell(truth_key, int(n_band), rule_key)
oc = EV.load_data("band")["oracle"]

oc_cov = np.asarray(oc["coverage"])
oracle = {"mean": float(oc_cov.mean()), "se": float(oc_cov.std(ddof=1) / len(oc_cov) ** 0.5), "n": len(oc_cov)}
kind, text = EV.verdict_band(run["coverage"], cell, oracle, rule, truth, int(n_band))
getattr(st, kind)(text)

trs = []
for rk_label, rk in C.RULE_KEYS.items():
    cc = EV.band_cell(truth_key, int(n_band), rk)
    trs.append({"Wahl": rk_label, "Abdeckung im Mittel (kleinster bis größter Satz)": f"{EV.pct(cc['mean_band'][0])} bis {EV.pct(cc['mean_band'][1])}",
                "Zufalls-Seeds unter 90 % (je Satz von 30)": f"{cc['below90_band'][0]} bis {cc['below90_band'][1]}",
                "z² Median (kleinster bis größter Satz)": f"{cc['z2_median_band'][0]:.2f} bis {cc['z2_median_band'][1]:.2f}".replace(".", ",")})
st.dataframe(trs, hide_index=True, width="stretch")
st.caption(f"Vorgerechnet: {len(EV.load_data('band')['sets'])} Sätze zu je {EV.load_data('band')['n_seeds']} Zufalls-Seeds für Wahrheit „{truth}“ und n = {int(n_band)}; z² = mittleres Quadrat der standardisierten Abweichung (Soll 1).")
fm = _far(truth_key, int(n_band), int(seed_b), rule_key, bool(center))
st.caption(f"Zentrierung als Konvention: am Punkt (6, 6) weit außerhalb der Daten gibt das Prior-Mittel 0 den Wert {fm['prior_zero']:.3f}, das zentrierte Modell {fm['centered']:.3f} "
           f"(Trainingsmittel {fm['train_mean']:.3f}).".replace(".", ","))

st.markdown("---")

# ================================================================ Experiment 3
st.subheader("🔬 Experiment 3: Leave-one-out ohne n-mal Rechnen, und was es wählt")
st.markdown(f"**{T.Q3}**")
st.caption(T.CAP_LOO)
chk = _loo(int(n_l), float(sigma), lam, int(seed_l), bool(center))
kind, text = EV.verdict_loo(chk)
getattr(st, kind)(text)
fl = _flaeche(int(n_l), int(seed_l), bool(center))
rk3 = C.RULE3_KEYS[rule_l]
picks_view = {rk3: fl["picks"][rk3]}
st.plotly_chart(V.build_flaeche(fl["table"]["test"], picks_view, fl["best"], {v: k for k, v in C.RULE3_KEYS.items()}), width="stretch", key="flaeche_chart")
rows3 = []
labels3 = {v: k for k, v in C.RULE3_KEYS.items()}
for rk, lb in labels3.items():
    sg_, lm_ = EV.pick_values(fl["picks"][rk])
    rows3.append({"Regel": lb, "gewähltes σ": f"{sg_:.2f}".replace(".", ","), "gewähltes λ": EV.fmt(lm_), "Überschussfehler": f"{fl['tests'][rk]:.4f}".replace(".", ","),
                  "Verhältnis zur Rückblick-Zelle": f"{fl['tests'][rk] / fl['best_test']:.2f}".replace(".", ",")})
st.dataframe(rows3, hide_index=True, width="stretch")
st.caption(f"Rückblick-Zelle (beste Zelle am Test, im Einsatz nicht ausführbar): Überschussfehler {fl['best_test']:.4f}.".replace(".", ","))
sets_rows = []
for sname in ("A", "B", "C"):
    sm = EV.flaeche_summary(sname)
    for rk, lb in labels3.items():
        sets_rows.append({"Satz": f"{sname} ({sm['n']} Zufalls-Seeds)", "Regel": lb, "Überschussfehler im Mittel": f"{sm['rules'][rk]['mean']:.4f}".replace(".", ","),
                          "über das Doppelte der Rückblick-Zelle": f"{sm['rules'][rk]['above_2']} von {sm['n']}"})
with st.expander("📊 Vorgerechnet: Wahlregeln über drei Sätze von Zufalls-Seeds (n = 60)"):
    st.dataframe(sets_rows, hide_index=True, width="stretch")
kind, text = EV.verdict_rules(EV.flaeche_summary("A"), "Satz A")
getattr(st, kind)(text)

st.markdown("---")

# ================================================================ Experiment 4
st.subheader("🔬 Experiment 4: Wann bringt der Kern gegenüber linear etwas?")
st.markdown(f"**{T.Q4}**")
st.caption(T.CAP_KERN)
if vehicle == C.VEHICLE_OPTIONS[1]:
    lv = dict(_container(int(n_c), int(seed_c)), n=int(n_c))
else:
    lv = dict(_plan(int(n_c), int(seed_c)), n=int(n_c))
lk_rows = []
for nn in EV.load_data("lernkurve")["ns"]:
    sm = EV.lernkurve_summary(nn)
    lk_rows.append(dict(sm, n=nn))
st.plotly_chart(V.build_lernkurve(lk_rows, lv if vehicle == C.VEHICLE_OPTIONS[1] else None, CT.floor_exact()), width="stretch", key="lern_chart")
kind, text = EV.verdict_kern(vehicle, lv, lk_rows, EV.flaeche_summary("A"), CT.floor_exact())
getattr(st, kind)(text)

st.markdown("---")

# ================================================================ Grenzen und Formeln
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS)
with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(T.LIMITS)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.LESEHILFE)
    st.markdown(T.MATH_A)
    st.latex(T.MATH_KERNEL)
    st.markdown(T.MATH_B)
    st.latex(T.MATH_ALPHA)
    st.markdown(T.MATH_C)
    st.latex(T.MATH_GPMEAN)
    st.markdown(T.MATH_D)
    st.latex(T.MATH_VAR)
    st.markdown(T.MATH_E)
    st.latex(T.MATH_LOO)
    st.markdown(T.MATH_F)
    st.latex(T.MATH_GCV)
    st.latex(T.MATH_EVID)
    st.markdown(T.MATH_G)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html)."
)
