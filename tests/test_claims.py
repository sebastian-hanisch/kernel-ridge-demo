"""Jede Zahl der README steht hier: die Tests rechnen die Zeichenfolge aus den vorgerechneten Messungen (krd_*.json, Skripte tools/krd_make_*.py) bzw. aus dem Code neu und
suchen sie in der README. Weicht die README von der Messung ab (oder die Messung nach einer Neumessung von der README), schlägt der Test fehl (DEMO-PLAYBOOK §11a Punkt 13).

Fehlerziffern der Rechenwege sind plattformabhängig: der Test liest die EINGEFRORENEN Werte und vergleicht sie mit der README; die Neurechnung einer Zelle als Band macht
`tools/krd_make_wege.py --check` (siehe tests/test_data_and_tools.py). Messungen ohne Test sind in der README ausdrücklich so benannt."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import krd_container as CT
import krd_evaluation as EV

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def sci(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Komma, eine Nachkommastelle, Exponent ohne führende Null (wie in der README)."""
    m, e = f"{x:.1e}".split("e")
    return f"{m.replace('.', ',')}e{int(e)}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",") + " %"


def num(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def thou(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def has(text: str) -> None:
    assert text in README, f"nicht in der README: {text!r}"


def _row(label: str) -> list[str]:
    for line in README.splitlines():
        if line.startswith(f"| {label} |"):
            return [c.strip() for c in line.strip().strip("|").split("|")][1:]
    raise AssertionError(f"Zeile fehlt: {label}")


def test_wege_tabelle_und_band():
    w = EV.load_data("wege")
    b = w["table"]["sigmas"]["4"]
    assert w["table"]["seed"] == 14101 and w["table"]["n"] == 30
    assert _row("σ = 4: cond(K + λI)") == [sci(c) for c in b["cond"]]
    for label, key in (("Cholesky", "cholesky"), ("LU", "lu"), ("Eigenzerlegung", "eig"), ("Minimierung", "min")):
        assert _row(f"σ = 4: {label}") == [sci(e) for e in b["err"][key]], label
    assert _row("σ = 4: Abstand der direkten Wege untereinander") == [sci(e) for e in b["spread"]]
    # Befund 1: σ = 1, λ = 1
    b1 = w["table"]["sigmas"]["1"]
    direct1 = max(b1["err"][k][0] for k in ("cholesky", "lu", "eig", "sk_kr", "sk_gp"))
    has(f"höchstens {sci(direct1)} neben der exakten Lösung, die Minimierung {sci(b1['err']['min'][0])}")
    # Befund 2: drei selbst gerechnete Wege bei σ = 4, λ = 1e-10
    three = [b["err"][k][-1] for k in ("cholesky", "lu", "eig")]
    has(f"zwischen {sci(min(three))} und {sci(max(three))}")
    assert all(b["notes"][k][-1] is None for k in ("cholesky", "lu", "eig", "sk_kr", "sk_gp"))          # keiner der fünf direkten Wege meldet etwas
    fz = EV.wege_frozen("4|1e-10")
    has(f"Fehler der fünf direkten Wege {sci(fz['direct_range'][0])} bis {sci(fz['direct_range'][1])} (cond {sci(fz['cond'][0])} bis {sci(fz['cond'][2])})")
    assert abs(fz["err"]["min"][0] - 6.3) < 0.05 and abs(fz["err"]["min"][2] - 140) < 5
    has(f"6,3 bis 140 und meldet „ABNORMAL“ in {fz['min_notes']} von {fz['n_seeds']} Läufen")
    f46 = EV.wege_frozen("4|1e-06")
    has(f"σ = 4, λ = 1e-6: {sci(f46['direct_range'][0])} bis {sci(f46['direct_range'][1])}")
    f110 = EV.wege_frozen("1|1e-10")
    has(f"σ = 1, λ = 1e-10: {sci(f110['direct_range'][0])} bis {sci(f110['direct_range'][1])} (cond {sci(f110['cond'][0])} bis {sci(f110['cond'][2])})")
    f16 = EV.wege_frozen("1|1e-06")
    has(f"σ = 1, λ = 1e-6: {sci(f16['direct_range'][0])} bis {sci(f16['direct_range'][1])}")
    allz = [EV.wege_frozen(k) for k in w["band"]["settings"]]
    assert len(allz) == 4 and sum(z["silent_runs"] for z in allz) == 80 and sum(z["n_seeds"] for z in allz) == 80
    has("In allen 80 Läufen")
    lo, hi = min(z["ratio"][0] for z in allz), max(z["ratio"][2] for z in allz)
    has(f"zwischen {num(lo, 2)} und {num(hi, 2)}")


def test_loo_und_zaehlung():
    fl = EV.load_data("flaeche")
    lc = fl["loo_check"]
    assert lc["n_instances"] == 100 and (lc["n_min"], lc["n_max"]) == (6, 30)
    has(f"Residuum höchstens {sci(lc['dev_residual_max'])} (relativ), Varianz höchstens {sci(lc['dev_var_max'])}")
    o20, o160 = fl["ops"]["20"], fl["ops"]["160"]
    has(f"bei n = 20: {thou(o20['closed'])} gegen {thou(o20['removal'])} (Faktor {num(o20['removal'] / o20['closed'], 1)})")
    has(f"bei n = 160: {thou(o160['closed'])} gegen {thou(o160['removal'])} (Faktor {num(o160['removal'] / o160['closed'], 1)})")


def test_wahlregeln_drei_saetze():
    names = {"loo": "Leave-one-out", "gcv": "GCV", "ev_grid": "Evidenz auf dem Raster", "ev_grad": "Evidenz per Gradient"}
    sums = {s: EV.flaeche_summary(s) for s in "ABC"}
    for rule, label in names.items():
        cells = [f"{num(sums[s]['rules'][rule]['mean'])} ({sums[s]['rules'][rule]['above_2']} von {sums[s]['n']})" for s in "ABC"]
        assert _row(label) == cells, label
    assert _row("Rückblick-Zelle (nicht ausführbar)") == [num(sums[s]["hindsight"]) for s in "ABC"]
    assert _row("lineare Ridge (λ per Leave-one-out)") == [num(sums[s]["linear_mean"]) for s in "ABC"]
    assert [sums[s]["kernel_wins"] for s in "ABC"] == [30, 40, 40] and [sums[s]["n"] for s in "ABC"] == [30, 40, 40]
    has("in 30 von 30, 40 von 40 und 40 von 40 Zufalls-Seeds")
    has(f"in {sums['A']['loo_min_optimistic']} von 30, {sums['B']['loo_min_optimistic']} von 40 und {sums['C']['loo_min_optimistic']} von 40 Zufalls-Seeds")
    for s in "ABC":                                                           # Rangfolge der Mittel: Evidenz (Raster) vor Leave-one-out vor GCV in jedem Satz
        r = sums[s]["rules"]
        assert r["ev_grid"]["mean"] < r["loo"]["mean"] < r["gcv"]["mean"], s
    g = [sums[s]["rules"]["ev_grad"]["mean"] for s in "ABC"]
    e = [sums[s]["rules"]["ev_grid"]["mean"] for s in "ABC"]
    has(f"({num(g[0])} gegen {num(e[0])}, {num(g[1])} gegen {num(e[1])}, {num(g[2])} gegen {num(e[2])}; Unterschiede {num(e[0] - g[0])}, {num(e[1] - g[1])} und {num(e[2] - g[2])}")
    assert all(x > 0 for x in (e[0] - g[0], e[1] - g[1], e[2] - g[2]))
    b8 = [sums[s]["ev_grid_b_mean"] for s in "ABC"]
    has(f"(Raster 8 × 8: {num(b8[0])}, {num(b8[1])}, {num(b8[2])})")
    has(f"(GCV {num(sums['A']['rules']['gcv']['mean'])} gegen {num(sums['C']['rules']['gcv']['mean'])} durch einzelne Fehlgriffe)")
    # Neumessung gegenüber früher: GCV über das Doppelte in 5 von 30 (Satz A)
    has(f"jetzt {sums['A']['rules']['gcv']['above_2']} von 30 (Satz A)")


def test_band_abdeckung():
    d = EV.load_data("band")
    cov = np.asarray(d["oracle"]["coverage"])
    z2 = np.asarray(d["oracle"]["z2"])
    assert len(cov) == 300
    has(f"Abdeckung {pct(cov.mean())} ± {pct(cov.std(ddof=1) / np.sqrt(300))[:-2].strip()} % (300 Zufalls-Seeds, Soll 95 %)")
    has(f"mittleres z² {num(z2.mean(), 3)} ± {num(z2.std(ddof=1) / np.sqrt(300), 3)} (Soll 1)")
    assert len(d["sets"]) == 3 and d["n_seeds"] == 30
    rows = {"V1-Funktion, n = 20": ("v1", 20), "V1-Funktion, n = 60": ("v1", 60), "V1-Funktion, n = 200": ("v1", 200), "raue Funktion, n = 60": ("rau", 60),
            "raue Funktion, n = 200": ("rau", 200), "aus dem Prior, n = 60": ("prior", 60)}
    for label, (truth, n) in rows.items():
        expected = []
        for rule in ("ev_grid", "ev_grad", "loo"):
            c = EV.band_cell(truth, n, rule)
            expected.append(f"{pct(c['mean_band'][0])} bis {pct(c['mean_band'][1])} ({c['below90_band'][0]} bis {c['below90_band'][1]})")
        assert _row(label) == expected, label
    # Lesart: Evidenz-Wahl, V1, n >= 60: 93,8 % bis 96,1 %, 2 bis 7 Zufalls-Seeds unter 90 %; zu eng bei n = 20 (84,0 % bis 89,7 %) und LOO
    cells = [EV.band_cell("v1", n, r) for n in (60, 200) for r in ("ev_grid", "ev_grad")]
    lo, hi = min(c["mean_band"][0] for c in cells[:2]), max(c["mean_band"][1] for c in cells[:2])
    has(f"im Mittel {pct(lo)} bis {pct(hi)} ab, mit {min(c['below90_band'][0] for c in cells[:2])} bis {max(c['below90_band'][1] for c in cells[:2])} von 30 Zufalls-Seeds unter 90 %")
    n20 = [EV.band_cell("v1", 20, r) for r in ("ev_grid", "ev_grad", "loo")]
    has(f"bei n = 20 ({pct(min(c['mean_band'][0] for c in n20))} bis {pct(max(c['mean_band'][1] for c in n20[:2]))})")
    has(f"(n = 60: {pct(EV.band_cell('rau', 60, 'ev_grid')['mean_band'][0])} bis {pct(EV.band_cell('rau', 60, 'ev_grid')['mean_band'][1])})")
    # Neumessung früher/jetzt
    has(f"jetzt {pct(EV.band_cell('v1', 60, 'ev_grid')['mean_band'][0])} bis {pct(EV.band_cell('v1', 60, 'ev_grid')['mean_band'][1])}")


def test_container_lernkurve():
    for n in (100, 300, 1000):
        s = EV.lernkurve_summary(n)
        label = thou(n) if n == 1000 else str(n)
        assert s["n_seeds"] == 20
        assert _row(label) == [num(s["rmse_krr"]), num(s["rmse_lin"]), f"{s['wins']} von 20", f"{'+' if s['diff'] >= 0 else '−'}{num(abs(s['diff']))} ± {num(s['se'])}"], n
    has(f"exakt {num(CT.floor_exact())}")
    s1000, s100 = EV.lernkurve_summary(1000), EV.lernkurve_summary(100)
    has(f"Stichprobenmittel der 20 Zufalls-Seeds {num(s1000['floor_sample'])}")
    has(f"bei n = 1 000 holt der Kern {num(100 * s1000['gain_share'], 1)} % des RMSE")
    has(f"Der konstante Mittelwert liegt bei {num(s100['const'], 2)}")
    has(f"jetzt 7, 11 und 19 von 20 ({num(100 * s1000['gain_share'], 1)} %)")
    assert [EV.lernkurve_summary(n)["wins"] for n in (100, 300, 1000)] == [7, 11, 19]
    assert s100["diff"] < -2 * s100["se"] and abs(EV.lernkurve_summary(300)["diff"]) < 2 * EV.lernkurve_summary(300)["se"] and EV.lernkurve_summary(1000)["diff"] > 2 * s1000["se"]
    ratio = np.std(CT.kunden_effekte())
    has(f"mit {num(ratio, 2)} statt 0,36 (Soll 0,30)")


def test_keine_stuetzvektoren():
    z = EV.load_data("stuetz")
    per = z["per_instance"]
    assert z["n"] == 100 and len(per) == 50 and sum(p["zeros"] for p in per) == 0
    has("0 Nullen in 5 000 Koeffizienten")
    sv = [p["sv_share"] for p in per]
    has(f"Stützvektoranteil im Mittel {pct(np.mean(sv))} ({100 * min(sv):.0f} % bis {100 * max(sv):.0f} %)")
    has(f"{num(np.mean([p['err_ridge'] for p in per]))} (Kernel Ridge) und {num(np.mean([p['err_svc'] for p in per]))} (`SVC`)")


def test_testzahl_in_der_readme_stimmt_mit_der_suite_ueberein():
    if os.environ.get("KRD_COLLECTING"):
        return
    env = dict(os.environ, KRD_COLLECTING="1")
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=600).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, out[-300:]
    assert f"Testzahl: {m.group(1)} Tests" in README, f"README nennt eine andere Testzahl als die Suite ({m.group(1)})"


def test_readme_enthaelt_kein_gedankenstrich_im_fliesstext_und_keine_planungsbegriffe():
    body = README.split("\n---\n")[0]
    for line in body.splitlines():
        if line.startswith(("#", "|", "<!--")) or not line.strip():
            continue
        assert " – " not in line.replace("Sebastian Hanisch – Operations", ""), line[:80]
    for word in ("trägt", "tragen", "Phase 0", "Saatwert", "Stück "):
        assert word not in body, word
