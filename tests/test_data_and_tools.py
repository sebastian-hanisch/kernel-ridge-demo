"""Vorgerechnete Messungen: Form der Dateien und Neurechnung einer Zelle je Werkzeug (`--check`, als Band; Plattform kann in der letzten Stelle abweichen)."""

import importlib

import pytest

import krd_constants as C
import krd_evaluation as EV


def _tool(name):
    return importlib.import_module(name)


def test_alle_messdateien_sind_da_und_haben_die_erwartete_form():
    w, f, b, lk, st = (EV.load_data(k) for k in ("wege", "flaeche", "band", "lernkurve", "stuetz"))
    assert w["table"]["lams"] == [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10] and set(w["table"]["sigmas"]) == {"1", "4"}
    assert set(w["band"]["settings"]) == {"1|1e-10", "1|1e-06", "4|1e-10", "4|1e-06"} and all(len(s["per_seed"]) == 20 for s in w["band"]["settings"].values())
    assert [len(f["sets"][k]["rows"]) for k in "ABC"] == [30, 40, 40] and f["n"] == 60
    assert len(b["sets"]) == 3 and all(len(s["cells"]) == 36 for s in b["sets"]) and all(len(c["coverage"]) == 30 for s in b["sets"] for c in s["cells"].values())
    assert all(len(r) == 20 for r in lk["runs"].values()) and lk["ns"] == [100, 300, 1000] and len(st["per_instance"]) == 50


def test_band_zellen_decken_alle_regler_kombinationen_ab():
    keys = set(EV.load_data("band")["sets"][0]["cells"])
    assert keys == {f"{t}|{n}|{r}" for t in C.TRUTH_KEYS.values() for n in C.N_BAND_OPTIONS for r in C.RULE_KEYS.values()}


def test_abdeckungen_sind_anteile_und_z2_positiv():
    for s in EV.load_data("band")["sets"]:
        for c in s["cells"].values():
            assert all(0.0 <= v <= 1.0 for v in c["coverage"]) and all(v > 0 for v in c["z2"])


@pytest.mark.parametrize("name", ["krd_make_flaeche", "krd_make_band", "krd_make_lernkurve", "krd_make_stuetz", "krd_make_wege"])
def test_check_rechnet_eine_zelle_neu_und_stimmt_ueberein(name):
    if name in ("krd_make_stuetz", "krd_make_wege"):
        pytest.importorskip("sklearn")
    assert _tool(name).check() == 0
