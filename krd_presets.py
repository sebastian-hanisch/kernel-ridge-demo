"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in der Seitenleiste und in SETTING_SPECS: dadurch hebt die Änderung jedes Reglers die Hervorhebung des aktiven Presets auf (tests/test_layout.py prüft das
für jeden Regler einzeln). Ein Preset nennt nur die Regler, die es verändert; alle übrigen meinen ihren Standardwert."""

from dataclasses import dataclass

import streamlit as st

import krd_constants as C
from krd_rng import SplitMix64


def _bool(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "ja", "wahr")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: object = None
    hi: object = None
    options: tuple | None = None


SETTING_SPECS = {
    "center": SettingSpec("zentriert", _bool, True, False, True),
    "n_w": SettingSpec("nw", int, C.N_W[2], C.N_W[0], C.N_W[1]),
    "sigma": SettingSpec("sigma", float, C.SIGMA[2], C.SIGMA[0], C.SIGMA[1]),
    "lam_exp": SettingSpec("lam", int, C.LAM_EXP[2], C.LAM_EXP[0], C.LAM_EXP[1]),
    "seed_w": SettingSpec("sw", int, C.SEED_W[2], C.SEED_W[0], C.SEED_W[1]),
    "n_seeds_w": SettingSpec("nsw", int, C.N_SEEDS_W[2], C.N_SEEDS_W[0], C.N_SEEDS_W[1]),
    "target_w": SettingSpec("zf", str, C.TARGET_OPTIONS[0], options=C.TARGET_OPTIONS),
    "way_cholesky": SettingSpec("wc", _bool, True, False, True),
    "way_lu": SettingSpec("wl", _bool, True, False, True),
    "way_eig": SettingSpec("we", _bool, True, False, True),
    "way_min": SettingSpec("wm", _bool, True, False, True),
    "hand_x": SettingSpec("hx", float, C.HAND_X_DEFAULT, options=C.HAND_X_OPTIONS),
    "truth": SettingSpec("wh", str, C.TRUTH_OPTIONS[0], options=C.TRUTH_OPTIONS),
    "n_band": SettingSpec("nb", int, C.N_BAND_DEFAULT, options=C.N_BAND_OPTIONS),
    "rule": SettingSpec("rg", str, C.RULE_OPTIONS[0], options=C.RULE_OPTIONS),
    "seed_b": SettingSpec("sb", int, C.SEED_B[2], C.SEED_B[0], C.SEED_B[1]),
    "n_l": SettingSpec("nl", int, C.N_L[2], C.N_L[0], C.N_L[1]),
    "rule_l": SettingSpec("rl", str, C.RULE3_OPTIONS[0], options=C.RULE3_OPTIONS),
    "seed_l": SettingSpec("sl", int, C.SEED_L[2], C.SEED_L[0], C.SEED_L[1]),
    "vehicle": SettingSpec("vh", str, C.VEHICLE_OPTIONS[0], options=C.VEHICLE_OPTIONS),
    "n_c": SettingSpec("nc", int, C.N_C[2], C.N_C[0], C.N_C[1]),
    "seed_c": SettingSpec("sc", int, C.SEED_C[2], C.SEED_C[0], C.SEED_C[1]),
}
PRESET_KEYS = {k: k for k in SETTING_SPECS}


def first_other_value(spec: SettingSpec, current):
    """Ein gültiger Wert, der vom aktuellen verschieden ist (für Tests der Hervorhebung)."""
    if spec.options is not None:
        return next(o for o in spec.options if o != current)
    return spec.hi if current != spec.hi else spec.lo


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _clean(spec: SettingSpec, value):
    """Wert auf die Grenzen begrenzt oder None, wenn er nicht zu den Optionen gehört."""
    if spec.options is not None:
        return value if value in spec.options else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (Optionen nur, wenn gültig)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = _clean(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler auf den Standardwert zurücksetzen und die vom Preset genannten setzen (ein Preset ist ein vollständiger Zustand)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt alle Regler):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    """Nächster Zufalls-Seed der Wege-Anzeige: eine Ziehung aus SplitMix64, die vom aktuellen Zufalls-Seed ausgeht (kein Python-`random`)."""
    st.session_state["seed_w"] = SplitMix64(int(st.session_state.get("seed_w", C.SEED_W[2]))).below(C.SEED_MAX + 1)
