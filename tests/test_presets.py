"""Presets: Vollständigkeit, gültige Werte, Standard-Preset gleich Standardeinstellungen, jedes Preset ändert mindestens einen Regler; Permalink-Begrenzung."""

import krd_constants as C
import krd_presets as P


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[name] for name in C.PRESETS)


def test_preset_keys_are_known_settings_and_values_are_valid():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            assert P._clean(spec, value) == value, (name, key)


def test_default_preset_equals_the_default_settings_and_others_differ():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {}
    for name, preset in C.PRESETS.items():
        if name != first:
            assert any(preset[k] != P.SETTING_SPECS[k].default for k in preset), name          # sonst wäre es vom Standard nicht zu unterscheiden


def test_presets_differ_from_each_other():
    states = [tuple(sorted((k, v) for k, v in {**{s: spec.default for s, spec in P.SETTING_SPECS.items()}, **pr}.items())) for pr in C.PRESETS.values()]
    assert len(set(states)) == len(states)


def test_presets_nennen_die_erwarteten_werte():
    assert C.PRESETS["Schlecht gestellt (σ = 4, λ = 1e-10)"] == {"sigma": 4.0, "lam_exp": -10}
    assert C.PRESETS["Band bei n = 60"]["n_band"] == 60 and C.PRESETS["Band bei n = 20"]["n_band"] == 20
    assert C.PRESETS["Container n = 300"] == {"vehicle": "Container-Erzeuger", "n_c": 300}


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_permalink_werte_werden_begrenzt_und_optionen_geprueft():
    spec = P.SETTING_SPECS["sigma"]
    assert P._clean(spec, 99.0) == spec.hi and P._clean(spec, -1.0) == spec.lo
    assert P._clean(P.SETTING_SPECS["n_band"], 61) is None and P._clean(P.SETTING_SPECS["n_band"], 60) == 60
    assert P._bool("True") is True and P._bool("False") is False
