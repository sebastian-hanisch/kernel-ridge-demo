"""Gemeinsame Hilfen der Messwerkstatt (tools/krd_make_*.py): Importpfad, JSON schreiben und lesen, Kommandozeile mit --check.

Aufruf der Werkzeuge (Python des Test-venv, OMP_NUM_THREADS=1): `python tools/krd_make_wege.py` schreibt krd_wege.json, `--check` rechnet eine Zelle neu und vergleicht."""

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def path_of(name: str) -> str:
    import krd_constants as C
    return os.path.join(ROOT, C.DATA_FILES[name])


def write_json(name: str, obj) -> None:
    with open(path_of(name), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"geschrieben: {path_of(name)}")


def read_json(name: str):
    with open(path_of(name), encoding="utf-8") as fh:
        return json.load(fh)


def wants_check(argv=None) -> bool:
    return "--check" in (sys.argv[1:] if argv is None else argv)
