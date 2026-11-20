"""Plotly-Abbildungen (alle Achsen fest, `fixedrange`, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import math

import numpy as np
import plotly.graph_objects as go

import krd_constants as C

_LAYOUT = dict(margin=dict(l=60, r=20, t=40, b=50), height=340, legend=dict(orientation="h", y=-0.28), template="plotly_white")


def _fix(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _color(err: float) -> str:
    return C.COLOR_GOOD if err < 1e-8 else ("#d9a400" if err < 1e-6 else C.COLOR_BAD)


def build_wege(errors: dict, labels: dict, notes: dict | None = None) -> go.Figure:
    """Größte Abweichung der Vorhersage je Weg gegen die exakte Lösung (logarithmische Achse). Ein Weg ohne Ergebnis (Ausnahme) steht als Beschriftung."""
    names = [w for w in errors]
    fig = go.Figure()
    xs, ys, cols, texts = [], [], [], []
    for w in names:
        e = errors[w]
        ys.append(labels[w])
        if e is None:
            xs.append(1e-16)
            cols.append(C.COLOR_GRAY)
            texts.append("Ausnahme, kein Ergebnis")
        else:
            xs.append(max(e, 1e-16))
            cols.append(_color(e))
            texts.append(f"{e:.1e}" + (f" ({notes[w]})" if notes and notes.get(w) else ""))
    fig.add_trace(go.Bar(x=xs, y=ys, orientation="h", marker_color=cols, text=texts, textposition="outside", cliponaxis=False, showlegend=False))
    fig.update_xaxes(type="log", range=[-16, 4], title="größte Abweichung der Vorhersage gegen die exakte Lösung (200 Testpunkte)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(**{**_LAYOUT, "height": 120 + 55 * max(1, len(names)), "legend": None}, title="Rechenwege gegen die exakte Lösung")
    return _fix(fig)


def build_lambda_table(block: dict, lams: list, sigma_label: str) -> go.Figure:
    """Fehler je Weg über lambda (eingefrorene Tabelle, ein Zufalls-Seed): bei großem lambda sind alle gleich, bei kleinem lambda laufen sie auseinander."""
    names = {"cholesky": "Cholesky", "lu": "LU", "eig": "Eigenzerlegung", "sk_kr": "sklearn KernelRidge", "sk_gp": "sklearn GaussianProcessRegressor", "min": "Minimierung"}
    fig = go.Figure()
    for w, label in names.items():
        ys = [max(e, 1e-16) if e is not None else None for e in block["err"][w]]
        fig.add_trace(go.Scatter(x=lams, y=ys, mode="lines+markers", name=label, line=dict(dash="dot" if w == "min" else "solid")))
    fig.update_xaxes(type="log", autorange="reversed", title="λ")
    fig.update_yaxes(type="log", range=[-16, 3], title="größter Fehler")
    fig.update_layout(**_LAYOUT, title=f"Fehler gegen die exakte Lösung über λ, {sigma_label} (ein Zufalls-Seed, eingefroren)")
    return _fix(fig)


def build_hand(xs, ys, x_star: float, lam: float = 1.0) -> go.Figure:
    """Handbeispiel: Mittel und 95-%-Band des Polynomkerns (Grad 2) über x, die drei Datenpunkte und der Testpunkt x*."""
    x = np.array([float(v) for v in xs])
    y = np.array([float(v) for v in ys])
    grid = np.linspace(-3.2, 3.2, 129)
    kern = lambda a, b: (np.outer(a, b) + 1.0) ** 2  # noqa: E731
    A = kern(x, x) + lam * np.eye(len(x))
    Ks = kern(grid, x)
    m = Ks @ np.linalg.solve(A, y)
    v = np.maximum((grid ** 2 + 1.0) ** 2 - np.einsum("ij,ji->i", Ks, np.linalg.solve(A, Ks.T)), 0.0)
    sd = np.sqrt(v)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=m + 1.96 * sd, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grid, y=m - 1.96 * sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="95-%-Band der Funktion"))
    fig.add_trace(go.Scatter(x=grid, y=m, mode="lines", line=dict(color=C.COLOR_MEAN), name="Mittel (Kernel Ridge = GP-Mittel)"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=C.COLOR_POINTS, size=9), name="Daten"))
    fig.add_vline(x=x_star, line_dash="dash", line_color=C.COLOR_GRAY, annotation_text="x*")
    fig.update_xaxes(range=[-3.2, 3.2], title="x")
    fig.update_yaxes(range=[-20, 25], title="Funktionswert")
    fig.update_layout(**_LAYOUT, title="Handbeispiel: drei Punkte, Polynomkern Grad 2")
    return _fix(fig)


def build_band(run: dict, title: str) -> go.Figure:
    """Schnitt bei x2 = 0: Wahrheit, GP-Mittel und 95-%-Band der Funktion; Beobachtungen mit |x2| <= 0,4 als Punkte (ihr y enthält Rauschen)."""
    x = run["slice_x"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=run["slice_hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=run["slice_lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="95-%-Band der Funktion"))
    fig.add_trace(go.Scatter(x=x, y=run["slice_mean"], mode="lines", line=dict(color=C.COLOR_MEAN), name="GP-Mittel"))
    fig.add_trace(go.Scatter(x=x, y=run["slice_truth"], mode="lines", line=dict(color=C.COLOR_TRUTH, dash="dash"), name="Wahrheit"))
    near = np.abs(run["X"][:, 1]) <= 0.4
    fig.add_trace(go.Scatter(x=run["X"][near, 0], y=run["y"][near], mode="markers", marker=dict(color=C.COLOR_POINTS, size=7), name="Beobachtungen mit |x₂| ≤ 0,4"))
    fig.update_xaxes(range=[-2, 2], title="x₁ (Schnitt bei x₂ = 0)")
    fig.update_yaxes(title="Funktionswert")
    fig.update_layout(**_LAYOUT, title=title)
    return _fix(fig)


def build_flaeche(test: np.ndarray, picks: dict, best, labels: dict) -> go.Figure:
    """Überschussfehler (log10) über das Raster (sigma, lambda) für einen Zufalls-Seed; Markierungen: gewählte Zelle je Regel und Rückblick-Zelle (nicht ausführbar)."""
    sig = [f"{s:g}".replace(".", ",") for s in C.SIGMAS_A]
    lam = [f"{lm:g}".replace(".", ",") for lm in C.LAMS_A]
    fig = go.Figure(go.Heatmap(z=np.log10(np.maximum(test, 1e-6)), x=lam, y=sig, colorscale="Viridis_r", colorbar=dict(title="log10 Fehler", thickness=12)))
    fig.add_trace(go.Scatter(x=[lam[best[1]]], y=[sig[best[0]]], mode="markers", marker=dict(symbol="star", size=16, color="white", line=dict(color="black", width=1)), name="Rückblick-Zelle"))
    from krd_evaluation import nearest_cell
    for rule, (i, j) in picks.items():
        if not (isinstance(i, int) and isinstance(j, int)):
            i, j = nearest_cell(i, j)          # Gradientenverfahren: nächste Rasterzelle markieren
        fig.add_trace(go.Scatter(x=[lam[j]], y=[sig[i]], mode="markers", marker=dict(symbol="circle-open", size=18, line=dict(color=C.COLOR_BAD, width=3)), name=f"gewählt: {labels[rule]}"))
    fig.update_xaxes(title="λ", type="category")
    fig.update_yaxes(title="σ", type="category")
    fig.update_layout(**{**_LAYOUT, "height": 380}, title="Überschussfehler über (σ, λ), ein Zufalls-Seed")
    return _fix(fig)


def build_lernkurve(rows: list, live: dict | None, floor: float) -> go.Figure:
    """Lernkurve auf dem Container-Erzeuger: RMSE (log-Tage) von Kernel Ridge und linearer Ridge über n (vorgerechnet, 20 Zufalls-Seeds), Boden, optional die Live-Ziehung."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["rmse_krr"] for r in rows], mode="lines+markers", name="Kernel Ridge (Mittel über 20 Zufalls-Seeds)", line=dict(color=C.COLOR_MEAN)))
    fig.add_trace(go.Scatter(x=ns, y=[r["rmse_lin"] for r in rows], mode="lines+markers", name="lineare Ridge (Mittel über 20 Zufalls-Seeds)", line=dict(color=C.COLOR_TRUTH)))
    fig.add_hline(y=floor, line_dash="dash", line_color=C.COLOR_GRAY, annotation_text=f"Boden {floor:.4f}".replace(".", ","))
    if live:
        fig.add_trace(go.Scatter(x=[live["n"]], y=[live["rmse_krr"]], mode="markers", marker=dict(symbol="diamond", size=12, color=C.COLOR_MEAN), name="Kernel Ridge, diese Ziehung"))
        fig.add_trace(go.Scatter(x=[live["n"]], y=[live["rmse_lin"]], mode="markers", marker=dict(symbol="diamond", size=12, color=C.COLOR_TRUTH), name="lineare Ridge, diese Ziehung"))
    fig.update_xaxes(type="log", title="Trainingscontainer n", tickvals=[100, 300, 1000], ticktext=["100", "300", "1 000"])
    fig.update_yaxes(range=[0.45, 0.65], title="RMSE der log-Verweilzeit (log-Tage)")
    fig.update_layout(**{**_LAYOUT, "height": 380}, title="Container-Erzeuger: Kern gegen linear")
    return _fix(fig)


def nice_exp(x: float) -> str:
    """Zehnerpotenz als Beschriftung, z. B. 1e-10."""
    return "1" if x == 1 else f"1e{int(round(math.log10(x)))}"
