"""Desenho do overlay em vídeo e gráfico da linha do tempo."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .i18n import label, tr
from .pipeline import FaceColorAnalyzer, WindowResult
from .regions import region_rects

EMOTION_COLORS = {  # cores (hex) para o gráfico
    "neutro": "#9aa0a6",
    "alegria": "#f4b400",
    "surpresa": "#ab47bc",
    "raiva": "#d93025",
    "vergonha": "#ff7aa2",
    "medo": "#1a73e8",
    "tristeza": "#3949ab",
    "nojo": "#34a853",
    "luz": "#3c4043",  # salto de exposição (só no demo)
}


def _delta_color(da: float) -> tuple[int, int, int]:
    """Vermelho para rubor, azul para palidez (BGR)."""
    k = float(np.clip(da / 3.0, -1, 1))
    if k >= 0:
        return (int(80 * (1 - k)), int(80 * (1 - k)), int(80 + 175 * k))
    return (int(80 + 175 * -k), int(80 * (1 + k)), int(80 * (1 + k)))


def draw_overlay(frame: np.ndarray, analyzer: FaceColorAnalyzer) -> np.ndarray:
    out = frame.copy()
    s = analyzer.last_sample
    res = analyzer.last_result
    if s is not None and s.box is not None:
        x, y, w, h = s.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 1)
        for name, (rx, ry, rw, rh) in region_rects(s.box, frame.shape).items():
            da = res.delta[name][1] if res is not None and name in res.delta else 0.0
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), _delta_color(da), 2)

    lines: list[tuple[str, tuple[int, int, int]]] = []
    if analyzer.calibrating:
        t = s.t if s else 0.0
        left = max(0.0, analyzer.cfg.baseline_s - t)
        lines.append((tr("calibrating_banner", s=f"{left:.0f}"), (0, 220, 255)))
    elif res is None:
        lines.append((tr("waiting"), (200, 200, 200)))
    elif res.status == "indeterminado":
        lines.append((label("indeterminado").upper() + ": " + ", ".join(label(r) for r in res.reasons), (0, 165, 255)))
    else:
        p = res.prediction
        lines.append((f"{label(p.label).upper()}  {100 * p.confidence:.0f}%   {tr('valence')} {p.valence:+.2f}", (255, 255, 255)))
    if res is not None and res.summary:
        lines.append((f"d_a* {res.summary['d_vermelho']:+.2f}   d_b* {res.summary['d_amarelo']:+.2f}", (230, 230, 230)))
    if res is not None and res.hr_bpm is not None:
        lines.append((f"{tr('pulse')} {res.hr_bpm:.0f} bpm (SNR {res.hr_snr_db:.1f} dB)", (230, 230, 230)))
    lines.append((tr("disclaimer_overlay"), (160, 160, 160)))

    pad, lh = 10, 24
    cv2.rectangle(out, (0, 0), (out.shape[1], pad + lh * len(lines)), (20, 20, 20), -1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(out, text, (pad, pad + 16 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def plot_timeline(
    results: Sequence[WindowResult],
    path: str | Path,
    truth: Sequence[tuple[float, float, str]] | None = None,
    title: str | None = None,
) -> None:
    """Salva um PNG com Δa*/Δb*, pulso e a emoção estimada ao longo do tempo."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("instale o extra de gráficos: pip install 'red-face[plot]'") from exc

    title = title or tr("plot_title")
    tt = np.array([r.t_end for r in results])
    da = np.array([r.summary.get("d_vermelho", np.nan) if r.summary else np.nan for r in results])
    db = np.array([r.summary.get("d_amarelo", np.nan) if r.summary else np.nan for r in results])
    hr = np.array([np.nan if r.hr_bpm is None else r.hr_bpm for r in results])

    nrows = 4 if truth else 3
    ratios = [3, 2, 0.6] + ([0.6] if truth else [])
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 6.8), sharex=True, gridspec_kw={"height_ratios": ratios})
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    for r in results:
        if r.status != "ok":
            color = "#fde7c8" if r.status == "indeterminado" else "#e8eaed"
            for ax in (ax1, ax2):
                ax.axvspan(r.t_end - 0.5, r.t_end + 0.5, color=color, lw=0, zorder=0)

    ax1.axhline(0, color="#5f6368", lw=0.8)
    ax1.axhspan(-0.7, 0.7, color="#5f6368", alpha=0.08, lw=0)
    ax1.plot(tt, da, color="#d93025", lw=2, label=tr("plot_da"))
    ax1.plot(tt, db, color="#f9ab00", lw=2, label=tr("plot_db"))
    ax1.set_ylabel(tr("plot_change"))
    ax1.legend(loc="upper right", fontsize=8, frameon=False)
    ax1.set_title(title, loc="left", fontsize=11)

    ax2.plot(tt, hr, color="#1a73e8", lw=2, marker=".", ms=4)
    ax2.set_ylabel(tr("plot_pulse"))

    def band(ax, spans, label):
        for t0, t1, lab in spans:
            ax.axvspan(t0, t1, color=EMOTION_COLORS.get(lab, "#ffffff" if lab is None else "#fbbc04"), lw=0)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)

    est = [(r.t_end - 0.5, r.t_end + 0.5, r.emotion if r.status == "ok" else None) for r in results]
    est = [(a, b, lab) for a, b, lab in est if lab]
    band(ax3, est, tr("plot_estimated"))
    if truth:
        band(axes[3], truth, tr("plot_truth"))
    axes[-1].set_xlabel(tr("plot_time"))

    used = sorted({lab for *_, lab in est} | ({lab for *_, lab in truth} if truth else set()))
    handles = [Patch(color=EMOTION_COLORS.get(lab, "#fbbc04"), label=label(lab)) for lab in used]
    handles += [Patch(color="#fde7c8", label=label("indeterminado")), Patch(color="#e8eaed", label=label("calibrando"))]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=8, frameon=False)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
