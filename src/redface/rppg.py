"""Pulso por vídeo (fotopletismografia remota, rPPG).

Implementa o método POS — *Plane-Orthogonal-to-Skin* — de Wang, den Brinker,
Stuijk & de Haan (2017, IEEE TBME 64(7):1479-1491). A cada batimento o volume
de sangue na pele muda um pouco e, com ele, a cor; o POS projeta o RGB médio da
pele num plano que separa essa variação do brilho e do movimento.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, filtfilt

_POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def pos_signal(rgb: np.ndarray, fps: float, window_s: float = 1.6) -> np.ndarray:
    """Sinal de pulso POS a partir de uma série ``T×3`` de RGB médio."""
    rgb = np.asarray(rgb, dtype=np.float64)
    T = len(rgb)
    win = max(int(round(window_s * fps)), 2)
    out = np.zeros(T)
    for start in range(0, T - win + 1):
        chunk = rgb[start : start + win]
        mean = chunk.mean(axis=0)
        if np.any(mean <= 0):
            continue
        s1, s2 = _POS_PROJECTION @ (chunk / mean).T
        h = s1 + (s1.std() / (s2.std() + 1e-12)) * s2
        out[start : start + win] += h - h.mean()
    return out


def estimate_heart_rate(
    rgb: np.ndarray, fps: float, lo_hz: float = 0.7, hi_hz: float = 3.5
) -> tuple[float | None, float | None]:
    """Frequência cardíaca (bpm) e SNR (dB) de uma série RGB.

    SNR segue de Haan & Jeanne (2013): potência perto do pico e do 1º harmônico
    dividida pelo resto da banda. Devolve ``(None, None)`` se a série for curta.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if len(rgb) < 4 * fps:
        return None, None
    h = detrend(pos_signal(rgb, fps))
    nyq = fps / 2.0
    hi = min(hi_hz, 0.95 * nyq)
    b, a = butter(3, [lo_hz / nyq, hi / nyq], btype="band")
    if len(h) <= 3 * max(len(a), len(b)):
        return None, None
    h = filtfilt(b, a, h)
    n = max(len(h), int(fps * 60))  # resolução de ~1 bpm
    spec = np.abs(np.fft.rfft(h * np.hanning(len(h)), n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    band = (freqs >= lo_hz) & (freqs <= hi)
    if not band.any():
        return None, None
    f0 = freqs[band][np.argmax(spec[band])]
    near = (np.abs(freqs - f0) <= 0.1) | (np.abs(freqs - 2 * f0) <= 0.2)
    signal = spec[band & near].sum()
    noise = spec[band & ~near].sum()
    snr = 10 * np.log10(signal / noise) if noise > 0 else float("inf")
    return float(f0 * 60.0), float(snr)
