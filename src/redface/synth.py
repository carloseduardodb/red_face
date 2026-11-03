"""Vídeo sintético com "verdade de campo" conhecida.

Serve para ver o pipeline funcionando e para testes automáticos: um rosto
desenhado com pulso, rubor e palidez controlados, e um salto de exposição
(como o de uma câmera com exposição automática) para mostrar o sistema
recusando responder.

Atenção: isto valida o *software* (medição, linha de base, controle de
qualidade), não a *ciência*. Os protótipos de emoção e o cenário sintético
seguem as mesmas hipóteses da literatura, então acertar aqui é esperado.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from .regions import Box


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    label: str
    da: float = 0.0  #: Δa* alvo (vermelhidão)
    db: float = 0.0  #: Δb* alvo (amarelo)
    hr: float = 70.0  #: pulso alvo (bpm)
    exposure: float = 1.0  #: multiplicador de brilho da cena inteira


DEFAULT_SCENARIO: tuple[Segment, ...] = (
    Segment(0, 15, "neutro", hr=70),
    Segment(15, 30, "raiva", da=3.0, db=1.0, hr=88),
    Segment(30, 42, "neutro", hr=72),
    Segment(42, 57, "medo", da=-2.5, db=-1.2, hr=90),
    Segment(57, 65, "neutro", hr=72),
    Segment(65, 72, "luz", hr=72, exposure=1.25),
    Segment(72, 80, "neutro", hr=72),
)


def segment_at(scenario: tuple[Segment, ...], t: float) -> Segment:
    for seg in scenario:
        if seg.t0 <= t < seg.t1:
            return seg
    return scenario[-1]


def face_geometry(size: tuple[int, int] = (640, 480)) -> tuple[tuple[int, int], tuple[int, int], Box]:
    w, h = size
    center = (w // 2, int(h * 0.52))
    axes = (int(w * 0.17), int(h * 0.30))
    box = (center[0] - axes[0], center[1] - axes[1], 2 * axes[0], 2 * axes[1])
    return center, axes, box


def synthetic_frames(
    scenario: tuple[Segment, ...] = DEFAULT_SCENARIO,
    fps: float = 30.0,
    size: tuple[int, int] = (640, 480),
    seed: int = 7,
    tau_s: float = 3.0,
) -> Iterator[tuple[float, np.ndarray, str]]:
    """Gera ``(t, quadro_bgr, rótulo_verdadeiro)``."""
    rng = np.random.default_rng(seed)
    w, h = size
    center, axes, (bx, by, bw, bh) = face_geometry(size)

    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    face = mask > 0
    features = np.zeros((h, w), np.uint8)  # olhos, sobrancelhas, boca
    for fx in (0.30, 0.70):
        cv2.ellipse(features, (int(bx + fx * bw), int(by + 0.38 * bh)), (int(0.07 * bw), int(0.03 * bh)), 0, 0, 360, 255, -1)
        cv2.line(features, (int(bx + (fx - 0.09) * bw), int(by + 0.31 * bh)), (int(bx + (fx + 0.09) * bw), int(by + 0.30 * bh)), 255, 6)
    cv2.ellipse(features, (int(bx + 0.5 * bw), int(by + 0.77 * bh)), (int(0.12 * bw), int(0.03 * bh)), 0, 0, 360, 255, -1)
    dark = features > 0

    texture = rng.normal(0, 5, (h, w, 1)).astype(np.float32)
    shading = np.linspace(1.04, 0.94, w, dtype=np.float32)[None, :, None]  # luz vindo da esquerda
    bg = np.empty((h, w, 3), np.float32)
    bg[:] = (118, 122, 112)
    bg[:, : w // 3] = (95, 120, 150)  # uma parede de cor diferente ajuda a ver a "testemunha"

    skin_bgr = np.array([[[120, 140, 200]]], np.float32) / 255.0
    skin_lab = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2Lab)[0, 0]

    state = np.array([0.0, 0.0, 70.0, 1.0])  # da, db, hr, exposição
    phase = 0.0
    dt = 1.0 / fps
    n = int(round(scenario[-1].t1 * fps))
    for i in range(n):
        t = i * dt
        seg = segment_at(scenario, t)
        target = np.array([seg.da, seg.db, seg.hr, seg.exposure])
        rate = np.array([dt / tau_s, dt / tau_s, dt / tau_s, min(1.0, dt / 0.3)])
        state += (target - state) * rate
        phase += 2 * np.pi * state[2] / 60.0 * dt
        pulse = np.sin(phase) + 0.3 * np.sin(2 * phase)

        lab = skin_lab + np.array([-0.25 * pulse, state[0] + 0.35 * pulse, state[1]], np.float32)
        color = cv2.cvtColor(lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2BGR)[0, 0] * 255.0

        frame = bg.copy()
        frame[face] = color
        frame[dark] = (45, 50, 70)
        frame = frame * shading * state[3] + texture + rng.normal(0, 2.0, (h, w, 1)).astype(np.float32)
        yield t, np.clip(frame, 0, 255).astype(np.uint8), seg.label
