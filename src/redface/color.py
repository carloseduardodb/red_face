"""Conversões de cor e índices de pele.

Cada região do rosto é resumida por quatro números:

* ``L``  – luminosidade CIELAB (L*), usada só para controle de qualidade;
* ``a``  – eixo vermelho↔verde CIELAB (a*), o "vermelhidão" usado por
  Stephen, Perrett, Re e Thorstenson nos estudos de percepção;
* ``b``  – eixo amarelo↔azul CIELAB (b*);
* ``ei`` – índice de eritema ``100·log10(R/G)`` em RGB linear. Como é uma
  razão, cancela o sombreamento (intensidade da luz) e responde à
  hemoglobina, que absorve muito verde e pouco vermelho.

a* e b* são canais oponentes, análogos aos canais vermelho–verde e
amarelo–azul usados por Benitez-Quiroz, Srinivasan & Martinez (2018).
"""

from __future__ import annotations

import cv2
import numpy as np

FEATURES: tuple[str, ...] = ("L", "a", "b", "ei")


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Remove a curva gama sRGB (entrada em 0..1)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def skin_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Máscara booleana de pixels com aparência de pele (YCrCb).

    Exclui também pixels muito escuros (sombra, sobrancelha) e muito claros
    (reflexo especular), que não carregam informação de cor da pele.
    """
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y >= 40) & (y <= 240)


def pixel_stats(pixels_bgr: np.ndarray) -> np.ndarray:
    """Resumo ``[L*, a*, b*, ei]`` de um conjunto de pixels BGR uint8 (N×3)."""
    px = np.asarray(pixels_bgr, dtype=np.uint8).reshape(-1, 3)
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    L, a, b = lab.reshape(-1, 3).mean(axis=0)
    lin = np.clip(srgb_to_linear(px[:, ::-1] / 255.0), 1e-4, None)
    ei = 100.0 * np.log10(lin[:, 0].mean() / lin[:, 1].mean())
    return np.array([L, a, b, ei], dtype=np.float64)


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Distância CIE76 entre duas cores Lab."""
    return float(np.linalg.norm(np.asarray(lab1)[:3] - np.asarray(lab2)[:3]))
