"""Modo malha: ~133 triângulos de pele, como os 126 do artigo da PNAS.

Usa o MediaPipe Face Landmarker (478 pontos) — ``pip install "red-face[malha]"``.
A triangulação é fixa (``data/malha_133.json``), para que um modelo treinado
veja sempre os mesmos triângulos em qualquer vídeo.

Atenção ao interpretar: na RAVDESS a malha separou emoções posadas bem melhor
que as 5 regiões, mas o ganho veio quase todo dos triângulos junto de olhos,
sobrancelhas e boca — ou seja, provavelmente *expressão vista pela cor*, não
fluxo de sangue. Ela só serve com um classificador treinado (``redface train``):
os protótipos da literatura não falam em triângulos.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .color import skin_mask
from .regions import Box, _SmoothedTracker

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

#: Índices do MediaPipe usados como os "5 pontos" (mesma função dos do YuNet):
#: centro da íris direita e esquerda, ponta do nariz, cantos da boca.
FIVE_POINTS = (468, 473, 1, 61, 291)


@lru_cache(maxsize=1)
def triangles() -> np.ndarray:
    data = json.loads((Path(__file__).parent / "data" / "malha_133.json").read_text())
    return np.array(data["triangulos"], dtype=np.int64)


def mesh_feature_names() -> list[str]:
    return [f"malha_{k:03d}_{c}" for k in range(len(triangles())) for c in ("da", "db")]


def model_path() -> str:
    """Baixa o modelo (~3,7 MB) para ``~/.cache/redface`` na 1ª vez (ou ``REDFACE_FACEMESH``)."""
    import hashlib
    import os
    import urllib.request

    env = os.environ.get("REDFACE_FACEMESH")
    if env:
        return env
    path = Path.home() / ".cache" / "redface" / "face_landmarker.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"baixando o modelo de malha facial para {path} …")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != MODEL_SHA256:
            tmp.unlink()
            raise RuntimeError("o modelo de malha baixado não confere (SHA-256 diferente)")
        tmp.rename(path)
    return str(path)


class MeshFaceTracker(_SmoothedTracker):
    """Rastreador com MediaPipe: caixa (pelos olhos, como o YuNet), 5 pontos e malha.

    Expõe ``landmarks`` (5×2) e ``mesh_points`` (478×2, em pixels) do quadro atual.
    """

    def __init__(self, smoothing: float = 0.7, max_missing: int = 10):
        super().__init__(smoothing, max_missing)
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('o modo malha precisa do MediaPipe: pip install "red-face[malha]"') from exc
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path()), running_mode=vision.RunningMode.IMAGE, num_faces=1
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.landmarks: np.ndarray | None = None
        self.mesh_points: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> Box | None:
        mp = self._mp
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        res = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        self.landmarks = self.mesh_points = None
        box = None
        if res.face_landmarks:
            h, w = frame_bgr.shape[:2]
            pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            self.mesh_points = pts
            self.landmarks = pts[list(FIVE_POINTS)]
            eye_r, eye_l = self.landmarks[0], self.landmarks[1]
            eyes = (eye_r + eye_l) / 2
            side = max(np.hypot(*(eye_l - eye_r)) / 0.40, 1.0)
            box = np.array([eyes[0] - side / 2, eyes[1] - 0.38 * side, side, side])
        return self._push(box)


def mesh_stats(frame_bgr: np.ndarray, points: np.ndarray, min_pixels: int = 10) -> np.ndarray:
    """Média de a* e b* da pele em cada triângulo: array ``T×2`` (``nan`` se faltar pele)."""
    tris = triangles()
    T = len(tris)
    out = np.full((T, 2), np.nan)
    used = points[np.unique(tris)]
    x0, y0 = np.floor(used.min(axis=0)).astype(int).clip(0)
    x1, y1 = np.ceil(used.max(axis=0)).astype(int) + 1
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return out
    labels = np.zeros(crop.shape[:2], np.int32)
    for k, t in enumerate(tris):
        cv2.fillConvexPoly(labels, np.round(points[t] - [x0, y0]).astype(np.int32), k + 1)
    labels[~skin_mask(crop)] = 0
    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    flat = labels.ravel()
    n = np.bincount(flat, minlength=T + 1)[1:]
    good = n >= min_pixels
    for j, ch in enumerate((1, 2)):
        sums = np.bincount(flat, lab[..., ch].ravel(), minlength=T + 1)[1:]
        out[good, j] = sums[good] / n[good]
    return out
