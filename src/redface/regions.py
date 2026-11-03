"""Detecção do rosto e recorte das regiões de pele."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .color import pixel_stats, skin_mask

Box = tuple[int, int, int, int]  # x, y, largura, altura

#: Regiões em frações da caixa do rosto (x0, y0, x1, y1).
#: Olhos e boca ficam de fora, como no artigo da PNAS, porque não são pele.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "testa": (0.30, 0.06, 0.70, 0.20),
    "bochecha_esq": (0.14, 0.50, 0.36, 0.70),
    "bochecha_dir": (0.64, 0.50, 0.86, 0.70),
    "nariz": (0.43, 0.42, 0.57, 0.62),
    "queixo": (0.40, 0.90, 0.60, 1.00),
}

#: Regiões usadas no sinal de pulso (rPPG): as mais vascularizadas e estáveis.
PULSE_REGIONS = ("testa", "bochecha_esq", "bochecha_dir")


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def yunet_model_path() -> str:
    """Caminho do modelo YuNet; baixa (~230 KB) para ``~/.cache/redface`` na 1ª vez.

    Pode ser apontado para um arquivo local com a variável ``REDFACE_YUNET``.
    """
    import hashlib
    import os
    import urllib.request
    from pathlib import Path

    env = os.environ.get("REDFACE_YUNET")
    if env:
        return env
    path = Path.home() / ".cache" / "redface" / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"baixando o detector de rosto YuNet para {path} …")
        urllib.request.urlretrieve(YUNET_URL, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != YUNET_SHA256:
            tmp.unlink()
            raise RuntimeError("o modelo YuNet baixado não confere (SHA-256 diferente)")
        tmp.rename(path)
    return str(path)


class _SmoothedTracker:
    """Suavização temporal da caixa e tolerância a perdas curtas."""

    def __init__(self, smoothing: float, max_missing: int):
        self.smoothing = smoothing
        self.max_missing = max_missing
        self._box: np.ndarray | None = None
        self._missing = 0

    def _push(self, box: np.ndarray | None) -> Box | None:
        if box is not None:
            if self._box is not None and np.hypot(*(box[:2] - self._box[:2])) > 0.5 * box[2]:
                self._box = None  # salto grande (corte de câmera / outra pessoa): recomeça
            self._box = box if self._box is None else self.smoothing * self._box + (1 - self.smoothing) * box
            self._missing = 0
        else:
            self._missing += 1
            if self._missing > self.max_missing:
                self._box = None
        return None if self._box is None else tuple(int(round(v)) for v in self._box)  # type: ignore[return-value]


class YuNetFaceTracker(_SmoothedTracker):
    """Detector YuNet (rede neural leve do OpenCV ≥ 4.8, padrão do red_face).

    A caixa é reconstruída a partir dos **olhos** (a distância entre eles não
    muda com a expressão), para que as frações de ``REGIONS`` caiam sempre na
    testa, bochechas, nariz e queixo. Os 5 pontos do quadro atual ficam em
    ``self.landmarks`` (olho dir., olho esq., nariz, boca dir., boca esq.).
    """

    def __init__(self, smoothing: float = 0.7, max_missing: int = 10, detect_width: int = 640, score: float = 0.8):
        super().__init__(smoothing, max_missing)
        self.detector = cv2.FaceDetectorYN.create(yunet_model_path(), "", (320, 320), score, 0.3, 50)
        self.detect_width = detect_width
        self._size: tuple[int, int] | None = None
        self.landmarks: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> Box | None:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.detect_width / w)
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1 else frame_bgr
        size = (small.shape[1], small.shape[0])
        if size != self._size:
            self.detector.setInputSize(size)
            self._size = size
        _, faces = self.detector.detect(small)
        box = None
        self.landmarks = None
        if faces is not None and len(faces):
            f = max(faces, key=lambda f: f[2] * f[3]) / np.r_[[scale] * 14, 1.0]
            self.landmarks = f[4:14].reshape(5, 2)
            right_eye, left_eye = f[4:6], f[6:8]
            eyes = (right_eye + left_eye) / 2
            # Em REGIONS os olhos ficam a 30 % e 70 % da largura (distância =
            # 0,40·lado) e a 38 % da altura, como num rosto frontal padrão.
            side = max(np.hypot(*(left_eye - right_eye)) / 0.40, 1.0)
            box = np.array([eyes[0] - side / 2, eyes[1] - 0.38 * side, side, side])
        return self._push(box)


class HaarFaceTracker(_SmoothedTracker):
    """Detector Haar clássico (só existe no OpenCV 4.x; removido no OpenCV 5)."""

    def __init__(self, smoothing: float = 0.7, max_missing: int = 10, detect_width: int = 320):
        super().__init__(smoothing, max_missing)
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError("este OpenCV não tem detectores Haar (OpenCV ≥ 5); use YuNetFaceTracker")
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"não consegui carregar o detector de rosto em {path}")
        self.detect_width = detect_width

    def update(self, frame_bgr: np.ndarray) -> Box | None:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.detect_width / w)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1 else gray
        faces = self.cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        box = max(faces, key=lambda f: f[2] * f[3]).astype(np.float64) / scale if len(faces) else None
        return self._push(box)


def default_tracker(kind: str = "auto"):
    """``yunet`` (padrão), ``haar`` ou ``auto`` (YuNet; se falhar, Haar)."""
    if kind == "haar":
        return HaarFaceTracker()
    try:
        return YuNetFaceTracker()
    except Exception as exc:
        if kind == "yunet" or not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(f"não consegui iniciar o detector YuNet: {exc}") from exc
        print(f"aviso: YuNet indisponível ({exc}); usando Haar")
        return HaarFaceTracker()


class FixedBox:
    """"Detector" que devolve sempre a mesma caixa (útil para vídeos com o
    rosto parado, apoio de queixo ou vídeos sintéticos)."""

    def __init__(self, box: Box):
        self.box = box

    def update(self, frame_bgr: np.ndarray) -> Box:  # noqa: ARG002
        return self.box


def region_rects(box: Box, frame_shape: tuple[int, ...]) -> dict[str, Box]:
    """Retângulos (x, y, w, h) de cada região dentro do quadro."""
    x, y, w, h = box
    H, W = frame_shape[:2]
    out: dict[str, Box] = {}
    for name, (fx0, fy0, fx1, fy1) in REGIONS.items():
        x0 = int(np.clip(x + fx0 * w, 0, W))
        x1 = int(np.clip(x + fx1 * w, 0, W))
        y0 = int(np.clip(y + fy0 * h, 0, H))
        y1 = int(np.clip(y + fy1 * h, 0, H))
        if x1 > x0 and y1 > y0:
            out[name] = (x0, y0, x1 - x0, y1 - y0)
    return out


def background_rects(box: Box, frame_shape: tuple[int, ...]) -> list[Box]:
    """Faixas de fundo à esquerda e à direita do rosto.

    Servem de "testemunha": se o fundo muda de cor ou brilho, quem mudou foi a
    câmera ou a luz, não o sangue no rosto.
    """
    x, y, w, h = box
    H, W = frame_shape[:2]
    rects = []
    for x0, x1 in ((x - 0.55 * w, x - 0.20 * w), (x + 1.20 * w, x + 1.55 * w)):
        x0, x1 = int(np.clip(x0, 0, W)), int(np.clip(x1, 0, W))
        y0, y1 = int(np.clip(y + 0.05 * h, 0, H)), int(np.clip(y + 0.45 * h, 0, H))
        if (x1 - x0) * (y1 - y0) >= 100:
            rects.append((x0, y0, x1 - x0, y1 - y0))
    return rects


@dataclass
class FrameSample:
    """Medidas de um quadro."""

    t: float
    box: Box | None = None
    regions: dict[str, np.ndarray] = field(default_factory=dict)  # nome -> [L,a,b,ei]
    pulse_rgb: np.ndarray | None = None  # RGB médio da pele (para rPPG)
    background: np.ndarray | None = None  # [L,a,b] do fundo
    motion: float = float("nan")  # deslocamento do centro / largura da caixa
    smile: float = float("nan")  # largura da boca / distância entre os olhos
    mesh: np.ndarray | None = None  # modo malha: média [a*, b*] por triângulo (T×2)
    ok: bool = False


def extract_sample(
    frame_bgr: np.ndarray,
    box: Box | None,
    t: float,
    prev_box: Box | None = None,
    min_region_pixels: int = 40,
    landmarks: np.ndarray | None = None,
) -> FrameSample:
    """Mede um quadro. Com ``landmarks`` (5 pontos do YuNet), a boca é excluída
    das regiões — lábios e dentes não são pele e mudariam a cor medida quando a
    pessoa fala ou sorri — e calcula-se o índice de sorriso."""
    s = FrameSample(t=t, box=box)
    if box is None:
        return s
    mouth_ellipse = None
    if landmarks is not None:
        lm = np.asarray(landmarks, dtype=np.float64)
        eye_dist = np.hypot(*(lm[1] - lm[0]))
        mouth_w = np.hypot(*(lm[4] - lm[3]))
        if eye_dist > 1:
            s.smile = float(mouth_w / eye_dist)
            mid = (lm[3] + lm[4]) / 2
            # elipse que cobre lábios, dentes e boca aberta (até ~0,45·distância
            # entre os olhos abaixo dos cantos), sem engolir o queixo
            mouth_ellipse = (mid + np.array([0.0, 0.12 * eye_dist]), (0.55 * mouth_w + 0.10 * eye_dist, 0.33 * eye_dist))
    pulse_px = []
    for name, (rx, ry, rw, rh) in region_rects(box, frame_bgr.shape).items():
        crop = frame_bgr[ry : ry + rh, rx : rx + rw]
        mask = skin_mask(crop)
        if mouth_ellipse is not None:
            (cx, cy), (ax, ay) = mouth_ellipse
            yy, xx = np.mgrid[ry : ry + rh, rx : rx + rw]
            mask &= ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 > 1.0
        px = crop[mask]
        if len(px) < min_region_pixels:
            continue
        s.regions[name] = pixel_stats(px)
        if name in PULSE_REGIONS:
            pulse_px.append(px)
    if pulse_px:
        s.pulse_rgb = np.concatenate(pulse_px)[:, ::-1].astype(np.float64).mean(axis=0)

    bg = [frame_bgr[ry : ry + rh, rx : rx + rw].reshape(-1, 3) for rx, ry, rw, rh in background_rects(box, frame_bgr.shape)]
    if bg:
        s.background = pixel_stats(np.concatenate(bg))[:3]

    if prev_box is not None:
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        px_, py_ = prev_box[0] + prev_box[2] / 2, prev_box[1] + prev_box[3] / 2
        s.motion = float(np.hypot(cx - px_, cy - py_) / max(box[2], 1))
    s.ok = len(s.regions) >= 3 and s.pulse_rgb is not None
    return s
