"""Analisador em fluxo: recebe quadros, devolve janelas analisadas."""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .emotion import Prediction, PrototypeClassifier, full_features, summary_features
from .regions import REGIONS, Box, FrameSample, default_tracker, extract_sample
from .rppg import estimate_heart_rate

#: Motivos para não arriscar uma resposta, com explicação para humanos.
REASONS = {
    "rosto_nao_detectado": "rosto ausente ou pouca pele visível na janela",
    "movimento": "a cabeça se mexeu demais (o recorte das regiões fica instável)",
    "cena_mudou": "o fundo mudou de cor/brilho: a câmera (balanço de branco/exposição) ou a luz mudou",
    "iluminacao_mudou": "o brilho do rosto mudou muito em relação à linha de base",
    "sem_linha_de_base": "não houve quadros bons suficientes para calibrar a linha de base",
}


class Tracker(Protocol):
    def update(self, frame_bgr: np.ndarray) -> Box | None: ...


class Classifier(Protocol):
    name: str

    def predict(self, delta: dict[str, np.ndarray], d_hr: float | None, d_smile: float | None = None) -> Prediction: ...


@dataclass
class AnalyzerConfig:
    baseline_s: float = 10.0  #: segundos iniciais tratados como estado neutro
    window_s: float = 4.0  #: tamanho da janela de cor
    hop_s: float = 1.0  #: passo entre janelas
    hr_window_s: float = 10.0  #: janela do pulso (precisa de vários batimentos)
    min_region_pixels: int = 40
    min_valid_fraction: float = 0.7
    max_motion: float = 0.02  #: deslocamento médio por quadro / largura do rosto
    max_bg_dL: float = 4.0  #: ΔL* do fundo tolerado
    max_bg_dchroma: float = 1.5  #: Δ(a*,b*) do fundo tolerado
    max_face_dL: float = 8.0
    min_hr_snr_db: float = -3.0


@dataclass
class Baseline:
    regions: dict[str, np.ndarray]
    background: np.ndarray | None
    hr_bpm: float | None
    hr_snr_db: float | None
    n_frames: int
    smile: float | None = None
    mesh: np.ndarray | None = None  #: modo malha: [a*, b*] médios por triângulo


@dataclass
class WindowResult:
    t_start: float
    t_end: float
    status: str  #: "calibrando" | "ok" | "indeterminado"
    reasons: list[str] = field(default_factory=list)
    delta: dict[str, np.ndarray] = field(default_factory=dict)  #: região -> Δ[L,a,b,ei]
    hr_bpm: float | None = None
    hr_snr_db: float | None = None
    d_hr: float | None = None
    d_smile: float | None = None  #: variação relativa da largura da boca (0,2 = 20 % mais larga)
    summary: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    prediction: Prediction | None = None

    @property
    def emotion(self) -> str | None:
        return self.prediction.label if self.prediction else None

    def to_row(self) -> dict[str, object]:
        def r(v: float | None, n: int = 3) -> object:
            return "" if v is None or not np.isfinite(v) else round(float(v), n)

        row: dict[str, object] = {
            "t_inicio": round(self.t_start, 2),
            "t_fim": round(self.t_end, 2),
            "status": self.status,
            "motivos": ";".join(self.reasons),
            "emocao": self.emotion or "",
            "confianca": r(self.prediction.confidence) if self.prediction else "",
            "valencia": r(self.prediction.valence) if self.prediction else "",
            "pulso_bpm": r(self.hr_bpm, 1),
            "pulso_snr_db": r(self.hr_snr_db, 1),
        }
        for k, v in self.summary.items():
            row[k] = r(v)
        for k, v in self.features.items():
            row[k] = r(v)
        if self.prediction:
            for lab, p in self.prediction.probs.items():
                row[f"p_{lab}"] = r(p)
        return row


def _mean_regions(samples: list[FrameSample]) -> dict[str, np.ndarray]:
    out = {}
    for name in REGIONS:
        vals = [s.regions[name] for s in samples if name in s.regions]
        if vals:
            out[name] = np.mean(vals, axis=0)
    return out


def _mean_mesh(samples: list[FrameSample]) -> np.ndarray | None:
    vals = [s.mesh for s in samples if s.mesh is not None]
    if not vals:
        return None
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # triângulo sem pele em todos os quadros → nan
        return np.nanmean(vals, axis=0)


def _mean_background(samples: list[FrameSample]) -> np.ndarray | None:
    vals = [s.background for s in samples if s.background is not None]
    return np.mean(vals, axis=0) if vals else None


class FaceColorAnalyzer:
    """Analisa um vídeo quadro a quadro.

    Uso::

        analyzer = FaceColorAnalyzer(fps=30)
        for t, frame in quadros:
            for janela in analyzer.process(frame, t):
                print(janela.status, janela.emotion)
    """

    def __init__(
        self,
        fps: float,
        config: AnalyzerConfig | None = None,
        tracker: Tracker | None = None,
        classifier: Classifier | None = None,
    ):
        self.fps = float(fps)
        self.cfg = config or AnalyzerConfig()
        self.tracker = tracker or default_tracker()
        self.classifier = classifier or PrototypeClassifier()
        keep = max(self.cfg.window_s, self.cfg.hr_window_s) + 2
        self.samples: deque[FrameSample] = deque(maxlen=int(self.fps * keep) + 1)
        self._calib: deque[FrameSample] = deque(maxlen=int(self.fps * self.cfg.baseline_s) + 1)
        self.baseline: Baseline | None = None
        self._next_emit = self.cfg.window_s
        self._prev_box: Box | None = None
        self.last_sample: FrameSample | None = None
        #: modo malha: ligado quando o rastreador fornece a malha (MeshFaceTracker)
        self.use_mesh = hasattr(self.tracker, "mesh_points")
        self.last_result: WindowResult | None = None

    # ------------------------------------------------------------------ fluxo
    def process(self, frame_bgr: np.ndarray, t: float) -> list[WindowResult]:
        box = self.tracker.update(frame_bgr)
        sample = extract_sample(
            frame_bgr, box, t, self._prev_box, self.cfg.min_region_pixels, getattr(self.tracker, "landmarks", None)
        )
        self._prev_box = box
        if self.use_mesh and sample.ok and getattr(self.tracker, "mesh_points", None) is not None:
            from .mesh import mesh_stats

            sample.mesh = mesh_stats(frame_bgr, self.tracker.mesh_points)
        self.last_sample = sample
        self.samples.append(sample)
        if self.baseline is None:
            self._calib.append(sample)
            if t >= self.cfg.baseline_s:
                self._try_baseline()
        results = []
        while t >= self._next_emit:
            results.append(self._window(self._next_emit))
            self._next_emit += self.cfg.hop_s
        if results:
            self.last_result = results[-1]
        return results

    @property
    def calibrating(self) -> bool:
        return self.baseline is None

    # ------------------------------------------------------------- internals
    def _try_baseline(self) -> None:
        ok = [s for s in self._calib if s.ok]
        if len(ok) < self.cfg.min_valid_fraction * self.fps * self.cfg.baseline_s:
            return  # continua calibrando até juntar quadros bons suficientes
        hr, snr = estimate_heart_rate(np.array([s.pulse_rgb for s in ok]), self.fps)
        if snr is not None and snr < self.cfg.min_hr_snr_db:
            hr = None
        smiles = [s.smile for s in ok if np.isfinite(s.smile)]
        smile = float(np.mean(smiles)) if len(smiles) >= 0.5 * len(ok) else None
        self.baseline = Baseline(_mean_regions(ok), _mean_background(ok), hr, snr, len(ok), smile, _mean_mesh(ok))

    def _window(self, t_end: float) -> WindowResult:
        cfg = self.cfg
        res = WindowResult(t_end - cfg.window_s, t_end, status="calibrando")
        if self.baseline is None:
            if t_end > 3 * cfg.baseline_s:
                res.status, res.reasons = "indeterminado", ["sem_linha_de_base"]
            return res
        win = [s for s in self.samples if t_end - cfg.window_s < s.t <= t_end]
        ok = [s for s in win if s.ok]
        reasons = []
        if not win or len(ok) < cfg.min_valid_fraction * len(win):
            reasons.append("rosto_nao_detectado")
        motions = [s.motion for s in ok if np.isfinite(s.motion)]
        if motions and np.mean(motions) > cfg.max_motion:
            reasons.append("movimento")

        means = _mean_regions(ok)
        res.delta = {r: m - self.baseline.regions[r] for r, m in means.items() if r in self.baseline.regions}
        # Os testes de qualidade são feitos quadro a quadro: um salto de
        # exposição de meio segundo já contamina a média da janela inteira.
        base = self.baseline
        face_dL = [
            np.mean([s.regions[r][0] - base.regions[r][0] for r in s.regions if r in base.regions]) for s in ok
        ]
        if face_dL and np.max(np.abs(face_dL)) > cfg.max_face_dL:
            reasons.append("iluminacao_mudou")
        if base.background is not None:
            for s in ok:
                if s.background is None:
                    continue
                dbg = s.background - base.background
                if abs(dbg[0]) > cfg.max_bg_dL or np.hypot(dbg[1], dbg[2]) > cfg.max_bg_dchroma:
                    reasons.append("cena_mudou")
                    break

        hr_samples = [s for s in self.samples if t_end - cfg.hr_window_s < s.t <= t_end and s.ok]
        if len(hr_samples) >= 0.7 * cfg.hr_window_s * self.fps:
            hr, snr = estimate_heart_rate(np.array([s.pulse_rgb for s in hr_samples]), self.fps)
            res.hr_snr_db = snr
            if hr is not None and snr is not None and snr >= cfg.min_hr_snr_db:
                res.hr_bpm = hr
                if self.baseline.hr_bpm is not None:
                    res.d_hr = hr - self.baseline.hr_bpm

        smiles = [s.smile for s in ok if np.isfinite(s.smile)]
        if self.baseline.smile and smiles:
            res.d_smile = float(np.mean(smiles) / self.baseline.smile - 1.0)

        if not res.delta and "rosto_nao_detectado" not in reasons:
            reasons.append("rosto_nao_detectado")
        res.summary = summary_features(res.delta, res.d_hr, res.d_smile) if res.delta else {}
        res.features = full_features(res.delta, res.d_hr, res.d_smile) if res.delta else {}
        mesh = _mean_mesh(ok)
        if res.features and mesh is not None and self.baseline.mesh is not None:
            from .mesh import mesh_feature_names

            res.features.update(zip(mesh_feature_names(), (mesh - self.baseline.mesh).ravel().tolist()))
        res.reasons = reasons
        if reasons:
            res.status = "indeterminado"
        else:
            res.status = "ok"
            if hasattr(self.classifier, "predict_features"):
                res.prediction = self.classifier.predict_features(res.features)
            else:
                res.prediction = self.classifier.predict(res.delta, res.d_hr, res.d_smile)
        return res
