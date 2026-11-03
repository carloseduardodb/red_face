"""Da mudança de cor à hipótese de emoção.

Dois classificadores com a mesma interface:

``PrototypeClassifier`` (padrão, não precisa de treino)
    Compara a mudança de cor/pulso com protótipos tirados da literatura:

    * raiva, vergonha e alegria avermelham o rosto; medo empalidece
      (Drummond 1994, 1997, 1999; Drummond & Quah 2001);
    * o rubor de vergonha concentra-se nas bochechas (Drummond & Lim 2000);
    * rostos mais vermelhos parecem mais irritados, mais azulados parecem
      mais tristes (Nakajima et al. 2017; Minami et al. 2018);
    * observadores pintam raiva, alegria e surpresa mais vermelhas e amarelas,
      e medo, tristeza e nojo menos; vermelho puxa para raiva e verde para nojo
      (Thorstenson et al. 2018, 2019, 2021).

    Além da cor e do pulso entra um único sinal de **expressão**: o sorriso
    (largura da boca relativa à distância entre os olhos, pelos pontos do
    YuNet). Sem ele, a cor sozinha confunde sorriso com rubor: sorrir levanta e
    avermelha as bochechas, o mesmo padrão do rubor de vergonha. A fusão de cor
    com expressão é o desenho que a literatura sustenta (o próprio artigo da
    PNAS conclui que a cor é *aditiva* aos movimentos musculares).

    Os números dos protótipos são **heurísticos**: indicam direção e ordem de
    grandeza (em unidades CIELAB, onde ~0,7 é o limiar de percepção segundo
    Re et al. 2011), não valores medidos.

``LDAClassifier``
    Análise discriminante linear regularizada com classificação pela média mais
    próxima — o mesmo tipo de classificador do artigo da PNAS — treinada com
    os seus próprios vídeos rotulados.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from .regions import REGIONS

#: Atributos resumidos usados pelos protótipos.
SUMMARY_FEATURES = ("d_vermelho", "d_amarelo", "bochecha_menos_testa", "d_pulso", "d_sorriso")

#: Atributos completos (por região) usados pela LDA treinável.
FULL_FEATURES = tuple(f"{r}_{c}" for r in REGIONS for c in ("da", "db", "dei")) + ("d_pulso", "d_sorriso")

#: Escala do sorriso: boca 5 % mais larga que na linha de base = 1 unidade.
#: Os cantos da boca do YuNet são grosseiros: um sorriso largo aparece como
#: só +6–10 % na razão boca/olhos (medido em 2 atores da RAVDESS; o ruído de
#: um rosto neutro falando é de ~±2 %).
SMILE_UNIT = 0.05

#: (Δa*, Δb*, bochecha−testa Δa*, Δpulso/10 bpm, Δsorriso/20 %)
#: Alegria: bochechas avermelham com o sorriso (zigomático maior). Vergonha:
#: rubor nas bochechas com no máximo um sorriso contido (Keltner 1995).
PROTOTYPES: dict[str, tuple[float, float, float, float, float]] = {
    "neutro": (0.0, 0.0, 0.0, 0.0, 0.0),
    "alegria": (1.5, 1.0, 1.0, 0.3, 1.5),
    "raiva": (2.5, 1.0, 0.0, 1.2, -0.2),
    "vergonha": (2.5, 0.5, 1.5, 0.8, 0.3),
    "surpresa": (1.0, 1.0, 0.0, 0.5, -0.3),
    "medo": (-2.0, -1.0, 0.0, 1.2, 0.0),
    "tristeza": (-0.5, -1.5, 0.0, -0.2, -0.2),
    "nojo": (-1.5, -0.3, 0.0, 0.0, 0.0),
}

#: Valência aproximada de cada rótulo (−1 negativa … +1 positiva).
VALENCE: dict[str, float] = {
    "neutro": 0.0,
    "alegria": 1.0,
    "surpresa": 0.3,
    "raiva": -1.0,
    "vergonha": -0.5,
    "medo": -1.0,
    "tristeza": -1.0,
    "nojo": -1.0,
}


def _nanmean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summary_features(
    delta: dict[str, np.ndarray], d_hr: float | None, d_smile: float | None = None
) -> dict[str, float]:
    """Resume as mudanças por região (vs. linha de base) em 4 números."""
    da = {r: d[1] for r, d in delta.items()}
    db = {r: d[2] for r, d in delta.items()}
    cheeks = _nanmean([da.get("bochecha_esq", np.nan), da.get("bochecha_dir", np.nan)])
    forehead = da.get("testa", np.nan)
    diff = cheeks - forehead if np.isfinite(cheeks) and np.isfinite(forehead) else 0.0
    return {
        "d_vermelho": _nanmean(list(da.values())),
        "d_amarelo": _nanmean(list(db.values())),
        "bochecha_menos_testa": float(diff),
        "d_pulso": float("nan") if d_hr is None else d_hr / 10.0,
        "d_sorriso": float("nan") if d_smile is None else d_smile / SMILE_UNIT,
    }


def full_features(
    delta: dict[str, np.ndarray], d_hr: float | None, d_smile: float | None = None
) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in REGIONS:
        d = delta.get(r)
        for name, idx in (("da", 1), ("db", 2), ("dei", 3)):
            out[f"{r}_{name}"] = float(d[idx]) if d is not None else float("nan")
    out["d_pulso"] = float("nan") if d_hr is None else d_hr / 10.0
    out["d_sorriso"] = float("nan") if d_smile is None else d_smile / SMILE_UNIT
    return out


@dataclass
class Prediction:
    label: str
    probs: dict[str, float]
    valence: float
    confidence: float


def _softmax_from_sqdist(sq: np.ndarray, sigma: float) -> np.ndarray:
    logits = -sq / (2 * sigma**2)
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def _prediction(labels: list[str], probs: np.ndarray) -> Prediction:
    i = int(np.argmax(probs))
    pd = {lab: float(p) for lab, p in zip(labels, probs)}
    val = float(sum(VALENCE.get(lab, 0.0) * p for lab, p in pd.items()))
    return Prediction(labels[i], pd, val, float(probs[i]))


class PrototypeClassifier:
    """Classificador por protótipos da literatura (sem treino)."""

    name = "prototipos"

    def __init__(self, sigma: float = 0.8, prototypes: dict[str, tuple[float, ...]] | None = None):
        self.sigma = sigma
        self.prototypes = prototypes or PROTOTYPES

    def predict(self, delta: dict[str, np.ndarray], d_hr: float | None, d_smile: float | None = None) -> Prediction:
        f = summary_features(delta, d_hr, d_smile)
        x = np.array([f[k] for k in SUMMARY_FEATURES])
        use = np.isfinite(x)  # sem pulso confiável ou sem sorriso (Haar/caixa fixa) → ignora a dimensão
        labels = list(self.prototypes)
        P = np.array([self.prototypes[lab] for lab in labels])
        sq = ((P[:, use] - x[use]) ** 2).sum(axis=1)
        return _prediction(labels, _softmax_from_sqdist(sq, self.sigma))


class LDAClassifier:
    """LDA regularizada + média mais próxima (como em Benitez-Quiroz et al. 2018)."""

    name = "lda"

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.features: tuple[str, ...] = FULL_FEATURES
        self.mu = self.W = self.centroids = None
        self.sd: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: list[str], shrinkage: float = 0.1) -> "LDAClassifier":
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        # padroniza: o encolhimento (shrinkage) da LDA depende da escala de cada atributo
        self.sd = X.std(axis=0) + 1e-9
        X = X / self.sd
        y = np.asarray(y)
        self.labels = sorted(set(y.tolist()))
        if len(self.labels) < 2:
            raise ValueError("preciso de pelo menos 2 rótulos diferentes para treinar")
        d = X.shape[1]
        self.mu = X.mean(axis=0)
        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))
        means = []
        for lab in self.labels:
            Xc = X[y == lab]
            mc = Xc.mean(axis=0)
            means.append(mc)
            Sw += (Xc - mc).T @ (Xc - mc)
            diff = (mc - self.mu)[:, None]
            Sb += len(Xc) * diff @ diff.T
        Sw /= len(X)
        Sb /= len(X)
        Sw = (1 - shrinkage) * Sw + shrinkage * (np.trace(Sw) / d + 1e-9) * np.eye(d)
        vals, vecs = eigh(Sb, Sw)
        k = min(len(self.labels) - 1, d)
        self.W = vecs[:, np.argsort(vals)[::-1][:k]]
        self.centroids = (np.array(means) - self.mu) @ self.W
        return self

    def predict_vector(self, x: np.ndarray) -> Prediction:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        if self.sd is not None:
            x = x / self.sd
        z = (x - self.mu) @ self.W
        sq = ((self.centroids - z) ** 2).sum(axis=1)
        return _prediction(self.labels, _softmax_from_sqdist(sq, 1.0))

    def predict(self, delta: dict[str, np.ndarray], d_hr: float | None, d_smile: float | None = None) -> Prediction:
        return self.predict_features(full_features(delta, d_hr, d_smile))

    def predict_features(self, features: dict[str, float]) -> Prediction:
        """Classifica a partir do dicionário de atributos da janela (inclui os da malha)."""
        missing = [k for k in self.features if k.startswith("malha_") and k not in features]
        if missing and len(missing) == sum(k.startswith("malha_") for k in self.features):
            raise RuntimeError("este modelo foi treinado com a malha; rode com --regions malha")
        return self.predict_vector(np.array([features.get(k, np.nan) for k in self.features]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "tipo": "lda",
                    "features": list(self.features),
                    "labels": self.labels,
                    "mu": self.mu.tolist(),
                    "sd": None if self.sd is None else self.sd.tolist(),
                    "W": self.W.tolist(),
                    "centroids": self.centroids.tolist(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "LDAClassifier":
        data = json.loads(Path(path).read_text())
        m = cls()
        m.features = tuple(data["features"])
        m.labels = data["labels"]
        m.mu, m.W, m.centroids = (np.array(data[k]) for k in ("mu", "W", "centroids"))
        m.sd = np.array(data["sd"]) if data.get("sd") is not None else None
        return m
