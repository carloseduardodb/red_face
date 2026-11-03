"""red_face — hipóteses de emoção a partir da cor do rosto em vídeo comum."""

__version__ = "0.1.0"

from .emotion import LDAClassifier, Prediction, PrototypeClassifier  # noqa: E402
from .pipeline import AnalyzerConfig, FaceColorAnalyzer, WindowResult  # noqa: E402
from .regions import FixedBox, HaarFaceTracker, YuNetFaceTracker, default_tracker  # noqa: E402

__all__ = [
    "AnalyzerConfig",
    "FaceColorAnalyzer",
    "FixedBox",
    "HaarFaceTracker",
    "LDAClassifier",
    "Prediction",
    "PrototypeClassifier",
    "WindowResult",
    "YuNetFaceTracker",
    "default_tracker",
]
