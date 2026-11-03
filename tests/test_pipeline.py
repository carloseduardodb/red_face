from redface.pipeline import AnalyzerConfig, FaceColorAnalyzer
from redface.regions import FixedBox
from redface.synth import Segment, face_geometry, synthetic_frames

SIZE = (320, 240)
SCENARIO = (
    Segment(0, 10, "neutro", hr=70),
    Segment(10, 22, "raiva", da=3.0, db=1.0, hr=90),
    Segment(22, 27, "luz", hr=90, exposure=1.3),
)


def run():
    _, _, box = face_geometry(SIZE)
    an = FaceColorAnalyzer(30.0, AnalyzerConfig(baseline_s=8.0), FixedBox(box))
    results = []
    for t, frame, _ in synthetic_frames(SCENARIO, 30.0, SIZE, tau_s=1.5):
        results += an.process(frame, t)
    return an, results


def test_end_to_end_synthetic():
    an, results = run()
    assert an.baseline is not None and abs(an.baseline.hr_bpm - 70) <= 3
    by_end = {round(r.t_end): r for r in results}
    assert by_end[6].status == "calibrando"
    assert by_end[9].emotion == "neutro"
    flushed = by_end[21]
    assert flushed.status == "ok" and flushed.emotion == "raiva"
    assert flushed.summary["d_vermelho"] > 2
    # o salto de exposição precisa ser recusado, não virar "emoção"
    assert by_end[25].status == "indeterminado"
    assert "cena_mudou" in by_end[25].reasons


def test_no_face_is_indeterminate():
    import numpy as np

    class NoFace:
        def update(self, frame):
            return None

    an = FaceColorAnalyzer(30.0, AnalyzerConfig(baseline_s=2.0), NoFace())
    out = []
    for i in range(30 * 8):
        out += an.process(np.zeros((120, 160, 3), np.uint8), i / 30)
    assert all(r.status in {"calibrando", "indeterminado"} for r in out)
    assert out[-1].status == "indeterminado"
