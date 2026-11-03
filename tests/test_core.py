import numpy as np
import pytest

from redface.color import pixel_stats
from redface.emotion import FULL_FEATURES, LDAClassifier, PrototypeClassifier
from redface.rppg import estimate_heart_rate


def test_redder_skin_has_higher_a_and_erythema():
    base = np.tile([[120, 140, 200]], (500, 1)).astype(np.uint8)  # BGR
    flushed = np.tile([[118, 132, 210]], (500, 1)).astype(np.uint8)
    L0, a0, b0, ei0 = pixel_stats(base)
    L1, a1, b1, ei1 = pixel_stats(flushed)
    assert a1 > a0 and ei1 > ei0


def test_erythema_index_ignores_brightness():
    # dobrar a luz = dobrar a intensidade *linear* (não os valores sRGB)
    from redface.color import srgb_to_linear

    px = np.tile([[60, 70, 100]], (100, 1)).astype(np.uint8)
    lin = srgb_to_linear(px / 255.0) * 2
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    brighter = np.round(srgb * 255).astype(np.uint8)
    assert pixel_stats(px)[3] == pytest.approx(pixel_stats(brighter)[3], abs=0.5)


@pytest.mark.parametrize("bpm", [55, 72, 110])
def test_pos_recovers_heart_rate(bpm):
    fps, secs = 30.0, 12
    t = np.arange(int(fps * secs)) / fps
    rng = np.random.default_rng(0)
    pulse = np.sin(2 * np.pi * bpm / 60 * t)
    rgb = np.stack([180 + 0.3 * pulse, 120 - 0.6 * pulse, 100 - 0.2 * pulse], axis=1)
    rgb += rng.normal(0, 0.15, rgb.shape) + np.linspace(0, 3, len(t))[:, None]  # ruído + deriva de luz
    hr, snr = estimate_heart_rate(rgb, fps)
    assert hr == pytest.approx(bpm, abs=2)
    assert snr > 0


def test_heart_rate_needs_enough_samples():
    assert estimate_heart_rate(np.ones((20, 3)), 30.0) == (None, None)


def _delta(da, db, cheeks=0.0):
    d = {}
    for r in ("testa", "nariz", "queixo"):
        d[r] = np.array([0, da, db, 0.0])
    for r in ("bochecha_esq", "bochecha_dir"):
        d[r] = np.array([0, da + cheeks, db, 0.0])
    return d


def test_prototypes_follow_literature_directions():
    clf = PrototypeClassifier()
    assert clf.predict(_delta(0, 0), 0.0).label == "neutro"
    assert clf.predict(_delta(2.6, 1.0), 12.0).label == "raiva"
    assert clf.predict(_delta(-2.0, -1.0), 12.0).label == "medo"
    assert clf.predict(_delta(1.8, 0.5, cheeks=1.8), 8.0).label == "vergonha"
    p = clf.predict(_delta(2.6, 1.0), None)  # sem pulso também funciona
    assert sum(p.probs.values()) == pytest.approx(1.0)


def test_lda_learns_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    d = len(FULL_FEATURES)
    centers = {"a": rng.normal(0, 3, d), "b": rng.normal(0, 3, d), "c": rng.normal(0, 3, d)}
    X, y = [], []
    for lab, c in centers.items():
        X.append(c + rng.normal(0, 0.5, (40, d)))
        y += [lab] * 40
    X = np.vstack(X)
    m = LDAClassifier().fit(X, y)
    acc = np.mean([m.predict_vector(x).label == lab for x, lab in zip(X, y)])
    assert acc > 0.95
    m.save(tmp_path / "m.json")
    m2 = LDAClassifier.load(tmp_path / "m.json")
    assert m2.predict_vector(X[0]).label == m.predict_vector(X[0]).label


def test_smile_separates_happiness_from_blushing():
    clf = PrototypeClassifier()
    red_cheeks = _delta(1.2, 0.3, cheeks=1.5)
    assert clf.predict(red_cheeks, None, d_smile=0.08).label == "alegria"
    assert clf.predict(_delta(2.0, 0.5, cheeks=1.5), None, d_smile=0.0).label == "vergonha"
    # sem informação de sorriso (Haar/caixa fixa) continua funcionando
    assert clf.predict(red_cheeks, None, d_smile=None).label in {"alegria", "vergonha"}


def test_lda_predict_features_uses_named_columns():
    rng = np.random.default_rng(3)
    names = ["x", "malha_000_da", "malha_000_db"]
    X = np.vstack([rng.normal(0, 1, (30, 3)) + [0, 4, 0], rng.normal(0, 1, (30, 3)) + [0, -4, 0]])
    m = LDAClassifier().fit(X, ["a"] * 30 + ["b"] * 30)
    m.features = tuple(names)
    assert m.predict_features({"x": 0.0, "malha_000_da": 4.0, "malha_000_db": 0.0}).label == "a"
    with pytest.raises(RuntimeError):  # modelo de malha usado sem malha
        m.predict_features({"x": 0.0})
