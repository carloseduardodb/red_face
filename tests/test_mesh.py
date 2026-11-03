import numpy as np

from redface.mesh import mesh_feature_names, mesh_stats, triangles


def test_triangulation_is_fixed_and_valid():
    tris = triangles()
    assert tris.shape == (133, 3)
    assert tris.max() < 478
    assert len(mesh_feature_names()) == 2 * 133


def test_mesh_stats_measures_redder_skin():
    # malha "achatada" numa grade: cada ponto do MediaPipe vira uma posição única
    rng = np.random.default_rng(0)
    pts = rng.uniform(20, 280, (478, 2))
    frame = np.full((300, 300, 3), (120, 140, 200), np.uint8)  # pele (BGR)
    base = mesh_stats(frame, pts, min_pixels=1)
    redder = frame.copy()
    redder[..., 2] = 215
    redder[..., 1] = 133
    d = mesh_stats(redder, pts, min_pixels=1) - base
    ok = np.isfinite(d[:, 0])
    assert ok.sum() > 50
    assert np.all(d[ok, 0] > 0)  # a* subiu em todos os triângulos medidos
