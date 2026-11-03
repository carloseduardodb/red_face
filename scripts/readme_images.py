"""Gera as imagens do README a partir de um vídeo real (sequência RAVDESS).

Uso: python scripts/readme_images.py seq_01.mkv seq_01.json docs/img/ [modelo_malha.json]
(seq_*.mkv/json são gerados por scripts/ravdess_sequences.py; com o modelo, gera
também a linha do tempo do modo malha — use um modelo treinado SEM este ator)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from redface.pipeline import AnalyzerConfig, FaceColorAnalyzer
from redface.video import VideoReader
from redface.viz import draw_overlay, plot_timeline

video, segs_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
mesh_model = sys.argv[4] if len(sys.argv) > 4 else None
segs = [tuple(s) for s in json.loads(segs_path.read_text())]

# agrupa clipes consecutivos da mesma emoção em blocos (para a faixa "verdade")
blocks: list[list] = []
for a, b, lab in segs:
    if blocks and blocks[-1][2] == lab:
        blocks[-1][1] = b
    else:
        blocks.append([a, b, lab])

reader = VideoReader(video)
fps = reader.info.fps
an = FaceColorAnalyzer(fps, AnalyzerConfig(baseline_s=12.0, window_s=3.0))


def mid(label: str) -> float:
    a, b, _ = next(bl for bl in blocks if bl[2] == label)
    return a + 0.6 * (b - a)


wanted = {"calibrando": 6.0, "alegria": mid("alegria"), "raiva": mid("raiva"), "nojo": mid("nojo")}
snaps: dict[str, np.ndarray] = {}
results = []
for i, frame in reader:
    t = i / fps
    results += an.process(frame, t)
    for name, ts in wanted.items():
        if name not in snaps and t >= ts:
            snaps[name] = draw_overlay(frame, an)

plot_timeline(results, out / "linha_do_tempo_real.png", truth=[tuple(b) for b in blocks],
              title="red_face - real actor (RAVDESS actor 01): neutral, then 7 posed emotions")

tiles = [snaps[k] for k in ("calibrando", "alegria", "raiva", "nojo")]
h, w = tiles[0].shape[:2]
v = np.full((h, 8, 3), 255, np.uint8)
row1 = np.hstack([tiles[0], v, tiles[1]])
row2 = np.hstack([tiles[2], v, tiles[3]])
grid = np.vstack([row1, np.full((8, row1.shape[1], 3), 255, np.uint8), row2])
cv2.imwrite(str(out / "quadros_real.png"), grid)

ok = [r for r in results if r.status == "ok"]
print("janelas:", len(results), "ok:", len(ok))
for a, b, lab in blocks:
    em = [r.emotion for r in ok if a <= r.t_start and r.t_end <= b]
    if em:
        top = max(set(em), key=em.count)
        print(f"  bloco {lab:<9} {a:6.1f}-{b:6.1f}s  janelas {len(em):3d}  mais frequente: {top} ({100 * em.count(top) / len(em):.0f}%)")


if mesh_model:
    from redface.emotion import LDAClassifier
    from redface.mesh import MeshFaceTracker

    reader = VideoReader(video)
    an = FaceColorAnalyzer(fps, AnalyzerConfig(baseline_s=12.0, window_s=3.0), MeshFaceTracker(), LDAClassifier.load(mesh_model))
    res_mesh = []
    for i, frame in reader:
        res_mesh += an.process(frame, i / fps)
    plot_timeline(res_mesh, out / "linha_do_tempo_malha.png", truth=[tuple(b) for b in blocks],
                  title="red_face --regions malha - actor 01 (model trained only on actors 02-04)")
    ok = [r for r in res_mesh if r.status == "ok"]
    hits = n = 0
    for a, b, lab in blocks[1:]:
        em = [r.emotion for r in ok if a <= r.t_start and r.t_end <= b]
        n += len(em)
        hits += sum(e == lab for e in em)
        if em:
            top = max(set(em), key=em.count)
            print(f"  [malha] bloco {lab:<9} janelas {len(em):3d}  certas {sum(e == lab for e in em):3d}  mais frequente: {top}")
    print(f"  [malha] acerto nas janelas emocionais: {hits}/{n} = {100 * hits / max(1, n):.1f}%")
