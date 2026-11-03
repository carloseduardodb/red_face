"""Compara o pulso do red_face com o oxímetro da UBFC-rPPG (DATASET_2).

Uso: python eval_ubfc.py PASTA_DO_SUJEITO   (com vid.avi e ground_truth.txt)
Imprime uma linha JSON com os resultados por janela.
"""

import json
import sys
from pathlib import Path

import numpy as np

from redface.pipeline import AnalyzerConfig, FaceColorAnalyzer
from redface.video import VideoReader

d = Path(sys.argv[1])
gt = np.loadtxt(d / "ground_truth.txt")
gt_hr, gt_t = gt[1], gt[2]

reader = VideoReader(str(d / "vid.avi"))
fps = reader.info.fps
an = FaceColorAnalyzer(fps, AnalyzerConfig(hop_s=2.0))
rows, i = [], 0
for i, frame in reader:
    for r in an.process(frame, i / fps):
        if r.status == "calibrando":
            continue
        t0, t1 = r.t_end - an.cfg.hr_window_s, r.t_end
        m = (gt_t >= t0) & (gt_t <= t1)
        ref = float(np.median(gt_hr[m])) if m.any() else None
        rows.append({"t": r.t_end, "status": r.status, "hr": r.hr_bpm, "snr": r.hr_snr_db, "ref": ref,
                     "da": r.summary.get("d_vermelho") if r.summary else None, "emo": r.emotion, "reasons": r.reasons})
print(json.dumps({"sujeito": d.name, "fps": fps, "quadros": i, "codec": "", "janelas": rows}))
