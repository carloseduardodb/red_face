"""Avalia o red_face na RAVDESS (emoções posadas por atores, vídeo sem áudio).

Para cada ator: linha de base = média dos clipes neutros dele (deixando o
próprio clipe de fora quando ele é neutro). Cada clipe vira um vetor de
mudanças de cor por região. Depois:
  1. média de Δa*/Δb* por emoção (direção bate com a literatura?)
  2. classificador por protótipos (sem treino)
  3. LDA deixando um ator de fora (como o artigo da PNAS, mas em vídeo)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from redface.emotion import FULL_FEATURES, LDAClassifier, PrototypeClassifier, full_features
from redface.regions import REGIONS, YuNetFaceTracker, extract_sample

EMO = {"01": "neutro", "02": "calmo", "03": "alegria", "04": "tristeza", "05": "raiva", "06": "medo", "07": "nojo", "08": "surpresa"}
root = Path(sys.argv[1])
cache = root / "features.json"


def clip_stats(path: Path):
    tr = YuNetFaceTracker()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    acc = defaultdict(list)
    smiles = []
    prev, i, n_ok = None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        box = tr.update(frame)
        s = extract_sample(frame, box, i / fps, prev, landmarks=tr.landmarks)
        prev = box
        i += 1
        if s.ok:
            n_ok += 1
            if np.isfinite(s.smile):
                smiles.append(s.smile)
            for r, v in s.regions.items():
                acc[r].append(v)
    return {r: np.mean(v, axis=0).tolist() for r, v in acc.items()}, n_ok, i, float(np.mean(smiles)) if smiles else None


if cache.exists():
    clips = json.loads(cache.read_text())
else:
    clips = []
    files = sorted(root.glob("Actor_*/02-01-*.mp4"))
    for k, f in enumerate(files):
        parts = f.stem.split("-")
        regions, n_ok, n, smile = clip_stats(f)
        clips.append({"file": f.name, "actor": parts[6], "emo": EMO[parts[2]], "intensity": "forte" if parts[3] == "02" else "normal", "regions": regions, "ok": n_ok, "n": n, "smile": smile})
        print(f"{k + 1}/{len(files)} {f.name} {n_ok}/{n} quadros ok", flush=True)
    cache.write_text(json.dumps(clips))

clips = [c for c in clips if c["ok"] >= 0.7 * c["n"] and len(c["regions"]) >= 4]
print(f"\nclipes válidos: {len(clips)}")


def baseline(actor, exclude):
    neu = [c for c in clips if c["actor"] == actor and c["emo"] == "neutro" and c["file"] != exclude]
    regs = {}
    for r in REGIONS:
        vals = [c["regions"][r] for c in neu if r in c["regions"]]
        if vals:  # região sem dado nos neutros fica de fora (nunca vira nan)
            regs[r] = np.mean(vals, axis=0)
    sm = [c["smile"] for c in neu if c.get("smile")]
    return regs, (float(np.mean(sm)) if sm else None)


rows = []
for c in clips:
    base, base_smile = baseline(c["actor"], c["file"])
    delta = {r: np.array(v) - base[r] for r, v in c["regions"].items() if r in base}
    c["delta"] = delta
    c["d_smile"] = (c["smile"] / base_smile - 1.0) if (c.get("smile") and base_smile) else None
    da = np.mean([d[1] for d in delta.values()])
    db = np.mean([d[2] for d in delta.values()])
    dL = np.mean([d[0] for d in delta.values()])
    rows.append((c["emo"], c["intensity"], da, db, dL, c["d_smile"] if c["d_smile"] is not None else np.nan))

print("\n1) Mudança média vs. clipes neutros do mesmo ator (unidades CIELAB)")
print(f"{'emoção':<10}{'n':>4}{'Δa* (verm.)':>14}{'Δb* (amar.)':>14}{'ΔL*':>9}{'Δsorriso':>10}   Δa* só intensidade forte")
for emo in ["neutro", "calmo", "alegria", "surpresa", "raiva", "medo", "tristeza", "nojo"]:
    sel = [r for r in rows if r[0] == emo]
    strong = [r[2] for r in sel if r[1] == "forte"]
    a = np.array([r[2] for r in sel])
    b = np.array([r[3] for r in sel])
    L = np.array([r[4] for r in sel])
    sm = np.nanmean([r[5] for r in sel])
    se = a.std(ddof=1) / np.sqrt(len(a))
    print(f"{emo:<10}{len(sel):>4}{a.mean():>+9.2f} ±{se:4.2f}{b.mean():>+14.2f}{L.mean():>+9.2f}{100 * sm:>+9.0f}%   {np.mean(strong) if strong else float('nan'):+.2f}")

print("\n2) Protótipos da literatura (sem treino, sem pulso: clipes de 3-4 s)")
target = {"neutro": "neutro", "calmo": "neutro", "alegria": "alegria", "surpresa": "surpresa", "raiva": "raiva", "medo": "medo", "tristeza": "tristeza", "nojo": "nojo"}
clf = PrototypeClassifier()
pred = [(target[c["emo"]], clf.predict(c["delta"], None, c["d_smile"])) for c in clips]
acc = np.mean([p.label == t for t, p in pred])
labels = sorted(set(target.values()))
maj = max(np.mean([t == lab for t, _ in pred]) for lab in labels)
print(f"acurácia 7 classes: {100 * acc:.1f}%   (acaso uniforme {100 / 7:.1f}%, classe majoritária {100 * maj:.1f}%)")
neg = {"raiva", "medo", "tristeza", "nojo"}
vp = [(p.valence, 1 if t == "alegria" else 0) for t, p in pred if t == "alegria" or t in neg]
from scipy.stats import mannwhitneyu  # noqa: E402

pos_v = [v for v, y in vp if y]
neg_v = [v for v, y in vp if not y]
auc = mannwhitneyu(pos_v, neg_v).statistic / (len(pos_v) * len(neg_v))
print(f"valência alegria × negativas: AUC {auc:.2f} (acaso 0,50)")
print("predições por emoção verdadeira:")
for emo in labels:
    cnt = defaultdict(int)
    for t, p in pred:
        if t == emo:
            cnt[p.label] += 1
    print(f"  {emo:<9} → " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))

print("\n3) LDA (16 atributos de cor por região) deixando um ator de fora")
X = np.array([[full_features(c["delta"], None, c["d_smile"])[k] for k in FULL_FEATURES] for c in clips])
y = [target[c["emo"]] for c in clips]
actors = sorted({c["actor"] for c in clips})
hits = n = 0
hits_v = n_v = 0
for a in actors:
    tr = [i for i, c in enumerate(clips) if c["actor"] != a]
    te = [i for i, c in enumerate(clips) if c["actor"] == a]
    m = LDAClassifier().fit(X[tr], [y[i] for i in tr], 0.3)
    for i in te:
        hits += m.predict_vector(X[i]).label == y[i]
        n += 1
    # valência binária: alegria × negativas
    trv = [i for i in tr if y[i] == "alegria" or y[i] in neg]
    tev = [i for i in te if y[i] == "alegria" or y[i] in neg]
    mv = LDAClassifier().fit(X[trv], ["pos" if y[i] == "alegria" else "neg" for i in trv], 0.3)
    for i in tev:
        hits_v += mv.predict_vector(X[i]).label == ("pos" if y[i] == "alegria" else "neg")
        n_v += 1
print(f"7 classes: {100 * hits / n:.1f}%  (acaso uniforme {100 / 7:.1f}%, majoritária {100 * maj:.1f}%)")
nv_pos = sum(1 for v in y if v == "alegria")
nv_neg = sum(1 for v in y if v in neg)
print(f"valência alegria × negativas: {100 * hits_v / n_v:.1f}%  (classe majoritária {100 * max(nv_pos, nv_neg) / (nv_pos + nv_neg):.1f}%)")
