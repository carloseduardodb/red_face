"""Compara 5 regiões grandes × malha densa de triângulos (como o artigo da PNAS).

Uso: python scripts/compare_mesh.py PASTA_RAVDESS
(precisa de ``mediapipe`` e do modelo face_landmarker.task em ~/.cache/redface;
rode antes ``scripts/eval_ravdess.py`` para gerar o features.json das 5 regiões).

Malha: subconjunto dos 468 pontos do MediaPipe Face Mesh → triangulação de
Delaunay (o mesmo método do artigo) → descarta triângulos sobre olhos,
sobrancelhas, boca e fora do contorno do rosto → ~126 triângulos de pele.
Por triângulo: média e variância de a* e b* (4 atributos, como no artigo).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from matplotlib.path import Path as MplPath
from mediapipe.tasks.python import BaseOptions, vision
from scipy.spatial import Delaunay

from redface.color import skin_mask
from redface.emotion import FULL_FEATURES, LDAClassifier, full_features
from redface.video import VideoReader

ROOT = Path(sys.argv[1])
MODEL = Path.home() / ".cache" / "redface" / "face_landmarker.task"
CACHE = ROOT / "mesh_features.npz"
EMO = {"01": "neutro", "02": "neutro", "03": "alegria", "04": "tristeza", "05": "raiva", "06": "medo", "07": "nojo", "08": "surpresa"}
TARGET_TRIANGLES = 126
FRAME_STEP = 2

C = vision.FaceLandmarksConnections


def ids(conns) -> list[int]:
    return sorted({c.start for c in conns} | {c.end for c in conns})


def ordered_loop(conns) -> list[int]:
    """Transforma um conjunto de arestas num polígono ordenado."""
    nxt = {}
    for c in conns:
        nxt.setdefault(c.start, []).append(c.end)
        nxt.setdefault(c.end, []).append(c.start)
    start = conns[0].start
    loop, prev, cur = [start], None, start
    while True:
        options = [n for n in nxt[cur] if n != prev]
        if not options or options[0] == start:
            break
        prev, cur = cur, options[0]
        loop.append(cur)
        if len(loop) > len(nxt):
            break
    return loop


def make_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE, num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(opts)


def landmarks_px(lmk, frame) -> np.ndarray | None:
    res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[:, :, ::-1])))
    if not res.face_landmarks:
        return None
    h, w = frame.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])


# ------------------------------------------------------------ triangulação
def build_triangles(template: np.ndarray) -> np.ndarray:
    oval = ordered_loop(C.FACE_LANDMARKS_FACE_OVAL)
    excluded_polys = [ids(C.FACE_LANDMARKS_LEFT_EYE), ids(C.FACE_LANDMARKS_RIGHT_EYE), ids(C.FACE_LANDMARKS_LIPS),
                      ids(C.FACE_LANDMARKS_LEFT_EYEBROW), ids(C.FACE_LANDMARKS_RIGHT_EYEBROW)]
    hulls = []
    for poly in excluded_polys:
        pts = template[poly].astype(np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        center = hull.mean(axis=0)
        hulls.append(MplPath(center + (hull - center) * 1.25))  # margem de 25 %
    oval_path = MplPath(template[oval])
    keypoints = sorted(set(oval) | set().union(*map(set, excluded_polys)))

    def triangulate(n_extra: int) -> np.ndarray:
        # amostragem do ponto mais distante: espalha os vértices pelo rosto
        chosen = list(keypoints)
        dist = np.min(np.linalg.norm(template[:, None] - template[chosen][None], axis=2), axis=1)
        for _ in range(n_extra):
            k = int(np.argmax(dist))
            chosen.append(k)
            dist = np.minimum(dist, np.linalg.norm(template - template[k], axis=1))
        chosen = np.array(sorted(set(chosen)))
        tri = chosen[Delaunay(template[chosen]).simplices]
        keep = []
        for t in tri:
            c = template[t].mean(axis=0)
            if oval_path.contains_point(c) and not any(h.contains_point(c) for h in hulls):
                keep.append(t)
        return np.array(keep)

    best = None
    for n_extra in range(10, 200, 2):
        t = triangulate(n_extra)
        if best is None or abs(len(t) - TARGET_TRIANGLES) < abs(len(best) - TARGET_TRIANGLES):
            best = t
        if len(t) >= TARGET_TRIANGLES:
            break
    return best


# --------------------------------------------------------------- medição
def clip_mesh_stats(path: Path, lmk, tris: np.ndarray):
    T = len(tris)
    s_mean = np.zeros((T, 2))
    s_var = np.zeros((T, 2))
    count = np.zeros(T)
    n_frames = n_ok = 0
    for i, frame in VideoReader(str(path)):
        if i % FRAME_STEP:
            continue
        n_frames += 1
        pts = landmarks_px(lmk, frame)
        if pts is None:
            continue
        x0, y0 = np.floor(pts.min(axis=0)).astype(int).clip(0)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
        crop = frame[y0:y1, x0:x1]
        labels = np.zeros(crop.shape[:2], np.int32)
        for k, t in enumerate(tris):
            cv2.fillConvexPoly(labels, np.round(pts[t] - [x0, y0]).astype(np.int32), k + 1)
        labels[~skin_mask(crop)] = 0
        lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        flat = labels.ravel()
        n = np.bincount(flat, minlength=T + 1)[1:]
        stats = []
        for ch in (1, 2):
            v = lab[..., ch].ravel()
            sm = np.bincount(flat, v, minlength=T + 1)[1:]
            sq = np.bincount(flat, v * v, minlength=T + 1)[1:]
            with np.errstate(invalid="ignore", divide="ignore"):
                m = sm / n
                stats.append((m, sq / n - m * m))
        good = n >= 10
        s_mean[good] += np.stack([stats[0][0], stats[1][0]], 1)[good]
        s_var[good] += np.stack([stats[0][1], stats[1][1]], 1)[good]
        count[good] += 1
        n_ok += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return s_mean / count[:, None], s_var / count[:, None], n_ok, n_frames


def main() -> None:
    files = sorted(ROOT.glob("Actor_*/02-01-*.mp4"))
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        tris, means, vars_, meta = data["tris"], data["means"], data["vars"], list(data["meta"])
    else:
        lmk = make_landmarker()
        first = next(f for f in files if f.name.startswith("02-01-01"))
        template = next(p for p in (landmarks_px(lmk, fr) for _, fr in VideoReader(str(first))) if p is not None)
        tris = build_triangles(template)
        print(f"malha: {len(tris)} triângulos de pele", flush=True)
        means, vars_, meta = [], [], []
        for k, f in enumerate(files):
            m, v, n_ok, n = clip_mesh_stats(f, lmk, tris)
            parts = f.stem.split("-")
            means.append(m)
            vars_.append(v)
            meta.append({"file": f.name, "actor": parts[6], "emo": EMO[parts[2]], "ok": n_ok, "n": n})
            print(f"{k + 1}/{len(files)} {f.name} {n_ok}/{n}", flush=True)
        means, vars_ = np.array(means), np.array(vars_)
        np.savez(CACHE, tris=tris, means=means, vars=vars_, meta=np.array(meta, dtype=object))
        # figura da malha num rosto
        fr = next(fr for i, fr in VideoReader(str(first)) if i == 20)
        pts = landmarks_px(lmk, fr)
        for t in tris:
            cv2.polylines(fr, [np.round(pts[t]).astype(np.int32)], True, (60, 220, 255), 1, cv2.LINE_AA)
        x0, y0 = pts.min(axis=0).astype(int) - 40
        x1, y1 = pts.max(axis=0).astype(int) + 40
        cv2.imwrite(str(ROOT / "malha.png"), fr[max(0, y0):y1, max(0, x0):x1])

    # ------------------------------------------------------------ avaliação
    T = len(tris)
    reg = {c["file"]: c for c in json.loads((ROOT / "features.json").read_text())}
    valid = [i for i, m in enumerate(meta) if m["ok"] >= 0.7 * m["n"] and m["file"] in reg]
    meta = [meta[i] for i in valid]
    means, vars_ = means[valid], vars_[valid]
    actors = sorted({m["actor"] for m in meta})

    # deltas vs. clipes neutros (emoção 01) do mesmo ator, deixando o próprio clipe de fora
    def deltas(arr):
        out = np.zeros_like(arr)
        for i, m in enumerate(meta):
            neu = [j for j, mm in enumerate(meta) if mm["actor"] == m["actor"] and mm["file"].startswith("02-01-01") and j != i]
            out[i] = arr[i] - np.nanmean(arr[neu], axis=0)
        return out

    dm, dv = deltas(means), deltas(vars_)
    X_mesh_mean = np.nan_to_num(dm.reshape(len(meta), -1))
    X_mesh_full = np.hstack([X_mesh_mean, np.nan_to_num(dv.reshape(len(meta), -1))])

    # 5 regiões: mesmos atributos de cor do red_face (sem pulso), com e sem sorriso
    def region_vec(m, smile):
        c = reg[m["file"]]
        neu = [reg[mm["file"]] for mm in meta if mm["actor"] == m["actor"] and mm["file"].startswith("02-01-01") and mm["file"] != m["file"]]
        delta = {r: np.array(v) - np.mean([n["regions"][r] for n in neu if r in n["regions"]], axis=0) for r, v in c["regions"].items()}
        bs = np.mean([n["smile"] for n in neu if n.get("smile")]) if smile else None
        ds = (c["smile"] / bs - 1) if (smile and c.get("smile") and bs) else None
        f = full_features(delta, None, ds)
        keys = [k for k in FULL_FEATURES if k != "d_pulso" and (smile or k != "d_sorriso")]
        return [f[k] for k in keys]

    X_reg = np.nan_to_num(np.array([region_vec(m, False) for m in meta]))
    X_reg_s = np.nan_to_num(np.array([region_vec(m, True) for m in meta]))
    y = np.array([m["emo"] for m in meta])
    groups = np.array([m["actor"] for m in meta])
    labels = sorted(set(y))
    maj = max(np.mean(y == lab) for lab in labels)

    def loao(X, shrink):
        hits = 0
        for a in actors:
            tr, te = groups != a, groups == a
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9  # padroniza com o treino
            model = LDAClassifier().fit((X[tr] - mu) / sd, list(y[tr]), shrink)
            hits += sum(model.predict_vector((x - mu) / sd).label == t for x, t in zip(X[te], y[te]))
        return hits / len(y)

    print(f"\n{len(meta)} clipes, {len(actors)} atores, {len(labels)} classes; acaso uniforme {100 / len(labels):.1f}%, majoritária {100 * maj:.1f}%")
    print(f"malha: {T} triângulos\n")
    shrinks = [0.1, 0.3, 0.6, 0.9]
    print(f"{'esquema':<44}{'atrib.':>7}" + "".join(f"{f'δ={s}':>9}" for s in shrinks))
    for name, X in [
        ("5 regiões (Δa*, Δb*, Δeritema)", X_reg),
        ("5 regiões + sorriso", X_reg_s),
        (f"malha {T} triângulos (média a*, b*)", X_mesh_mean),
        (f"malha {T} triângulos (média + variância)", X_mesh_full),
    ]:
        accs = [loao(X, s) for s in shrinks]
        print(f"{name:<44}{X.shape[1]:>7}" + "".join(f"{100 * a:>8.1f}%" for a in accs))


if __name__ == "__main__":
    main()
