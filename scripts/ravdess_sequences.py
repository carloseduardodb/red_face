"""Avalia o red_face *pelo caminho do produto* (redface analyze/train) na RAVDESS.

Para cada ator monta um vídeo contínuo: 4 clipes neutros (linha de base) seguidos
de todos os clipes emocionais. Roda ``redface analyze`` (5 regiões e/ou malha),
rotula cada janela pela emoção do clipe em que ela cai inteira, e faz validação
deixando um ator de fora com ``redface train``.

Uso: python scripts/ravdess_sequences.py PASTA_RAVDESS PASTA_SAIDA [cinco] [malha]
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODES = sys.argv[3:] or ["cinco", "malha"]
EMO = {"01": "neutro", "02": "neutro", "03": "alegria", "04": "tristeza", "05": "raiva", "06": "medo", "07": "nojo", "08": "surpresa"}
REDFACE = str(Path(sys.executable).parent / "redface")
OUT.mkdir(parents=True, exist_ok=True)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def build_sequence(actor: str) -> tuple[Path, list[tuple[float, float, str]]]:
    d = ROOT / f"Actor_{actor}"
    neutral = sorted(d.glob("02-01-01-*.mp4"))
    others = sorted(f for f in d.glob("02-01-*.mp4") if not f.name.startswith("02-01-01"))
    clips = neutral + others
    video, segs_file = OUT / f"seq_{actor}.mkv", OUT / f"seq_{actor}.json"
    if not video.exists():
        lst = OUT / f"seq_{actor}.txt"
        lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-an",
                        "-r", "30", "-vf", "scale=640:360", "-c:v", "ffv1", str(video)], check=True)
        t, segs = 0.0, []
        for c in clips:
            dur = duration(c)
            segs.append((t, t + dur, EMO[c.name.split("-")[2]]))
            t += dur
        segs_file.write_text(json.dumps(segs))
    return video, [tuple(s) for s in json.loads(segs_file.read_text())]


def label_csv(csv_path: Path, segs, n_baseline: int, dest: Path, actor: str) -> int:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    baseline_end = segs[n_baseline - 1][1]
    keep = []
    for r in rows:
        t0, t1 = float(r["t_inicio"]), float(r["t_fim"])
        if r["status"] != "ok" or t0 < baseline_end:
            continue
        inside = [lab for a, b, lab in segs if a - 0.05 <= t0 and t1 <= b + 0.05]
        if inside:
            r["rotulo"], r["fonte"] = inside[0], f"ator_{actor}"
            keep.append(r)
    if keep:
        with dest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(keep[0]))
            w.writeheader()
            w.writerows(keep)
    return len(keep)


actors = sorted(p.name.split("_")[1] for p in ROOT.glob("Actor_*"))
seqs = {a: build_sequence(a) for a in actors}
n_neutral = 4
results = {}
for mode in MODES:
    labelled = {}
    for a, (video, segs) in seqs.items():
        out = OUT / f"{mode}_{a}"
        if not (out / "janelas.csv").exists():
            subprocess.run([REDFACE, "analyze", str(video), "-o", str(out), "--regions", mode, "--window", "3",
                            "--baseline", "12", "-q"], check=True, stderr=subprocess.DEVNULL)
        n = label_csv(out / "janelas.csv", segs, n_neutral, out / "rotulado.csv", a)
        labelled[a] = out / "rotulado.csv"
        print(f"[{mode}] ator {a}: {n} janelas rotuladas", flush=True)
    hits = total = 0
    per_actor = {}
    for test in actors:
        model = OUT / f"{mode}_modelo_sem_{test}.json"
        train = [str(labelled[a]) for a in actors if a != test]
        subprocess.run([REDFACE, "train", *train, "-o", str(model)], check=True, capture_output=True)
        # aplica o modelo nas janelas do ator de teste (mesmas medidas do analyze)
        from redface.emotion import LDAClassifier

        m = LDAClassifier.load(model)
        rows = list(csv.DictReader(labelled[test].open(encoding="utf-8")))
        ok = 0
        for r in rows:
            feats = {k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items() if k in set(m.features)}
            ok += m.predict_features(feats).label == r["rotulo"]
        per_actor[test] = ok / max(1, len(rows))
        hits += ok
        total += len(rows)
    labels = [r["rotulo"] for a in actors for r in csv.DictReader(labelled[a].open(encoding="utf-8"))]
    maj = max(labels.count(x) for x in set(labels)) / len(labels)
    results[mode] = {"acerto": hits / total, "janelas": total, "por_ator": per_actor, "majoritaria": maj, "classes": len(set(labels))}
    print(f"[{mode}] acerto deixando um ator de fora: {100 * hits / total:.1f}%  ({total} janelas; "
          f"acaso uniforme {100 / len(set(labels)):.1f}%, majoritária {100 * maj:.1f}%)  por ator: "
          + ", ".join(f"{a}: {100 * v:.0f}%" for a, v in per_actor.items()), flush=True)
(OUT / "resultados.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
