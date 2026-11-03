"""Linha de comando: ``redface analyze | live | demo | train``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .emotion import FULL_FEATURES, LDAClassifier, PrototypeClassifier
from .i18n import label, set_lang, tr
from .pipeline import REASONS, AnalyzerConfig, FaceColorAnalyzer, WindowResult
from .regions import FixedBox, default_tracker

LOSSY_CODECS = {"avc1", "h264", "x264", "hev1", "hvc1", "hevc", "mp4v", "fmp4", "xvid", "divx", "vp80", "vp90", "av01", "mjpg"}


# --------------------------------------------------------------- utilidades
def _fourcc(cap: cv2.VideoCapture) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower()


def _parse_box(text: str | None):
    if not text:
        return None
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 4:
        raise SystemExit(tr("e_box"))
    return tuple(parts)


def _build(args, fps: float) -> FaceColorAnalyzer:
    box = _parse_box(getattr(args, "box", None))
    if getattr(args, "regions", "cinco") == "malha":
        if box:
            raise SystemExit(tr("e_mesh_box"))
        from .mesh import MeshFaceTracker

        tracker = MeshFaceTracker()
    else:
        tracker = FixedBox(box) if box else default_tracker(getattr(args, "detector", "auto"))
    classifier = LDAClassifier.load(args.model) if getattr(args, "model", None) else PrototypeClassifier()
    cfg = AnalyzerConfig(baseline_s=args.baseline, window_s=args.window)
    return FaceColorAnalyzer(fps, cfg, tracker, classifier)


def _write_csv(results: list[WindowResult], path: Path, label: str | None = None, source: str = "") -> None:
    rows = [r.to_row() for r in results]
    if label is not None:
        for row in rows:
            row["rotulo"] = label
            row["fonte"] = source
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summary(results: list[WindowResult], analyzer: FaceColorAnalyzer, extra: dict) -> dict:
    status = Counter(r.status for r in results)
    emotions = Counter(r.emotion for r in results if r.status == "ok")
    reasons = Counter(m for r in results for m in r.reasons)
    hrs = [r.hr_bpm for r in results if r.hr_bpm is not None]
    b = analyzer.baseline
    return {
        **extra,
        "classificador": analyzer.classifier.name,
        "janelas": dict(status),
        "fracao_respondida": round(status.get("ok", 0) / max(1, len(results)), 3),
        "emocoes": dict(emotions.most_common()),
        "motivos_indeterminado": {k: {"n": v, "explicacao": REASONS.get(k, "")} for k, v in reasons.items()},
        "pulso_medio_bpm": round(float(np.mean(hrs)), 1) if hrs else None,
        "linha_de_base": None
        if b is None
        else {
            "quadros": b.n_frames,
            "pulso_bpm": None if b.hr_bpm is None else round(b.hr_bpm, 1),
            "regioes_Lab_ei": {k: [round(float(x), 2) for x in v] for k, v in b.regions.items()},
        },
        "aviso": "Hipóteses estatísticas a partir da cor do rosto. Não é diagnóstico, não mede emoção sentida e não detecta mentira.",
    }


def _print_window(r: WindowResult) -> None:
    if r.status == "calibrando":
        msg = tr("calibrating_line")
    elif r.status == "indeterminado":
        msg = label("indeterminado") + " (" + ", ".join(label(x) for x in r.reasons) + ")"
    else:
        p = r.prediction
        msg = f"{label(p.label):<9} {100 * p.confidence:4.0f}%  {tr('valence_cli')} {p.valence:+.2f}"
    extra = ""
    if r.summary:
        extra = f"  Δa* {r.summary['d_vermelho']:+.2f}  Δb* {r.summary['d_amarelo']:+.2f}"
    if r.hr_bpm is not None:
        extra += f"  {tr('pulse')} {r.hr_bpm:5.1f} bpm"
    print(f"[{r.t_start:6.1f}–{r.t_end:6.1f}s] {msg}{extra}")


# ------------------------------------------------------------------ comandos
def cmd_analyze(args) -> int:
    from .video import VideoReader

    try:
        reader = VideoReader(args.video, args.reader)
    except OSError as exc:
        print(exc, file=sys.stderr)
        return 1
    fps, codec = reader.info.fps, reader.info.codec
    warnings = []
    lossy = {"h264", "hevc", "mpeg4", "vp8", "vp9", "av1", "mjpeg", "mpeg2video", "wmv2", "wmv3"} | LOSSY_CODECS
    if codec in lossy:
        warnings.append(tr("w_lossy", codec=codec))
    if fps < 20:
        warnings.append(tr("w_low_fps", fps=f"{fps:.1f}"))
    for w in warnings:
        print(f"{tr('warning')}:", w, file=sys.stderr)

    analyzer = _build(args, fps)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    results: list[WindowResult] = []
    i = -1
    for i, frame in reader:
        t = i / fps
        new = analyzer.process(frame, t)
        results += new
        if not args.quiet:
            for r in new:
                _print_window(r)
        if args.annotated:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "anotado.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            from .viz import draw_overlay

            writer.write(draw_overlay(frame, analyzer))
    i += 1  # número de quadros lidos
    if writer is not None:
        writer.release()

    _write_csv(results, out_dir / "janelas.csv", args.label, Path(args.video).name)
    summary = _summary(
        results,
        analyzer,
        {"video": str(args.video), "fps": fps, "quadros": i, "duracao_s": round(i / fps, 2), "codec": codec, "avisos": warnings},
    )
    (out_dir / "resumo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.plot:
        from .viz import plot_timeline

        plot_timeline(results, out_dir / "linha_do_tempo.png", title=f"red_face - {Path(args.video).name}")
    print(f"\n{ {label(k): v for k, v in summary['janelas'].items()} }  →  {tr('results_in')} {out_dir}/")
    return 0


def _lock_camera(cap: cv2.VideoCapture) -> None:
    """Tenta travar balanço de branco e exposição (funciona em muitas webcams V4L2)."""
    for _ in range(45):  # deixa a câmera se ajustar à luz primeiro
        cap.read()
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual, 3 = automático
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)


def cmd_live(args) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(tr("e_camera", cam=args.camera), file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not args.no_lock:
        _lock_camera(cap)
    from .viz import draw_overlay

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analyzer = _build(args, fps)
    t0 = time.monotonic()
    print(tr("live_start"))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for r in analyzer.process(frame, time.monotonic() - t0):
            _print_window(r)
        cv2.imshow("red_face", draw_overlay(frame, analyzer))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def cmd_demo(args) -> int:
    from .synth import DEFAULT_SCENARIO, face_geometry, segment_at, synthetic_frames
    from .viz import draw_overlay, plot_timeline

    fps, size = 30.0, (640, 480)
    _, _, box = face_geometry(size)
    analyzer = FaceColorAnalyzer(fps, AnalyzerConfig(baseline_s=args.baseline, window_s=args.window), FixedBox(box))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out_dir / "demo.avi"), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    results: list[WindowResult] = []
    snapshots = {12.0: "quadro_calibrando.png", 27.0: "quadro_raiva.png", 54.0: "quadro_medo.png", 68.0: "quadro_luz.png"}
    for t, frame, _ in synthetic_frames(DEFAULT_SCENARIO, fps, size):
        new = analyzer.process(frame, t)
        results += new
        for r in new:
            _print_window(r)
        overlay = draw_overlay(frame, analyzer) if (writer or args.snapshots) else None
        if writer is not None:
            writer.write(frame)
        if args.snapshots:
            for ts, name in snapshots.items():
                if abs(t - ts) < 0.5 / fps:
                    cv2.imwrite(str(out_dir / name), overlay)
    if writer is not None:
        writer.release()

    truth = [(s.t0, s.t1, s.label) for s in DEFAULT_SCENARIO]
    _write_csv(results, out_dir / "janelas.csv")
    plot_timeline(results, out_dir / "linha_do_tempo.png", truth=truth, title=tr("plot_demo_title"))

    # avaliação contra a verdade (só janelas inteiramente dentro de um segmento estável)
    hits = total = 0
    for r in results:
        seg = segment_at(DEFAULT_SCENARIO, r.t_end)
        settled = r.t_start >= seg.t0 + 4 and r.t_end <= seg.t1
        if r.status == "ok" and settled and seg.label != "luz":
            total += 1
            hits += r.emotion == seg.label
    light_segs = [s for s in DEFAULT_SCENARIO if s.exposure != 1.0]
    light = [r for r in results if any(r.t_start < s.t1 and r.t_end > s.t0 for s in light_segs)]
    refused = sum(r.status == "indeterminado" for r in light)
    print(f"\n{tr('demo_hits')}: {hits}/{total}")
    print(f"{tr('demo_refused')}: {refused}/{len(light)}")
    print(f"{tr('files_in')} {out_dir}/")
    return 0


def cmd_train(args) -> int:
    rows = []
    for path in args.csv:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok" and row.get("rotulo"):
                    row.setdefault("fonte", path)
                    rows.append(row)
    if not rows:
        print(tr("e_no_rows"), file=sys.stderr)
        return 1

    header = set(rows[0])
    names = [k for k in FULL_FEATURES if k in header]
    mesh_cols = sorted(k for k in header if k.startswith("malha_"))
    if mesh_cols and not args.no_mesh:
        names += mesh_cols
        print(tr("t_mesh", n=len(mesh_cols)))

    def vec(row):
        return [float(row[k]) if row.get(k) not in (None, "") else np.nan for k in names]

    X = np.array([vec(r) for r in rows])
    y = [r["rotulo"] for r in rows]
    groups = [r.get("fonte", "") for r in rows]
    labels = sorted(set(y))
    print(tr("t_rows", n=len(rows), v=len(set(groups)), labels=dict(Counter(y))))

    # validação deixando um vídeo de fora (evita vazamento de identidade/sessão)
    uniq = sorted(set(groups))
    if len(uniq) >= 3:
        hits = n = 0
        for g in uniq:
            tr_idx = [i for i, gg in enumerate(groups) if gg != g]
            te = [i for i, gg in enumerate(groups) if gg == g]
            if len({y[i] for i in tr_idx}) < 2:
                continue
            m = LDAClassifier().fit(X[tr_idx], [y[i] for i in tr_idx], args.shrinkage)
            m.features = tuple(names)
            hits += sum(m.predict_vector(X[i]).label == y[i] for i in te)
            n += len(te)
        if n:
            print(tr("t_acc", acc=f"{100 * hits / n:.1f}", chance=f"{100 / len(labels):.1f}"))
    else:
        print(tr("t_few"))

    model = LDAClassifier().fit(X, y, args.shrinkage)
    model.features = tuple(names)
    model.save(args.output)
    print(tr("t_saved", path=args.output))
    return 0


# ---------------------------------------------------------------------- main
def _pre_lang(argv: list[str]) -> None:
    """Aplica --lang antes de montar o parser, para a própria ajuda sair no idioma certo."""
    for k, arg in enumerate(argv):
        if arg == "--lang" and k + 1 < len(argv):
            set_lang(argv[k + 1])
        elif arg.startswith("--lang="):
            set_lang(arg.split("=", 1)[1])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    _pre_lang(argv)
    p = argparse.ArgumentParser(prog="redface", description=tr("h_prog"))
    p.add_argument("--version", action="version", version=f"red_face {__version__}", help=tr("h_version"))
    p.add_argument("--lang", choices=["auto", "en", "pt"], default="auto", help=tr("h_lang"))
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--baseline", type=float, default=10.0, help=tr("h_baseline"))
        sp.add_argument("--window", type=float, default=4.0, help=tr("h_window"))

    a = sub.add_parser("analyze", help=tr("h_analyze"), description=tr("h_analyze"))
    a.add_argument("video", help=tr("h_video"))
    a.add_argument("-o", "--output", default="saida", help=tr("h_output"))
    a.add_argument("--model", help=tr("h_model"))
    a.add_argument("--box", help=tr("h_box"))
    a.add_argument("--label", help=tr("h_label"))
    a.add_argument("--annotated", action="store_true", help=tr("h_annotated"))
    a.add_argument("--plot", action="store_true", help=tr("h_plot"))
    a.add_argument("-q", "--quiet", action="store_true", help=tr("h_quiet"))
    a.add_argument("--detector", choices=["auto", "yunet", "haar"], default="auto", help=tr("h_detector"))
    a.add_argument("--regions", choices=["cinco", "malha"], default="cinco", help=tr("h_regions"))
    a.add_argument("--reader", choices=["auto", "ffmpeg", "opencv"], default="auto", help=tr("h_reader"))
    common(a)
    a.set_defaults(func=cmd_analyze)

    lv = sub.add_parser("live", help=tr("h_live"), description=tr("h_live"))
    lv.add_argument("--camera", type=int, default=0, help=tr("h_camera"))
    lv.add_argument("--model", help=tr("h_model"))
    lv.add_argument("--box", help=tr("h_box"))
    lv.add_argument("--detector", choices=["auto", "yunet", "haar"], default="auto", help=tr("h_detector"))
    lv.add_argument("--regions", choices=["cinco", "malha"], default="cinco", help=tr("h_regions"))
    lv.add_argument("--no-lock", action="store_true", help=tr("h_no_lock"))
    common(lv)
    lv.set_defaults(func=cmd_live)

    d = sub.add_parser("demo", help=tr("h_demo"), description=tr("h_demo"))
    d.add_argument("-o", "--output", default="demo", help=tr("h_demo_output"))
    d.add_argument("--save-video", action="store_true", help=tr("h_save_video"))
    d.add_argument("--snapshots", action="store_true", help=tr("h_snapshots"))
    common(d)
    d.set_defaults(func=cmd_demo)

    trn = sub.add_parser("train", help=tr("h_train"), description=tr("h_train"))
    trn.add_argument("csv", nargs="+", help=tr("h_csv"))
    trn.add_argument("-o", "--output", default="modelo.json", help=tr("h_train_output"))
    trn.add_argument("--shrinkage", type=float, default=0.3, help=tr("h_shrinkage"))
    trn.add_argument("--no-mesh", action="store_true", help=tr("h_no_mesh"))
    trn.set_defaults(func=cmd_train)

    args = p.parse_args(argv)
    set_lang(args.lang)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
