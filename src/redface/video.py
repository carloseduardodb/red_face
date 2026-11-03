"""Leitura de vídeo.

Por padrão usa o ``ffmpeg`` (se estiver no PATH): ele decodifica praticamente
qualquer formato e não derruba o processo. O OpenCV fica como alternativa — mas
atenção: algumas versões do OpenCV causam *segmentation fault* ao abrir AVIs
grandes sem compressão (ex.: UBFC-rPPG), o que não pode ser capturado em Python.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    codec: str
    backend: str


class VideoReader:
    """Itera ``(índice, quadro_bgr)``; use ``info`` para fps/tamanho/codec."""

    def __init__(self, path: str, backend: str = "auto"):
        self.path = str(path)
        if backend == "auto":
            backend = "ffmpeg" if shutil.which("ffmpeg") and shutil.which("ffprobe") else "opencv"
        self.backend = backend
        self.info = self._probe_ffmpeg() if backend == "ffmpeg" else self._probe_opencv()

    # ----------------------------------------------------------------- ffmpeg
    def _probe_ffmpeg(self) -> VideoInfo:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height,avg_frame_rate,r_frame_rate", "-of", "json", self.path],
            capture_output=True, text=True, check=False,
        )
        streams = json.loads(out.stdout or "{}").get("streams", [])
        if not streams:
            raise OSError(f"não consegui abrir {self.path}: {out.stderr.strip()[:200]}")
        s = streams[0]

        def rate(txt: str) -> float:
            num, _, den = txt.partition("/")
            return float(num) / float(den or 1) if float(den or 1) else 0.0

        fps = rate(s.get("avg_frame_rate", "0/1")) or rate(s.get("r_frame_rate", "0/1")) or 30.0
        return VideoInfo(fps, int(s["width"]), int(s["height"]), s.get("codec_name", "?").lower(), "ffmpeg")

    def _iter_ffmpeg(self) -> Iterator[tuple[int, np.ndarray]]:
        w, h = self.info.width, self.info.height
        size = w * h * 3
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", self.path, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE,
        )
        try:
            i = 0
            while True:
                buf = bytearray(size)  # bytearray → array NumPy gravável
                view, got = memoryview(buf), 0
                while got < size:
                    n = proc.stdout.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got < size:
                    break
                yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                i += 1
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()

    # ----------------------------------------------------------------- opencv
    def _probe_opencv(self) -> VideoInfo:
        import cv2

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise OSError(f"não consegui abrir {self.path}")
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((code >> 8 * k) & 0xFF) for k in range(4)).strip("\x00 ").lower()
        info = VideoInfo(cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(3)), int(cap.get(4)), codec, "opencv")
        cap.release()
        return info

    def _iter_opencv(self) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(self.path)
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
                i += 1
        finally:
            cap.release()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        return self._iter_ffmpeg() if self.backend == "ffmpeg" else self._iter_opencv()
