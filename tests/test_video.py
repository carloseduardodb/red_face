import shutil

import cv2
import numpy as np
import pytest

from redface.video import VideoReader


@pytest.mark.parametrize("backend", ["opencv", "ffmpeg"])
def test_reader_backends_agree(tmp_path, backend):
    if backend == "ffmpeg" and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg não instalado")
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (64, 48))
    for k in range(10):
        w.write(np.full((48, 64, 3), (10 * k, 100, 200), np.uint8))
    w.release()
    r = VideoReader(str(path), backend)
    frames = list(r)
    assert r.info.fps == pytest.approx(25, abs=0.5)
    assert (r.info.width, r.info.height) == (64, 48)
    assert len(frames) == 10
    assert frames[3][1][0, 0].tolist() == [30, 100, 200]
    frames[0][1][0, 0] = 0  # quadros precisam ser graváveis (para desenhar overlay)
