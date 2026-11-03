"""Idioma das mensagens exibidas (terminal, overlay, gráficos).

Os dados (rótulos internos, colunas do CSV, chaves do JSON) ficam sempre iguais;
só a exibição muda. Padrão: variável ``REDFACE_LANG`` ou o idioma do sistema
(``pt*`` → português; qualquer outro → inglês).
"""

from __future__ import annotations

import locale
import os

_LANG: str | None = None

LABELS_EN = {
    # emoções
    "neutro": "neutral",
    "alegria": "happiness",
    "surpresa": "surprise",
    "raiva": "anger",
    "vergonha": "embarrassment",
    "medo": "fear",
    "tristeza": "sadness",
    "nojo": "disgust",
    "calmo": "calm",
    "luz": "light jump",
    # status
    "calibrando": "calibrating",
    "ok": "ok",
    "indeterminado": "undetermined",
    # motivos
    "rosto_nao_detectado": "no_face",
    "movimento": "motion",
    "cena_mudou": "scene_changed",
    "iluminacao_mudou": "lighting_changed",
    "sem_linha_de_base": "no_baseline",
}

TEXT = {
    "calibrating_banner": ("CALIBRANDO - fique neutro ({s}s)", "CALIBRATING - keep a neutral face ({s}s)"),
    "waiting": ("aguardando janela...", "waiting for first window..."),
    "valence": ("valencia", "valence"),
    "pulse": ("pulso", "pulse"),
    "disclaimer_overlay": ("hipotese pela cor do rosto - nao e diagnostico", "hypothesis from facial color - not a diagnosis"),
    "calibrating_line": ("calibrando linha de base…", "calibrating baseline…"),
    "valence_cli": ("valência", "valence"),
    "plot_change": ("mudança vs.\nlinha de base", "change vs.\nbaseline"),
    "plot_da": ("Δa* (vermelho ↔ verde)", "Δa* (red ↔ green)"),
    "plot_db": ("Δb* (amarelo ↔ azul)", "Δb* (yellow ↔ blue)"),
    "plot_pulse": ("pulso (bpm)", "pulse (bpm)"),
    "plot_estimated": ("estimado", "estimated"),
    "plot_truth": ("verdade", "ground truth"),
    "plot_time": ("tempo (s)", "time (s)"),
    "plot_title": ("red_face - linha do tempo", "red_face - timeline"),
    "plot_demo_title": ("red_face - demonstração sintética", "red_face - synthetic demo"),
    "demo_hits": ("acerto nas janelas estáveis", "correct on settled windows"),
    "demo_refused": ("janelas com salto de luz recusadas", "light-jump windows refused"),
    "files_in": ("arquivos em", "files in"),
    "results_in": ("resultados em", "results in"),
    # ajuda do CLI
    "h_prog": ("Hipóteses de emoção a partir da cor do rosto em vídeo.", "Emotion hypotheses from facial color in video."),
    "h_lang": ("idioma das mensagens, ajuda e gráficos (padrão: idioma do sistema; os dados não mudam)",
               "language of messages, help and charts (default: system language; data keys never change)"),
    "h_version": ("mostra a versão e sai", "show the version and exit"),
    "h_baseline": ("segundos iniciais de calibração neutra (padrão 10)", "initial seconds of neutral calibration (default 10)"),
    "h_window": ("tamanho da janela de análise em segundos (padrão 4)", "analysis window length in seconds (default 4)"),
    "h_analyze": ("analisa um arquivo de vídeo", "analyze a video file"),
    "h_video": ("arquivo de vídeo", "video file"),
    "h_output": ("pasta de saída (padrão: saida/)", "output folder (default: saida/)"),
    "h_model": ("modelo LDA treinado (JSON); sem isso usa os protótipos da literatura",
                "trained LDA model (JSON); without it, the literature prototypes are used"),
    "h_box": ("caixa fixa do rosto x,y,largura,altura (pula a detecção)", "fixed face box x,y,width,height (skips detection)"),
    "h_label": ("rótulo verdadeiro deste vídeo, para gerar dados de treino", "true label of this video, to produce training data"),
    "h_annotated": ("salva anotado.avi com o overlay", "save anotado.avi with the overlay"),
    "h_plot": ("salva linha_do_tempo.png (requer matplotlib)", "save linha_do_tempo.png, the timeline chart (needs matplotlib)"),
    "h_quiet": ("não imprime cada janela", "don't print every window"),
    "h_detector": ("detector de rosto (padrão: YuNet)", "face detector (default: YuNet)"),
    "h_regions": ("5 regiões (padrão) ou malha de 133 triângulos como no artigo da PNAS (requer red-face[malha])",
                  "5 regions (default) or a 133-triangle mesh as in the PNAS paper (needs red-face[malha])"),
    "h_reader": ("leitor de vídeo (padrão: ffmpeg se instalado)", "video reader (default: ffmpeg if installed)"),
    "h_live": ("análise ao vivo pela webcam", "live analysis from the webcam"),
    "h_camera": ("índice da câmera (padrão 0)", "camera index (default 0)"),
    "h_no_lock": ("não tenta travar balanço de branco/exposição", "don't try to lock white balance/exposure"),
    "h_demo": ("roda o pipeline num vídeo sintético com verdade conhecida", "run the pipeline on a synthetic video with known ground truth"),
    "h_demo_output": ("pasta de saída (padrão: demo/)", "output folder (default: demo/)"),
    "h_save_video": ("salva o vídeo sintético (FFV1, sem perdas)", "save the synthetic video (FFV1, lossless)"),
    "h_snapshots": ("salva quadros com overlay", "save frames with the overlay"),
    "h_train": ("treina uma LDA com CSVs rotulados", "train an LDA on labeled CSVs"),
    "h_csv": ("arquivos janelas.csv gerados com --label", "janelas.csv files produced with --label"),
    "h_train_output": ("arquivo do modelo (padrão: modelo.json)", "model file (default: modelo.json)"),
    "h_shrinkage": ("regularização da LDA, de 0 a 1 (padrão 0.3)", "LDA shrinkage, 0 to 1 (default 0.3)"),
    "h_no_mesh": ("ignora colunas da malha mesmo se existirem no CSV", "ignore mesh columns even if present in the CSV"),
    # mensagens do CLI
    "e_box": ("--box precisa de 4 números: x,y,largura,altura", "--box needs 4 numbers: x,y,width,height"),
    "e_mesh_box": ("--regions malha precisa detectar o rosto; não combina com --box", "--regions malha needs face detection; it can't be combined with --box"),
    "w_lossy": ("vídeo com compressão com perdas ({codec}): mudanças sutis de cor podem ter sido apagadas (McDuff et al. 2017). Prefira FFV1/PNG ou bitrate alto.",
                "lossy-compressed video ({codec}): subtle color changes may have been erased (McDuff et al. 2017). Prefer FFV1/PNG or a high bitrate."),
    "w_low_fps": ("fps baixo ({fps}); o pulso fica pouco confiável abaixo de ~20 fps", "low fps ({fps}); pulse is unreliable below ~20 fps"),
    "warning": ("aviso", "warning"),
    "e_camera": ("não consegui abrir a câmera {cam}", "could not open camera {cam}"),
    "live_start": ("Olhe para a câmera com expressão neutra durante a calibração. Tecla q para sair.",
                   "Look at the camera with a neutral face during calibration. Press q to quit."),
    "e_no_rows": ("nenhuma janela 'ok' com rótulo. Gere os CSVs com: redface analyze VIDEO --label ROTULO",
                  "no labeled 'ok' windows. Produce the CSVs with: redface analyze VIDEO --label LABEL"),
    "t_mesh": ("usando a malha: {n} atributos de triângulos", "using the mesh: {n} triangle features"),
    "t_rows": ("{n} janelas, {v} vídeos, rótulos: {labels}", "{n} windows, {v} videos, labels: {labels}"),
    "t_acc": ("acurácia deixando um vídeo de fora: {acc}%  (acaso ≈ {chance}%)", "leave-one-video-out accuracy: {acc}%  (chance ≈ {chance}%)"),
    "t_few": ("aviso: com menos de 3 vídeos não dá para estimar acurácia sem vazamento",
              "warning: with fewer than 3 videos, accuracy can't be estimated without leakage"),
    "t_saved": ("modelo salvo em {path}", "model saved to {path}"),
}


def lang() -> str:
    global _LANG
    if _LANG is None:
        env = os.environ.get("REDFACE_LANG")
        if env:
            _LANG = "pt" if env.lower().startswith("pt") else "en"
        else:
            loc = os.environ.get("LC_ALL") or os.environ.get("LANG") or (locale.getlocale()[0] or "")
            _LANG = "pt" if loc.lower().startswith("pt") else "en"
    return _LANG


def set_lang(value: str | None) -> None:
    global _LANG
    _LANG = None if value in (None, "auto") else ("pt" if value.startswith("pt") else "en")


def label(name: str | None) -> str:
    """Rótulo interno (emoção, status, motivo) no idioma de exibição."""
    if name is None:
        return ""
    return name if lang() == "pt" else LABELS_EN.get(name, name)


def tr(key: str, **kw: object) -> str:
    pt, en = TEXT[key]
    return (pt if lang() == "pt" else en).format(**kw)


t = tr  # nome curto
