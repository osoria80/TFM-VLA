import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


def cargar_fotograma(ruta_video, indice_fotograma):
    """Lee un fotograma concreto y lo devuelve como imagen RGB de PIL."""
    captura = cv2.VideoCapture(str(ruta_video))
    captura.set(cv2.CAP_PROP_POS_FRAMES, int(indice_fotograma))
    correcto, imagen_bgr = captura.read()
    captura.release()

    if not correcto:
        raise RuntimeError(f"No se pudo leer el fotograma {indice_fotograma} de {ruta_video}")

    return Image.fromarray(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))


def mapear_indices_globales(indices_globales, directorio_videos):
    """Mapea indices globales del dataset a un MP4 fisico y a su frame local."""
    rutas_video = sorted(Path(directorio_videos).rglob("*.mp4"))
    if not rutas_video:
        raise FileNotFoundError(f"No se encontraron videos en {directorio_videos}")

    total_por_video = []
    for ruta_video in rutas_video:
        captura = cv2.VideoCapture(str(ruta_video))
        total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        captura.release()
        if total_frames <= 0:
            raise RuntimeError(f"No se pudieron contar los frames de {ruta_video}")
        total_por_video.append(total_frames)

    indices = np.asarray(indices_globales, dtype=np.int64)
    limites = np.cumsum(total_por_video)
    if indices.size and (indices.min() < 0 or indices.max() >= limites[-1]):
        raise IndexError(
            f"Los indices globales deben estar entre 0 y {limites[-1] - 1}"
        )

    indice_video = np.searchsorted(limites, indices, side="right")
    inicio_video = np.concatenate(([0], limites[:-1]))
    indice_local = indices - inicio_video[indice_video]
    rutas = [str(rutas_video[i]) for i in indice_video]

    return rutas, indice_local


class VLADataset(Dataset):
    def __init__(self, tabla, preprocess, tokenizer):
        self.tabla = tabla.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tabla)

    def __getitem__(self, indice):
        muestra = self.tabla.iloc[indice]

        # La imagen se decodifica bajo demanda para no cargar todos los vídeos en memoria.
        imagen = cargar_fotograma(muestra["imagen"], muestra["fotograma_video"])
        imagen = self.preprocess(imagen)
        tokens = self.tokenizer([muestra["instruccion"]])[0]
        accion = torch.tensor(muestra["accion"], dtype=torch.float32)

        return imagen, tokens, accion
