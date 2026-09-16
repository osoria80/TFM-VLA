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
    def __init__(
        self, tabla, preprocess, tokenizer, lectura_secuencial=False, tamano_bloque=1024
    ):
        self.tabla = tabla.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.lectura_secuencial = lectura_secuencial
        self.tamano_bloque = tamano_bloque
        self._captura = None
        self._ruta_abierta = None
        self._ultimo_frame = -1
        self._inicio_bloque = None
        self._tokens_por_instruccion = {}

    def __len__(self):
        return len(self.tabla)

    def _cargar_fotograma_secuencial(self, ruta_video, indice_fotograma):
        """Lee frames ordenados sin reiniciar la decodificacion del MP4."""
        ruta_video = str(ruta_video)
        indice_fotograma = int(indice_fotograma)

        nuevo_bloque = (
            self._inicio_bloque is None
            or indice_fotograma < self._inicio_bloque
            or indice_fotograma >= self._inicio_bloque + self.tamano_bloque
        )
        if self._ruta_abierta != ruta_video or indice_fotograma <= self._ultimo_frame or nuevo_bloque:
            if self._captura is not None:
                self._captura.release()
            self._captura = cv2.VideoCapture(ruta_video)
            self._ruta_abierta = ruta_video
            self._inicio_bloque = (indice_fotograma // self.tamano_bloque) * self.tamano_bloque
            self._captura.set(cv2.CAP_PROP_POS_FRAMES, self._inicio_bloque)
            self._ultimo_frame = self._inicio_bloque - 1

        imagen_bgr = None
        while self._ultimo_frame < indice_fotograma:
            correcto, imagen_bgr = self._captura.read()
            self._ultimo_frame += 1
            if not correcto:
                raise RuntimeError(
                    f"No se pudo leer el fotograma {indice_fotograma} de {ruta_video}"
                )

        return Image.fromarray(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))

    def cerrar_video(self):
        """Libera el lector secuencial al terminar la extraccion."""
        if self._captura is not None:
            self._captura.release()
            self._captura = None
        self._ruta_abierta = None
        self._inicio_bloque = None

    def __getitem__(self, indice):
        muestra = self.tabla.iloc[indice]

        # La imagen se decodifica bajo demanda para no cargar todos los vídeos en memoria.
        if self.lectura_secuencial:
            try:
                imagen = self._cargar_fotograma_secuencial(
                    muestra["imagen"], muestra["fotograma_video"]
                )
            except RuntimeError:
                # Reintento aislado si el decodificador falla dentro de un bloque.
                self.cerrar_video()
                imagen = cargar_fotograma(muestra["imagen"], muestra["fotograma_video"])
        else:
            # La lectura aleatoria se reserva para inspecciones y entrenamiento.
            imagen = cargar_fotograma(muestra["imagen"], muestra["fotograma_video"])
        imagen = self.preprocess(imagen)
        instruccion = muestra["instruccion"]
        if instruccion not in self._tokens_por_instruccion:
            self._tokens_por_instruccion[instruccion] = self.tokenizer([instruccion])[0]
        tokens = self._tokens_por_instruccion[instruccion]
        accion = torch.tensor(muestra["accion"], dtype=torch.float32)

        return imagen, tokens, accion
