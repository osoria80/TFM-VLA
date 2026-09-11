import cv2
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
