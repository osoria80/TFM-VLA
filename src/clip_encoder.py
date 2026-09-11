import torch
import open_clip


class CLIPEncoder:
    def __init__(self, nombre_modelo, pesos_pretrenados, dispositivo, congelar):
        self.dispositivo = dispositivo
        self.congelado = congelar
        self.modelo, _, self.preprocess = open_clip.create_model_and_transforms(
            nombre_modelo,
            pretrained=pesos_pretrenados,
        )
        self.modelo = self.modelo.to(dispositivo)
        self.tokenizer = open_clip.get_tokenizer(nombre_modelo)

        if congelar:
            for parametro in self.modelo.parameters():
                parametro.requires_grad = False
            self.modelo.eval()

    def encode_image(self, imagenes):
        imagenes = imagenes.to(self.dispositivo)
        if self.congelado:
            with torch.no_grad():
                return self.modelo.encode_image(imagenes)
        return self.modelo.encode_image(imagenes)

    def encode_text(self, tokens):
        tokens = tokens.to(self.dispositivo)
        if self.congelado:
            with torch.no_grad():
                return self.modelo.encode_text(tokens)
        return self.modelo.encode_text(tokens)
