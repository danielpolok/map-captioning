from llms.providers.base import CaptioningBase
from vertexai.generative_models import GenerativeModel


class VertexAICaptioning(CaptioningBase):
    def __init__(self):
        self.client = GenerativeModel

    def generate_caption(self, model_name: str, img_str: str, prompt: str) -> str:
        response = self.client(model_name).generate_content([prompt, img_str])
        return response.candidates[0].content.parts[0].text
