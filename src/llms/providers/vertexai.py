from llms.providers.base import BaseCaptioning
from vertexai.generative_models import GenerativeModel

from dotenv import load_dotenv

load_dotenv()


class VertexAICaptioning(BaseCaptioning):
    def __init__(self):
        self.client = GenerativeModel

    def generate_caption(self, model: str, img: str, prompt: str) -> str:
        response = self.client(model).generate_content([prompt, img])
        return response.candidates[0].content.parts[0].text
