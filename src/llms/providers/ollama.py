from llms.providers.base import BaseCaptioning
import ollama

from dotenv import load_dotenv

load_dotenv()


class OllamaCaptioning(BaseCaptioning):
    def __init__(self):
        self.client = ollama

    def generate_caption(self, model: str, img: str, prompt: str) -> str:
        response = self.client.generate(model=model, prompt=prompt, images=[img])
        return response["response"]
