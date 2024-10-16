from src.llms.providers.base import CaptioningBase
import ollama


class OllamaCaptioning(CaptioningBase):
    def __init__(self):
        self.client = ollama

    def generate_caption(self, model_name: str, img_str: str, prompt: str) -> str:
        response = self.client.generate(
            model=model_name, prompt=prompt, images=[img_str]
        )
        return response["response"]
