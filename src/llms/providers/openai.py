from llms.providers.base import BaseCaptioning
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


# Implementation for OpenAI provider
class OpenAICaptioning(BaseCaptioning):
    def __init__(self):
        self.client = OpenAI()

    def generate_caption(self, model: str, img: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content

    def evaluate_caption(self, model: str, caption: str, prompt: str) -> float:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                    ],
                },
            ],
        )
        return response.choices[0].message.content
