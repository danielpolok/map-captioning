from llms.providers.base import CaptioningBase
from openai import OpenAI


# Implementation for OpenAI provider
class OpenAICaptioning(CaptioningBase):
    def __init__(self):
        self.client = OpenAI()

    def generate_caption(self, model_name: str, img_str: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=model_name,
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
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content
