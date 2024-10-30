from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML

from datasets import load_from_disk
from functools import partial

from PIL.Image import Image
import base64
from io import BytesIO

from llms.providers.base import BaseCaptioning
from llms.providers import OllamaCaptioning, OpenAICaptioning, VertexAICaptioning

from typing import List

yaml = YAML(typ="safe")


captioning_provider_mapping = {
    "openai": OpenAICaptioning,
    "ollama": OllamaCaptioning,
    "vertexai": VertexAICaptioning,
}


## Dataset Processing


def generate(provider: BaseCaptioning, model: str, img: str, prompt: str) -> str:
    return provider.generate_caption(model, img, prompt)


def process(example, provider: BaseCaptioning, model: str):
    caption = generate(provider, model, example["image_encoded"], example["prompt"])
    return {"caption": caption}


def captioning():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    provider = params.captioning.provider
    model = params.captioning.model

    dataset_dir = Path("data/dataset/preprocessed")
    dataset = load_from_disk(dataset_dir)

    provider_obj = captioning_provider_mapping[provider]()

    dataset_captioned = dataset.map(
        partial(process, provider=provider_obj, model=model),
        # batch=true, # TODO
    )

    dataset_out_dir = Path("data") / "dataset" / "output"

    dataset_captioned.save_to_disk(dataset_out_dir)


if __name__ == "__main__":
    captioning()
