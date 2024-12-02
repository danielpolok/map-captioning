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


def process(example, provider: BaseCaptioning, model: str, prompt_template: str):
    prompt = prompt_template.format(image_additional_informations=example["context"])
    while True:
        try:
            caption = generate(provider, model, example["image_encoded"], prompt)
            break
        except:
            pass

    return {"caption": caption}


def caption_generation():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    prompt_template_param = params.caption_generation.prompt_template
    provider = params.caption_generation.provider
    model = params.caption_generation.model

    prompt_file = (
        prompt_template_param
        if prompt_template_param.endswith(".txt")
        else prompt_template_param.strip() + ".txt"
    )

    prompt_template_path = Path("data/prompt_template/captioning") / prompt_file
    prompt_template = prompt_template_path.read_text()

    dataset_dir = Path("data/dataset/preprocessed")
    dataset = load_from_disk(dataset_dir)

    provider_obj = captioning_provider_mapping[provider]()

    dataset_captioned = dataset.map(
        partial(
            process, provider=provider_obj, model=model, prompt_template=prompt_template
        ),
        # batch=true, # TODO
    )

    dataset_out_dir = Path("data") / "dataset" / "output"

    dataset_captioned.save_to_disk(dataset_out_dir)


if __name__ == "__main__":
    caption_generation()
