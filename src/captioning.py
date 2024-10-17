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


## Dataset Pre-processing

### Image


def encode_img(img: Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# def encode_img_batch(batch: List[Image]) -> List[str]:
#     return [encode_img(img) for img in batch]

### Text


def create_prompt(
    context: str,
    prompt_template: str,
) -> str:

    if "{context}" in prompt_template:
        # Format the prompt_template with the provided context
        formatted_prompt = prompt_template.format(context=context)
    else:
        # If {context} is not present, return the prompt_template as is
        formatted_prompt = prompt_template

    return formatted_prompt


### Pre-process


def preprocess(example, prompt_template: str):
    image = encode_img(example["image"])
    prompt = create_prompt(example["caption"], prompt_template)

    return {"image_encoded": image, "prompt": prompt}


## Dataset Processing


def generate(provider: BaseCaptioning, model: str, img: str, prompt: str) -> str:
    return provider.generate_caption(model, img, prompt)


def process(example, idx: int, provider: BaseCaptioning, model: str):
    caption = generate(provider, model, example["image_encoded"], example["prompt"])
    return {
        # "image_idx": idx,
        "caption": caption
    }


def captioning():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    provider = params.captioning.provider
    model = params.captioning.model
    prompt_template = params.captioning.prompt_template

    prompt_template = (
        prompt_template
        if prompt_template.endswith(".txt")
        else prompt_template.strip() + ".txt"
    )

    prompt_file = Path("data") / "prompt_template" / prompt_template
    prompt_template = prompt_file.read_text()

    dataset_dir = Path("data") / "dataset" / "input"
    dataset = load_from_disk(dataset_dir)

    dataset = dataset.map(
        partial(preprocess, prompt_template=prompt_template)
        # batch=true # TODO
    )

    dataset = dataset.remove_columns(["caption"])

    provider_obj = captioning_provider_mapping[provider]()

    dataset_captioned = dataset.map(
        partial(process, provider=provider_obj, model=model),
        # batch=true, # TODO
        with_indices=True,
    )

    dataset_captioned = dataset_captioned.remove_columns(["image_encoded"])

    dataset_out_dir = Path("data") / "dataset" / "output"

    dataset_captioned.save_to_disk(dataset_out_dir)


if __name__ == "__main__":
    captioning()
