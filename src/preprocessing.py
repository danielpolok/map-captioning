from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML

from datasets import load_from_disk
from functools import partial

from PIL.Image import Image
import base64
from io import BytesIO

from typing import List

yaml = YAML(typ="safe")

## Dataset Pre-processing


### Image
def encode_img(img: Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# def encode_img_batch(batch: List[Image]) -> List[str]:
#     return [encode_img(img) for img in batch]


### Text
def transform_caption(caption: str):
    return caption.split(":", 1)[1].strip()


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
    transformed_caption = transform_caption(example["caption"])
    prompt = create_prompt(transformed_caption, prompt_template)

    return {
        "image_encoded": image,
        "prompt": prompt,
        "context": transformed_caption,
    }


def dataset_preprocessing():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    prompt_template_param = params.preprocessing.prompt_template

    prompt_template_file = (
        prompt_template_param
        if prompt_template_param.endswith(".txt")
        else prompt_template_param.strip() + ".txt"
    )

    prompt_template_path = (
        Path("data/prompt_template/captioning") / prompt_template_file
    )
    prompt_template = prompt_template_path.read_text()

    dataset_dir = Path("data/dataset/input")
    dataset = load_from_disk(dataset_dir)

    dataset_preprocessed = dataset.map(
        partial(preprocess, prompt_template=prompt_template)
        # batch=true # TODO
    )

    # dataset_preprocessed = dataset_preprocessed.rename_column("caption", "context")

    new_column_order = ["image_encoded", "context", "prompt"]
    dataset_preprocessed = dataset_preprocessed.select_columns(new_column_order)

    dataset_out_dir = Path("data/dataset/preprocessed")

    dataset_preprocessed.save_to_disk(dataset_out_dir)


if __name__ == "__main__":
    dataset_preprocessing()
