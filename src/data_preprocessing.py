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


### Pre-process
def process(example):
    image = encode_img(example["image"])
    transformed_caption = transform_caption(example["caption"])

    return {
        "image_encoded": image,
        "context": transformed_caption,
    }


def dataset_preprocessing():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    dataset_dir = Path("data/dataset/input")
    dataset = load_from_disk(dataset_dir)

    dataset_preprocessed = dataset.map(
        process
        # batch=true # TODO
    )

    new_column_order = ["image_encoded", "context"]
    dataset_preprocessed = dataset_preprocessed.select_columns(new_column_order)

    dataset_out_dir = Path("data/dataset/preprocessed")

    dataset_preprocessed.save_to_disk(dataset_out_dir)


if __name__ == "__main__":
    dataset_preprocessing()
