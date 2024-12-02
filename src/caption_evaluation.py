from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
from collections import Counter

from datasets import load_from_disk, Dataset
from functools import partial
import re

import shutil

from llms.providers.base import BaseCaptioning
from llms.providers import OllamaCaptioning, OpenAICaptioning, VertexAICaptioning

from typing import List
import json

yaml = YAML(typ="safe")


captioning_provider_mapping = {
    "openai": OpenAICaptioning,
    "ollama": OllamaCaptioning,
    "vertexai": VertexAICaptioning,
}


class DatasetsError(Exception):
    def __init__(
        self,
        message="The evaluation stage requires minimum of 2 datasets in ./data/datasets/output to create the evaluation based on ranking different generated captions.",
    ):
        self.message = message
        super().__init__(self.message)


## Preprocessing


def merge_datasets(datasets_dir: Path) -> Dataset:

    datasets_paths = [subdir for subdir in datasets_dir.iterdir() if subdir.is_dir()]

    if len(datasets_paths) < 2:
        raise DatasetsError

    datasets_names = [subdir.name for subdir in datasets_paths]
    datasets = [load_from_disk(path) for path in datasets_paths]

    num_rows = datasets[0].num_rows
    for dataset in datasets:
        if dataset.num_rows != num_rows:
            raise ValueError("All datasets must have the same number of rows.")

    merged_data = {
        "image_encoded": datasets[0]["image_encoded"],
        "context": datasets[0]["context"],
    }

    for name, dataset in zip(datasets_names, datasets):
        merged_data[f"caption_{name}"] = dataset["caption"]

    return Dataset.from_dict(merged_data)


def get_generated_captions_string(example: dict) -> str:
    column_names = example.keys()

    dataset_caption_columns = [
        column for column in column_names if re.match(r"caption_\w+", column)
    ]

    generated_captions = ""

    for idx, column_name in enumerate(dataset_caption_columns):
        generated_captions += f"### Option nr {idx + 1}:\n"
        generated_captions += example[column_name] + "\n\n"

    return generated_captions


## Postprocessing


def postprocess(llm_choice: str, example: dict):
    if "tie" in llm_choice:
        return "tie"
    option_number = re.findall(r"\d+", llm_choice)[0]

    column_names = example.keys()
    dataset_caption_columns = [
        column for column in column_names if re.match(r"caption_\w+", column)
    ]

    return dataset_caption_columns[int(option_number) - 1]


def create_metric(dataset: Dataset):
    dataset_evaluated_df = dataset.to_pandas()

    column_names = dataset_evaluated_df.columns

    dataset_caption_columns = [
        column for column in column_names if re.match(r"caption_\w+", column)
    ]

    counts = dict.fromkeys(dataset_caption_columns, 0)
    counts.update({"tie": 0})

    # Count occurrences of different values in llm_choice
    llm_choices = dataset_evaluated_df["llm_choice"]
    counts.update(dict(Counter(llm_choices)))

    # Save counts to JSON
    metrics_path = Path("data/eval/metrics.json")
    json.dump(counts, open(metrics_path, "w"), indent=4)


## Dataset Processing


def evaluate(provider: BaseCaptioning, model: str, img: str, prompt: str) -> str:
    result = provider.generate(model, img, prompt)
    return result


def process(example, provider: BaseCaptioning, model: str, prompt_template: str):
    prompt = prompt_template.format(
        generated_captions=get_generated_captions_string(example)
    )

    while True:
        try:
            llm_choice = evaluate(provider, model, example["image_encoded"], prompt)
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")

    return {"llm_choice": postprocess(llm_choice, example)}


def caption_evaluation():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    prompt_template_param = params.caption_evaluation.prompt_template
    provider = params.caption_evaluation.provider
    model = params.caption_evaluation.model

    prompt_file = (
        prompt_template_param
        if prompt_template_param.endswith(".txt")
        else prompt_template_param.strip() + ".txt"
    )

    prompt_template_path = Path("data/prompt_template/evaluation") / prompt_file
    prompt_template = prompt_template_path.read_text()

    datasets_dir = Path("data/dataset/output")

    dataset_merged = merge_datasets(datasets_dir)

    provider_obj = captioning_provider_mapping[provider]()

    dataset_evaluated = dataset_merged.map(
        partial(
            process, provider=provider_obj, model=model, prompt_template=prompt_template
        ),
        # batch=true, # TODO
    )

    dataset_out_dir = Path("data/dataset/evaluated")

    dataset_evaluated.save_to_disk(dataset_out_dir)

    create_metric(dataset_evaluated)


if __name__ == "__main__":
    caption_evaluation()
