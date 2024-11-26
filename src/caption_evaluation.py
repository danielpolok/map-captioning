from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML

from datasets import load_from_disk
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


## Dataset Processing


def evaluate(provider: BaseCaptioning, model: str, caption: str, prompt: str) -> str:
    score = provider.evaluate_caption(model, prompt)

    match = re.search(r"\d+", score)
    score = match.group(0)

    return int(score)


def process(example, provider: BaseCaptioning, model: str, prompt_template: str):
    prompt = prompt_template.format(
        image_additional_informations=example["context"],
        generated_caption=example["caption"],
    )
    score = evaluate(provider, model, example["caption"], prompt)
    return {"score": score}


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

    dataset_dir = Path("data/dataset/output")
    dataset = load_from_disk(dataset_dir)

    provider_obj = captioning_provider_mapping[provider]()

    dataset_evaluated = dataset.map(
        partial(
            process, provider=provider_obj, model=model, prompt_template=prompt_template
        ),
        # batch=true, # TODO
    )

    dataset_out_dir = Path("data/dataset/evaluated")

    dataset_evaluated.save_to_disk(dataset_out_dir)

    dataset_evaluated_df = dataset_evaluated.to_pandas()

    metrics = {
        "average_score": float(dataset_evaluated_df["score"].mean()),
        "min_score": float(dataset_evaluated_df["score"].min()),
        "max_score": float(dataset_evaluated_df["score"].max()),
    }

    metrics_path = Path("data/eval/metrics.json")
    json.dump(metrics, open(metrics_path, "w"))


if __name__ == "__main__":
    caption_evaluation()
