from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML

from datasets import load_from_disk
from functools import partial

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
    grade = provider.evaluate_caption(model, caption, prompt)
    return int(grade)


def process(example, provider: BaseCaptioning, model: str, prompt_template: str):
    prompt = prompt_template.format(context=example["context"])
    grade = evaluate(provider, model, example["caption"], prompt)
    return {"llm_metric": grade}


def evaluation():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
    provider = params.evaluation.provider
    model = params.evaluation.model
    prompt_template_param = params.evaluation.prompt_template

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
        "average_score": float(dataset_evaluated_df["llm_metric"].mean()),
        "min_score": float(dataset_evaluated_df["llm_metric"].min()),
        "max_score": float(dataset_evaluated_df["llm_metric"].max()),
    }

    metrics_path = Path("eval/metrics.json")
    json.dump(metrics, open(metrics_path, "w"))


if __name__ == "__main__":
    evaluation()
