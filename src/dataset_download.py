from pathlib import Path
from box import ConfigBox

from datasets import load_dataset
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def dataset_download():
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    dataset_name = params["dataset"]
    split = params["split"]

    dataset_dir = Path("data") / "input"

    dataset = load_dataset(path=dataset_name, split=split)
    dataset.save_to_disk(dataset_dir)


if __name__ == "__main__":
    dataset_download()
