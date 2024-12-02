from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML

from datasets import load_dataset


yaml = YAML(typ="safe")


def data_collection() -> None:
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    dataset_name = params.data_collection.path
    split = params.data_collection.split

    dataset_dir = Path("data") / "dataset" / "input"

    dataset = load_dataset(path=dataset_name, split=split)
    dataset.save_to_disk(dataset_dir)


if __name__ == "__main__":
    data_collection()
