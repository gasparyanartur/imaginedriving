from typing import Any
from pathlib import Path
import os
import logging
import yaml

import torch


def set_env(key: str, val: any) -> None:
    os.environ[key] = str(val)


def get_env(key: str) -> str | Path:
    val = os.environ.get(key)

    if val and (val[0] == "/") and (val_path := Path(val)).exists():
        return val_path

    return val


def set_if_no_key(config, key, val):
    if not config.get(key):
        config[key] = val

    return val





def read_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(path: Path, data: dict[str, Any]):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)




def setup_project(config_path: Path = None):
    logging.getLogger().setLevel(logging.INFO)

    if not torch.cuda.is_available():
        logging.warning(f"CUDA not detected. Running on CPU. The code is not supported for CPU and will most likely give incorrect results. Proceed with caution.")


    proj_dir = get_env("PROJ_DIR") or Path.cwd()

    if config_path is None:
        config_path = get_env("PROJ_CONF_PATH")

        if config_path is None:
            config_path = proj_dir / "proj_config.yml"

            if not config_path.exists():
                raise ValueError(f"No config path specified")

    config = read_yaml(config_path)

    proj_dir = set_if_no_key(config, "PROJ_DIR", proj_dir)
    cache_dir = set_if_no_key(config, "CACHE_DIR", proj_dir / ".cache")
    cache_dir = config["CACHE_DIR"]

    set_env("HF_HUB_CACHE", cache_dir / "hf")  # Huggingface cache dir
    set_env("MPLCONFIGDIR", cache_dir / "mpl")  # Matplotlib cache dir

    return config
