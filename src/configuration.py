from pathlib import Path
import os
import yaml

def set_env(key: str, val: any) -> None:
    os.environ[key] = str(val)

def get_env(key: str) -> str | Path:
    val = os.environ.get(key)

    if bool(val) and (val[0] == '/') and (val_path := Path(val)).exists():
        return val_path

    return val


def set_if_no_key(config, key, val):
    if not config.get(key):   
        config[key] = val


def setup_project(config_path: Path = None):
    if Path.cwd().stem == "notebooks":
        os.chdir(Path.cwd().parent)

    if config_path is None:
        config_path = get_env("PROJ_CONF_PATH")

        if config_path is None:
            config_path = Path.cwd() / "proj_config.yml"
            print(config_path)
            if not config_path.exists():
                raise ValueError(f"No config path specified")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_if_no_key(config, "proj_dir", Path.cwd())
    proj_dir = config["proj_dir"]

    set_if_no_key(config, "cache_dir", proj_dir / "cache_dir")
    cache_dir = config["cache_dir"]

    set_env("HF_HOME", cache_dir / 'hf' / '.cache')              # Huggingface cache dir
    set_env("MPLCONFIGDIR", cache_dir / 'mpl' / '.cache')        # Matplotlib cache dir

    return config