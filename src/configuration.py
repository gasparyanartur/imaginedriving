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


def iter_scene_names(start_name, end_name):
    assert len(end_name) == len(start_name)

    start_name = list(start_name)
    end_name = list(end_name)

    for c1, c2 in zip(start_name, end_name):    # Check if start_name > end_name initially
        if c1 > c2:
            return
        
        elif c1 < c2:
            break
    else:       # Special case: Both words are equal
        yield ''.join(start_name)
        return

    def inc_char(word, i):
        c = word[i]
        c = chr(ord(c)+1)

        if c.isdigit():
            word[i] = c
        else:
            word[i] = '0'
            if i >= 0:
                inc_char(word, i-1)

    def words_equal(word1, word2):
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                return False

        return True

    word = start_name.copy()
    i_end = len(start_name)-1
    x = 0
    while True:
        yield ''.join(word)

        inc_char(word, i_end)
        
        if words_equal(word, end_name):
            yield ''.join(word)
            break

        if words_equal(word, start_name):
            break

        x += 1
        if x == 10:
            break


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

    for dataset in config["datasets"].values():
        dataset["path"] = Path(dataset["path"])
        scene_list = []
        for scene in dataset["scenes"]:
            if scene.isdigit():
                scene_list.append(scene)
            else:
                assert  '-' in scene
                scene_from, scene_to = scene.split('-')                    
                assert scene_from.isdigit() and scene_to.isdigit()

                scene_list.extend(iter_scene_names(scene_from, scene_to))

        dataset["scenes"] = scene_list

    return config