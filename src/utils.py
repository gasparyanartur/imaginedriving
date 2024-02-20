from pathlib import Path
import json


def sort_paths_numerically(paths: list[Path]) -> list[Path]:
  return sorted(paths, key=lambda path: int(path.stem))


def load_json(path):
    with open(path, 'rb') as f:
      return json.load(f)