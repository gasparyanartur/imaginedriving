from pathlib import Path
from torch import Tensor
from src.data import PandasetDataset


PandasetDataset(Path('~/data/pandaset'), ["01", "02"])