from pathlib import Path
from src.data import DynamicDataset, PandasetInfoGetter, PandasetImageDataGetter, read_data_tree
from torch.utils.data import DataLoader


data_path = Path("/home/x_artga/dev/data/pandaset")
data_tree = read_data_tree(Path("scripts/tmp-dict.yml"))
data_tree2 = read_data_tree(Path("scripts/tmp-dict2.yml"))
info_getter = PandasetInfoGetter()
data_getters = {"image": PandasetImageDataGetter()}

ds1 = DynamicDataset(data_path, data_tree, info_getter, data_getters)
ds2 = DynamicDataset(data_path, data_tree2, info_getter, data_getters)


dl = DataLoader(ds1, batch_size=4)

ex_batch = next(iter(dl))
print(ex_batch["image"].shape)

matches = ds1.get_matching(ds2)
print(len(matches))

