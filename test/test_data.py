import unittest

import torch
from torch import Tensor

from src.data import NamedImageDataset


class TestRGBDataset(unittest.TestCase):
    def test_getitem(self):
        ex_names = ["a", "b", "c", "d"]
        ex_imgs = torch.arange(300).reshape(4, 3, 5, 5) / 300

        rgb = NamedImageDataset(ex_names, ex_imgs)
        ex_name, ex_img = rgb[0]

        self.assertTrue(ex_name == ex_names[0])
        self.assertTrue(torch.allclose(ex_img, ex_imgs[0]))

    def test_iterate(self):
        ex_names = ["a", "b", "c", "d"]
        ex_imgs = torch.arange(300).reshape(4, 3, 5, 5) / 300

        rgb = NamedImageDataset(ex_names, ex_imgs)

        for i, (name, img) in enumerate(rgb):
            self.assertTrue(torch.allclose(img, ex_imgs[i]))


if __name__ == "__main__":
    unittest.main()
