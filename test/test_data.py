import unittest

import torch
from torch import Tensor

from src.data import ImageDataset



class TestRGBDataset(unittest.TestCase):
    def test_getitem(self):
        ex_names = ["a", "b", "c", "d"]
        ex_imgs = torch.arange(300).reshape(4, 3, 5, 5) / 300

        rgb = ImageDataset(ex_names, ex_imgs)
        ex_rgb = rgb[0]

        self.assertTrue(ex_rgb["name"] == ex_names[0])
        self.assertTrue(torch.allclose(ex_rgb["image"], ex_imgs[0]))

    def test_iterate(self):
        ex_names = ["a", "b", "c", "d"]
        ex_imgs = torch.arange(300).reshape(4, 3, 5, 5) / 300

        rgb = ImageDataset(ex_names, ex_imgs)

        for i in range(4):
            self.assertTrue(torch.allclose(rgb[i]["image"], ex_imgs[i]))

if __name__ == '__main__':
    unittest.main()
