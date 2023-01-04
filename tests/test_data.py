import unittest
from unittest import TestCase
import torch
from torch import LongTensor
from simple_transformer.data import Data

class TestData(TestCase):
    def test_data(self) -> None:
        data = Data('This is my dataset. Watch it roar!', 3)
        self.assertEqual(len(data), 6)
        self.assertTrue(torch.equal(data[0][0], LongTensor([205267, 112694, 131660])))
        self.assertTrue(torch.equal(data[0][1], LongTensor([112694, 131660,      0])))
        self.assertTrue(torch.equal(data[1][0], LongTensor([112694, 131660,      0])))
        self.assertTrue(torch.equal(data[1][1], LongTensor([131660,      0, 236737])))
        self.assertTrue(torch.equal(data[2][0], LongTensor([131660,      0, 236737])))
        self.assertTrue(torch.equal(data[2][1], LongTensor([     0, 236737, 232213])))
        self.assertTrue(torch.equal(data[3][0], LongTensor([     0, 236737, 232213])))
        self.assertTrue(torch.equal(data[3][1], LongTensor([236737, 232213, 113273])))
        self.assertTrue(torch.equal(data[4][0], LongTensor([236737, 232213, 113273])))
        self.assertTrue(torch.equal(data[4][1], LongTensor([232213, 113273, 176544])))
        self.assertTrue(torch.equal(data[5][0], LongTensor([232213, 113273, 176544])))
        self.assertTrue(torch.equal(data[5][1], LongTensor([113273, 176544, 236739])))
