import unittest
from unittest import TestCase
import torch
from torch import LongTensor
from simple_transformer.data import Data

class TestData(TestCase):
    def test_data(self) -> None:
        data = Data('AbCdEf123', 3)
        self.assertEqual(len(data), 6)

        self.assertTrue(torch.equal(data[0][0], LongTensor([23, 53, 25])))
        self.assertTrue(torch.equal(data[0][1], LongTensor([53, 25, 55])))

        self.assertTrue(torch.equal(data[1][0], LongTensor([53, 25, 55])))
        self.assertTrue(torch.equal(data[1][1], LongTensor([25, 55, 27])))

        self.assertTrue(torch.equal(data[2][0], LongTensor([25, 55, 27])))
        self.assertTrue(torch.equal(data[2][1], LongTensor([55, 27, 57])))

        self.assertTrue(torch.equal(data[3][0], LongTensor([55, 27, 57])))
        self.assertTrue(torch.equal(data[3][1], LongTensor([27, 57, 13])))

        self.assertTrue(torch.equal(data[4][0], LongTensor([27, 57, 13])))
        self.assertTrue(torch.equal(data[4][1], LongTensor([57, 13, 14])))

        self.assertTrue(torch.equal(data[5][0], LongTensor([57, 13, 14])))
        self.assertTrue(torch.equal(data[5][1], LongTensor([13, 14, 15])))
