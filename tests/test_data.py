import unittest
from unittest import TestCase
import torch
from torch import LongTensor
from simple_transformer.data import Data

class TestData(TestCase):
    def test_data(self) -> None:
        data = Data('This is my dataset. Watch it roar!', 3)
        self.assertEqual(len(data), 6)
        self.assertTrue(torch.equal(data[0][0], LongTensor([205266, 112693, 131659])))
        self.assertTrue(torch.equal(data[0][1], LongTensor([112693, 131659,     -1])))
        self.assertTrue(torch.equal(data[1][0], LongTensor([112693, 131659,     -1])))
        self.assertTrue(torch.equal(data[1][1], LongTensor([131659,     -1, 236736])))
        self.assertTrue(torch.equal(data[2][0], LongTensor([131659,     -1, 236736])))
        self.assertTrue(torch.equal(data[2][1], LongTensor([    -1, 236736, 232212])))
        self.assertTrue(torch.equal(data[3][0], LongTensor([    -1, 236736, 232212])))
        self.assertTrue(torch.equal(data[3][1], LongTensor([236736, 232212, 113272])))
        self.assertTrue(torch.equal(data[4][0], LongTensor([236736, 232212, 113272])))
        self.assertTrue(torch.equal(data[4][1], LongTensor([232212, 113272, 176543])))
        self.assertTrue(torch.equal(data[5][0], LongTensor([232212, 113272, 176543])))
        self.assertTrue(torch.equal(data[5][1], LongTensor([113272, 176543, 236738])))
