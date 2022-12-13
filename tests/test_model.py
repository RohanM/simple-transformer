import unittest
from unittest import TestCase
import torch
from torch import LongTensor
from simple_transformer.model import Model

class TestModel(TestCase):
    def test_model(self) -> None:
        model = Model(4, 8)
        result = model(LongTensor([[0, 1, 2, 3], [1, 2, 3, 0]]))
        self.assertEqual(result.shape, torch.Size([2, 4]))

    def test_reverse_embedding(self) -> None:
        model = Model(4, 8)
        result = model.reverse_embedding(model.embedding(LongTensor([[0, 1, 3, 2], [1, 2, 3, 0]])))
        self.assertTrue(torch.equal(result, LongTensor([[0, 1, 3, 2], [1, 2, 3, 0]])))
