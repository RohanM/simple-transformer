import unittest
from unittest import TestCase
import torch
from torch import tensor, LongTensor
from simple_transformer.model import Model

class TestModel(TestCase):
    def test_model(self) -> None:
        model = Model(4, 8)
        result = model(LongTensor([[0, 1, 2, 3], [1, 2, 3, 0]]))
        self.assertEqual(result.shape, torch.Size([2, 4, 8]))

    def test_reverse_embedding(self) -> None:
        model = Model(4, 8)
        result = model.reverse_embedding(model.embedding(LongTensor([[0, 1, 3, 2], [1, 2, 3, 0]])))
        self.assertTrue(torch.equal(result, LongTensor([[0, 1, 3, 2], [1, 2, 3, 0]])))

    def test_loss(self) -> None:
        model = Model(4, 8)
        result = model.loss(
            tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.float),
            tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.float),
        )
        self.assertEqual(result, 0)

    def test_infer_one(self) -> None:
        model = Model(4, 8)
        x = LongTensor([0, 1, 2, 3])
        result = model.infer_one(x)
        self.assertEqual(result.shape, torch.Size((1,)))

    def test_embed(self) -> None:
        model = Model(4, 8)
        x = LongTensor([0, 1, 2, 3])
        result = model.embed(x)
        self.assertEqual(result.shape, torch.Size([4, 8]))
