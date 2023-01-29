import unittest
from unittest import TestCase
import torch
from torch import tensor, LongTensor
from simple_transformer.transformer_model import TransformerModel

class TestTransformerModel(TestCase):
    def setUp(self) -> None:
        self.model = TransformerModel(4, 4, 8, 2)

    def test_model(self) -> None:
        y, _ = self.model(LongTensor([[0, 1], [1, 2]]))
        self.assertEqual(y.shape, torch.Size([2, 2, 4]))

    def test_loss(self) -> None:
        _, loss = self.model(
            tensor([[0, 1], [1, 2]], dtype=torch.long),
            tensor([[0, 1], [1, 2]], dtype=torch.long),
        )
        self.assertTrue(loss.item() > 0)

    def test_infer_one(self) -> None:
        x = LongTensor([0, 1, 2, 3])
        result = self.model.infer_one(x)
        self.assertEqual(result.shape, torch.Size((1,)))
