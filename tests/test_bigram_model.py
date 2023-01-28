import unittest
from unittest import TestCase
import torch
from torch import tensor, LongTensor
from simple_transformer.bigram_model import BigramModel

class TestBigramModel(TestCase):
    def test_model(self) -> None:
        model = BigramModel(4)
        y, _ = model(LongTensor([[0, 1], [1, 2]]))
        self.assertEqual(y.shape, torch.Size([2, 2, 4]))

    def test_loss(self) -> None:
        model = BigramModel(4)
        _, loss = model(
            tensor([[0, 1], [1, 2]], dtype=torch.long),
            tensor([[0, 1], [1, 2]], dtype=torch.long),
        )
        self.assertTrue(loss.item() > 0)

    def test_infer_one(self) -> None:
        model = BigramModel(4)
        x = LongTensor([0, 1, 2, 3])
        result = model.infer_one(x)
        self.assertEqual(result.shape, torch.Size((1,)))
