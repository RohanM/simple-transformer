import unittest
from unittest import TestCase
import torch
from torch import LongTensor
from simple_transformer.model import Model

class TestModel(TestCase):
    def test_model(self) -> None:
        model = Model()
        self.assertTrue(torch.equal(
            model(LongTensor([[1, 2, 3]])),
            LongTensor([[1, 2, 3]])
        ))
