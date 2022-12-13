import unittest
from unittest import TestCase
from simple_transformer.data import Data

class TestData(TestCase):
    def test_data(self) -> None:
        data = Data('This is my dataset. Watch it roar!', 3)
        self.assertEqual(len(data), 6)
        self.assertEqual(
            list(data), [
                ((202292, 98417, 121336), (98417, 121336, -1)),
                ((98417, 121336, -1), (121336, -1, 236736)),
                ((121336, -1, 236736), (-1, 236736, 230736)),
                ((-1, 236736, 230736), (236736, 230736, 99082)),
                ((236736, 230736, 99082), (230736, 99082, 170332)),
                ((230736, 99082, 170332), (99082, 170332, 236738))
            ]
        )
