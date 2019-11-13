import unittest
from code.inference import transform

class TestTransform(unittest.TestCase):
    def test_transform(self):
        sentences = [
            "hello world",
            "hello world",
            "hello world",
            "hello world",
            "hello world",
            "hello world",
            "hello world",
            "hello world",
            "hello world",
            "hello world",
            "huiying is awesome"
    ]*1000

        #print(transform(sentences, 10))
        self.assertEqual(
            ([[101, 7592, 2088, 102, 0, 0, 0, 0, 0, 0], [101, 17504, 14147, 2003, 12476, 102, 0, 0, 0, 0]],
            [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ), transform(sentences, 10))

