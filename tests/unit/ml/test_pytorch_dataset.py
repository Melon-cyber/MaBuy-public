import unittest
import torch
import numpy as np
from src.infrastructure.ml.pytorch.dataset import CryptoSequenceDataset


class TestCryptoSequenceDataset(unittest.TestCase):

    def test_3d_tensor_creation(self):
        N = 200
        F = 12
        seq_len = 60

        X_dummy = np.random.randn(N, F)
        y_dummy = np.random.randn(N)

        dataset = CryptoSequenceDataset(X_dummy, y_dummy, seq_len=seq_len)

        self.assertEqual(len(dataset), N - seq_len)

        x_seq, y_target, y_dir = dataset[0]
        self.assertEqual(x_seq.shape, (seq_len, F))
        self.assertEqual(y_target.shape, (1,))
        self.assertEqual(y_dir.shape, (1,))


if __name__ == "__main__":
    unittest.main()
