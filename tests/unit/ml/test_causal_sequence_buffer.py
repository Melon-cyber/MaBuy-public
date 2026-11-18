import time
import numpy as np
import torch
import pytest

from src.infrastructure.ml.pytorch.causal_sequence_buffer import (
    CausalRollingSequenceBuffer,
    MultiAssetSequenceBufferManager
)


class TestCausalSequenceBuffer:
    def test_buffer_push_and_tensor_shape(self):
        buf = CausalRollingSequenceBuffer(seq_len=60, num_features=88)
        assert len(buf) == 0

        # Push 10 vectors
        for i in range(10):
            vec = np.ones(88, dtype=np.float32) * float(i + 1)
            buf.push(vec)

        assert len(buf) == 10
        tensor = buf.get_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 60, 88)
        
        # Verify causal padding: the first 50 elements should repeat the earliest step (1.0)
        # and the last element should be 10.0
        assert torch.allclose(tensor[0, 0, :], torch.tensor(1.0))
        assert torch.allclose(tensor[0, 49, :], torch.tensor(1.0))
        assert torch.allclose(tensor[0, 50, :], torch.tensor(1.0))
        assert torch.allclose(tensor[0, 59, :], torch.tensor(10.0))

    def test_buffer_populate_from_matrix_chronology(self):
        buf = CausalRollingSequenceBuffer(seq_len=60, num_features=14)
        
        # 100 timesteps historical matrix
        mat = np.arange(100, dtype=np.float32)[:, None] * np.ones((100, 14), dtype=np.float32)
        buf.populate_from_matrix(mat)
        
        assert len(buf) == 60
        tensor = buf.get_tensor()
        assert tensor.shape == (1, 60, 14)
        
        # The first step in 60-window from [0..99] must be 40.0, and the last must be 99.0
        assert torch.allclose(tensor[0, 0, :], torch.tensor(40.0))
        assert torch.allclose(tensor[0, 59, :], torch.tensor(99.0))

    def test_multi_asset_manager_isolation(self):
        manager = MultiAssetSequenceBufferManager(seq_len=60, num_features=88)
        
        mat_btc = np.ones((60, 88), dtype=np.float32) * 50000.0
        mat_eth = np.ones((60, 88), dtype=np.float32) * 3000.0
        
        t_btc = manager.update_and_get_tensor("BTCUSDT", "1h", feature_matrix=mat_btc)
        t_eth = manager.update_and_get_tensor("ETHUSDT", "1h", feature_matrix=mat_eth)
        
        assert torch.allclose(t_btc[0, -1, 0], torch.tensor(50000.0))
        assert torch.allclose(t_eth[0, -1, 0], torch.tensor(3000.0))

    def test_sub_20ms_inference_benchmark(self):
        """Rigorous latency benchmark ensuring tensor generation is sub-millisecond."""
        buf = CausalRollingSequenceBuffer(seq_len=60, num_features=88)
        mat = np.random.randn(60, 88).astype(np.float32)
        buf.populate_from_matrix(mat)
        
        # Measure 1000 iterations
        start = time.perf_counter()
        for _ in range(500):
            t = buf.get_tensor()
        elapsed_total_ms = (time.perf_counter() - start) * 1000.0
        avg_ms = elapsed_total_ms / 500
        
        # Extraction must take < 1.0 ms per call
        assert avg_ms < 1.0, f"Average extraction time {avg_ms:.3f}ms exceeded 1.0ms limit"
