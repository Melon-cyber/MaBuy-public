# -*- coding: utf-8 -*-
import os
import shutil
import pytest
from src.infrastructure.mlops.dataset_collector import MaIADatasetCollector

@pytest.fixture
def temp_collector(tmp_path):
    storage = str(tmp_path / 'fine_tuning_test')
    return MaIADatasetCollector(storage_dir=storage)

def test_add_dialogue(temp_collector):
    messages = [
        {'role': 'user', 'content': 'Привет, Майя!'},
        {'role': 'assistant', 'content': 'Привет, трейдер!'}
    ]
    dp = temp_collector.add_dialogue(messages=messages, persona='yandere', category='yandere_protection')
    assert dp.id.startswith('mabuy_dp_')
    assert len(temp_collector.entries) == 1
    assert temp_collector.entries[0].persona == 'yandere'

def test_export_huggingface_splits(temp_collector):
    temp_collector.generate_golden_seed_dataset()
    meta = temp_collector.export_huggingface_splits(train_ratio=0.8)
    assert meta['total_samples'] >= 5
    assert meta['train_samples'] >= 1
    assert os.path.exists(os.path.join(temp_collector.storage_dir, 'train.jsonl'))
    assert os.path.exists(os.path.join(temp_collector.storage_dir, 'val.jsonl'))
