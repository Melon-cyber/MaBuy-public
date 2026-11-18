import os
import sys
import json
import time
import torch
import joblib
import numpy as np

BASE_DIR = '/root/mabuy2' if os.name != 'nt' and os.path.exists('/root/mabuy2') else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SNAPSHOT_PATH = os.path.join(MODELS_DIR, 'pre_retrain_snapshot_latest.json')

snapshot = {
    'timestamp': time.time(),
    'timestamp_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    'pytorch': {},
    'gbdt': {},
    'registry': {}
}

# 1. PyTorch Models
for symbol in ['BTC', 'ETH', 'SOL']:
    pt_path = os.path.join(MODELS_DIR, f'pytorch_attention_lstm_{symbol}USDT_1h.pt')
    if os.path.exists(pt_path):
        checkpoint = torch.load(pt_path, map_location='cpu')
        state = checkpoint.get('model_state_dict', checkpoint)
        w_ih = state.get('lstm.weight_ih_l0')
        w_att = state.get('attention.w_a.weight')
        w_ret = state.get('fc_return.3.weight') if 'fc_return.3.weight' in state else state.get('head_1h.fc_return.4.weight')
        w_prob = state.get('fc_prob.3.weight') if 'fc_prob.3.weight' in state else state.get('head_1h.fc_prob.4.weight')
        
        snapshot['pytorch'][symbol] = {
            'file_size_bytes': os.path.getsize(pt_path),
            'mtime': os.path.getmtime(pt_path),
            'w_ih_norm': float(torch.norm(w_ih)) if w_ih is not None else 0.0,
            'w_att_norm': float(torch.norm(w_att)) if w_att is not None else 0.0,
            'w_ret_norm': float(torch.norm(w_ret)) if w_ret is not None else 0.0,
            'w_prob_norm': float(torch.norm(w_prob)) if w_prob is not None else 0.0,
        }

# 2. GBDT Models
for symbol in ['BTC', 'ETH', 'SOL']:
    gbdt_path = os.path.join(MODELS_DIR, f'gbdt_{symbol}USDT_1h.joblib')
    if os.path.exists(gbdt_path):
        m = joblib.load(gbdt_path)
        snapshot['gbdt'][symbol] = {
            'file_size_bytes': os.path.getsize(gbdt_path),
            'mtime': os.path.getmtime(gbdt_path),
            'n_iter_': getattr(m, 'n_iter_', 0),
            'train_score_last': float(m.train_score_[-1]) if hasattr(m, 'train_score_') and len(m.train_score_) > 0 else 0.0
        }

# 3. Registry Manifest
manifest_path = os.path.join(MODELS_DIR, 'registry_manifest.json')
if os.path.exists(manifest_path):
    with open(manifest_path, 'r') as f:
        snapshot['registry'] = json.load(f).get('active_models', {})

with open(SNAPSHOT_PATH, 'w') as f:
    json.dump(snapshot, f, indent=2)

print(f"Pre-retrain snapshot saved successfully to {SNAPSHOT_PATH}!")
