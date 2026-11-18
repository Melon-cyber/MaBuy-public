import os, sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
from src.infrastructure.mlops.dataset_collector import MaIADatasetCollector
def main():
    print("=" * 60)
    print("MaBuy MLOps: Building Custom MaIA LLM Fine-Tuning Dataset")
    print("=" * 60)
    collector = MaIADatasetCollector(storage_dir='data/datasets/fine_tuning')
    existing = len(collector.entries)
    print(f"[*] Existing Raw Samples: {existing}")
    if existing < 5:
        added = collector.generate_golden_seed_dataset()
        print(f"[+] Added {added} seed multi-turn training pairs.")
    meta = collector.export_huggingface_splits(train_ratio=0.85)
    print(f"[+] Total Samples: {meta['total_samples']}")
    print(f"[+] Train: {meta['train_samples']}, Val: {meta['val_samples']}")
    print("Ready for Unsloth / Hugging Face LoRA fine-tuning!")
if __name__ == "__main__": main()