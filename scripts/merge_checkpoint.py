# -*- coding: utf-8 -*-
"""
Script to merge LoRA adapter from checkpoint into base model.
Nhận 3 tham số:
1. Base model directory (chứa model-00001-of-00002.safetensors và model-00002-of-00002.safetensors)
2. Checkpoint directory (chứa adapter_model.safetensors)
3. Output directory (thư mục lưu model đã merge)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

def merge_checkpoint(
    base_model_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Merge LoRA adapter from checkpoint into base model.
    
    Args:
        base_model_dir: Thư mục chứa base model (Uni-MuMER-Qwen2.5-VL-3B)
                       với model-00001-of-00002.safetensors và model-00002-of-00002.safetensors
        checkpoint_dir: Thư mục checkpoint chứa adapter_model.safetensors
        output_dir: Thư mục lưu model đã merge
    """
    base_model_dir = Path(base_model_dir)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    
    # Validate paths
    if not base_model_dir.exists():
        raise ValueError(f"Base model directory does not exist: {base_model_dir}")
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Check required files
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    adapter_model_path = checkpoint_dir / "adapter_model.safetensors"
    
    if not adapter_config_path.exists():
        raise ValueError(f"Adapter config not found: {adapter_config_path}")
    
    if not adapter_model_path.exists():
        raise ValueError(f"Adapter model not found: {adapter_model_path}")
    
    # Check base model files
    model_index = base_model_dir / "model.safetensors.index.json"
    config_json = base_model_dir / "config.json"
    
    if not config_json.exists():
        raise ValueError(f"Base model config.json not found: {config_json}")
    
    print("=" * 60)
    print("MERGE LORA ADAPTER INTO BASE MODEL")
    print("=" * 60)
    print(f"Base model directory: {base_model_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load base model
    print("\n[1/4] Loading base model from local directory...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load to CPU first for merging
    )
    print("✓ Base model loaded")
    
    # Load adapter
    print("\n[2/4] Loading LoRA adapter from checkpoint...")
    adapter_model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_dir),
        device_map="cpu",
    )
    print("✓ Adapter loaded")
    
    # Merge adapter into base model
    print("\n[3/4] Merging adapter into base model...")
    merged_model = adapter_model.merge_and_unload()
    print("✓ Adapter merged")
    
    # Save merged model
    print(f"\n[4/4] Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
    )
    print("✓ Model saved")
    
    # Copy processor and tokenizer files from base model
    print("\n[5/5] Copying processor and tokenizer files...")
    processor = AutoProcessor.from_pretrained(
        str(base_model_dir),
        trust_remote_code=True,
    )
    processor.save_pretrained(str(output_dir))
    
    # Copy other necessary files from base model
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    
    for file_name in files_to_copy:
        src_file = base_model_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, output_dir / file_name)
    
    print("✓ Processor and tokenizer files copied")
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Merged model saved to: {output_dir}")
    print(f"\nYou can now use this model for inference:")
    print(f"  python scripts/infer.py \\")
    print(f"    --model {output_dir} \\")
    print(f"    --input-dir data/CROHME/prompts \\")
    print(f"    --output-dir data/CROHME/results")
    print("=" * 60)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter from checkpoint into base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python scripts/merge_checkpoint.py \\
    --base-model ./Uni-MuMER-Qwen2.5-VL-3B \\
    --checkpoint saves/qwen2.5_vl-3b/qlora/sft/standred/uni-mumer_qlora/checkpoint-1 \\
    --output ./Uni-MuMER-Qwen2.5-VL-3B-checkpoint1

Tham số:
  --base-model: Thư mục chứa base model (có model-00001-of-00002.safetensors, model-00002-of-00002.safetensors)
  --checkpoint: Thư mục checkpoint chứa adapter_model.safetensors
  --output: Thư mục lưu model đã merge
        """,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Thư mục chứa base model (ví dụ: ./Uni-MuMER-Qwen2.5-VL-3B)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Thư mục checkpoint chứa adapter (ví dụ: saves/.../checkpoint-1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Thư mục lưu model đã merge (ví dụ: ./Uni-MuMER-Qwen2.5-VL-3B-checkpoint1)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    try:
        merge_checkpoint(
            base_model_dir=args.base_model,
            checkpoint_dir=args.checkpoint,
            output_dir=args.output,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()

