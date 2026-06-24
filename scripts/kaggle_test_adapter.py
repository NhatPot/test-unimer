#!/usr/bin/env python3
"""
Kaggle Test Adapter - Convert backup JSON và test model với adapter.

Chức năng:
1. Convert backup JSON format sang prompts format (giống tác giả)
2. Load base model + LoRA adapter
3. Inference với vLLM (prefer) hoặc Transformers (fallback)
4. Tự động gọi eval_metrics_calculator.py để tính metrics
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def convert_backup_to_prompts(
    backup_json_path: str | Path,
    output_prompts_path: str | Path,
    project_dir: str | Path,
) -> Path:
    """
    Convert backup JSON format to prompts format.

    Input (backup):
        [{"image": "relative/path.png", "latex": "x^2"}]

    Output (prompts):
        [{
          "images": ["absolute/path.png"],
          "messages": [
            {"from": "human", "value": "<image>Prompt..."},
            {"from": "gpt", "value": "x^2"}
          ]
        }]

    Args:
        backup_json_path: Path to backup JSON file
        output_prompts_path: Path to output prompts JSON file
        project_dir: Project root directory for resolving image paths

    Returns:
        Path to created prompts file
    """
    backup_path = Path(backup_json_path)
    output_path = Path(output_prompts_path)
    project_dir = Path(project_dir)

    LOGGER.info(f"Converting {backup_path} to prompts format...")

    with backup_path.open(encoding="utf-8") as f:
        backup_data = json.load(f)

    prompts_data = []
    prompt_text = (
        "I have an image of a handwritten mathematical expression. "
        "Please write out the expression of the formula in the image using LaTeX format."
    )

    for item in backup_data:
        # Resolve image path
        image_path = item.get("image", "")
        if not Path(image_path).is_absolute():
            # Convert relative path to absolute
            image_path = str(project_dir / image_path)

        prompts_data.append({
            "images": [image_path],
            "messages": [
                {
                    "from": "human",
                    "value": f"<image>{prompt_text}"
                },
                {
                    "from": "gpt",
                    "value": item.get("latex", "")
                }
            ]
        })

    # Save prompts file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"✓ Converted {len(prompts_data)} samples → {output_path}")
    return output_path


def check_vllm_available() -> bool:
    """Check if vLLM is available."""
    try:
        import vllm
        LOGGER.info("✓ vLLM is available")
        return True
    except ImportError:
        LOGGER.warning("✗ vLLM not available, will use Transformers fallback")
        return False


def inference_with_vllm(
    model_path: str,
    adapter_path: Optional[str],
    prompts_path: str | Path,
    output_path: str | Path,
    batch_size: int = 4,
) -> Path:
    """
    Run inference using vLLM.

    Uses the existing scripts/vllm_infer.py with modifications for adapter support.

    Args:
        model_path: Base model path or HF repo
        adapter_path: LoRA adapter path (optional for merged models)
        prompts_path: Input prompts JSON file
        output_path: Output predictions JSON file
        batch_size: Batch size for inference

    Returns:
        Path to predictions file
    """
    LOGGER.info("Running inference with vLLM...")

    # For now, vLLM will use the model directly
    # If adapter_path is provided, we need to use it as merged model or load adapter
    # vLLM supports LoRA via enable_lora=True

    # Import vllm_infer script
    sys.path.insert(0, str(Path(__file__).parent))
    from vllm_infer import run_inference

    # vllm_infer expects input_dir with JSON files, not single file
    # So we need to create a temp directory with the prompts file
    prompts_path = Path(prompts_path)
    output_path = Path(output_path)

    # Create temp input dir
    temp_input_dir = prompts_path.parent / "temp_prompts"
    temp_input_dir.mkdir(exist_ok=True)

    # Copy or symlink prompts file to temp dir
    temp_prompts_file = temp_input_dir / prompts_path.name
    if not temp_prompts_file.exists():
        import shutil
        shutil.copy(prompts_path, temp_prompts_file)

    # Run inference
    # Note: vllm_infer.py doesn't support adapter loading yet
    # We need to use merged model or modify vllm_infer.py
    LOGGER.warning(
        "vLLM inference with adapter is not fully supported. "
        "Please merge adapter first or use Transformers fallback."
    )

    # For now, we'll use Transformers fallback
    return inference_with_transformers(
        model_path=model_path,
        adapter_path=adapter_path,
        prompts_path=prompts_path,
        output_path=output_path,
        batch_size=batch_size,
    )


def inference_with_transformers(
    model_path: str,
    adapter_path: Optional[str],
    prompts_path: str | Path,
    output_path: str | Path,
    batch_size: int = 4,
) -> Path:
    """
    Run inference using Transformers + PEFT (fallback).

    Args:
        model_path: Base model path or HF repo
        adapter_path: LoRA adapter path
        prompts_path: Input prompts JSON file
        output_path: Output predictions JSON file
        batch_size: Batch size for inference

    Returns:
        Path to predictions file
    """
    LOGGER.info("Running inference with Transformers + PEFT...")

    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    # Import correct model class for Qwen2.5-VL
    try:
        from transformers import Qwen2VLForConditionalGeneration
        LOGGER.info("Using Qwen2VLForConditionalGeneration")
    except ImportError:
        # Fallback to AutoModelForVision2Seq or AutoModel
        from transformers import AutoModel as Qwen2VLForConditionalGeneration
        LOGGER.warning("Qwen2VLForConditionalGeneration not found, using AutoModel")

    # Load base model with 4-bit quantization
    LOGGER.info(f"Loading base model: {model_path}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load adapter if provided
    if adapter_path:
        LOGGER.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Load prompts
    prompts_path = Path(prompts_path)
    with prompts_path.open(encoding="utf-8") as f:
        prompts_data = json.load(f)

    LOGGER.info(f"Loaded {len(prompts_data)} samples from {prompts_path}")

    # Inference
    predictions = []

    for i in tqdm(range(0, len(prompts_data), batch_size), desc="Inference"):
        batch = prompts_data[i:i+batch_size]

        # Process each sample in batch (vLM models need special handling)
        for item in batch:
            try:
                # Get image path and load image
                image_path = item["images"][0]
                image = Image.open(image_path).convert("RGB")

                # Get prompt text
                prompt_text = None
                gt_text = None
                for msg in item["messages"]:
                    if msg["from"] == "human":
                        # Remove <image> tag as processor will handle it
                        prompt_text = msg["value"].replace("<image>", "").strip()
                    elif msg["from"] == "gpt":
                        gt_text = msg["value"]

                if not prompt_text or gt_text is None:
                    continue

                # Prepare messages for processor
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]}
                ]

                # Apply chat template
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Prepare inputs
                inputs = processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.2,
                        top_p=0.8,
                        do_sample=True,
                    )

                # Decode
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                pred_text = processor.decode(
                    generated_ids,
                    skip_special_tokens=True
                ).strip()

                # Save prediction
                predictions.append({
                    "gt": gt_text,
                    "pred": pred_text,
                    "image_path": image_path,
                    "img_id": Path(image_path).stem,
                })

            except Exception as e:
                LOGGER.error(f"Error processing {item.get('images', ['unknown'])[0]}: {e}")
                # Add empty prediction to maintain alignment
                predictions.append({
                    "gt": item["messages"][1]["value"] if len(item["messages"]) > 1 else "",
                    "pred": "",
                    "image_path": item["images"][0],
                    "img_id": Path(item["images"][0]).stem,
                })
                continue

    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"✓ Saved {len(predictions)} predictions → {output_path}")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return output_path


def evaluate(pred_json_path: str | Path, output_txt_path: str | Path) -> Dict:
    """
    Evaluate predictions using eval_metrics_calculator.py.

    Args:
        pred_json_path: Path to predictions JSON file
        output_txt_path: Path to output results text file

    Returns:
        Dictionary with metrics
    """
    LOGGER.info(f"Evaluating predictions: {pred_json_path}")

    # Import eval_metrics_calculator
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_metrics_calculator import evaluate_text_generation

    # Run evaluation
    metrics = evaluate_text_generation(
        json_path=str(pred_json_path),
        output_path=str(output_txt_path)
    )

    LOGGER.info(f"✓ Evaluation complete → {output_txt_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test model with adapter on Kaggle (convert + inference + eval)"
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model path or HF repo (e.g., Qwen/Qwen2.5-VL-3B-Instruct)"
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="LoRA adapter path (optional if using merged model)"
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to backup JSON test data"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for prompts, predictions, and results"
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project root directory (auto-detected if not provided)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Force use vLLM (fail if not available)"
    )

    args = parser.parse_args()

    # Auto-detect project dir
    if args.project_dir is None:
        args.project_dir = Path(__file__).resolve().parents[1]

    test_data_path = Path(args.test_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset name from test data filename
    dataset_name = test_data_path.stem

    LOGGER.info("="*60)
    LOGGER.info(f"Testing dataset: {dataset_name}")
    LOGGER.info("="*60)

    # Step 1: Convert backup to prompts
    prompts_path = output_dir / f"{dataset_name}_prompts.json"
    convert_backup_to_prompts(
        backup_json_path=test_data_path,
        output_prompts_path=prompts_path,
        project_dir=args.project_dir,
    )

    # Step 2: Inference
    pred_path = output_dir / f"{dataset_name}_pred.json"

    if args.use_vllm:
        if not check_vllm_available():
            raise RuntimeError("vLLM requested but not available")
        inference_with_vllm(
            model_path=args.base_model,
            adapter_path=args.adapter_path,
            prompts_path=prompts_path,
            output_path=pred_path,
            batch_size=args.batch_size,
        )
    else:
        # Use Transformers by default
        inference_with_transformers(
            model_path=args.base_model,
            adapter_path=args.adapter_path,
            prompts_path=prompts_path,
            output_path=pred_path,
            batch_size=args.batch_size,
        )

    # Step 3: Evaluate
    results_path = output_dir / f"{dataset_name}_results.txt"
    evaluate(
        pred_json_path=pred_path,
        output_txt_path=results_path,
    )

    LOGGER.info("="*60)
    LOGGER.info("✓ Test complete!")
    LOGGER.info(f"  Predictions: {pred_path}")
    LOGGER.info(f"  Results: {results_path}")
    LOGGER.info("="*60)


if __name__ == "__main__":
    main()
