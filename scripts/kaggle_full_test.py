#!/usr/bin/env python3
"""
Kaggle Full Test - Test model trên tất cả 3 CROHME datasets và so sánh với base model.

Chức năng:
1. Loop qua 3 CROHME datasets (2014, 2016, 2019)
2. Gọi kaggle_test_adapter.py cho mỗi dataset
3. Parse kết quả base model (nếu có)
4. Tạo bảng so sánh chi tiết
"""

import argparse
import json
import logging
import re
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


def parse_base_results(base_results_file: Path) -> Optional[Dict]:
    """
    Parse base model results file to extract metrics.

    Args:
        base_results_file: Path to base model results text file

    Returns:
        Dictionary with metrics or None if file not found
    """
    if not base_results_file.exists():
        LOGGER.warning(f"Base results not found: {base_results_file}")
        return None

    with base_results_file.open(encoding="utf-8") as f:
        content = f.read()

    # Extract metrics using regex
    metrics = {}

    # Mean Edit Score
    match = re.search(r'Mean Edit Score:\s+([\d.]+)%', content)
    if match:
        metrics['mean_edit_score'] = float(match.group(1))

    # BLEU-4 Score
    match = re.search(r'BLEU-4 Score:\s+([\d.]+)%', content)
    if match:
        metrics['bleu4'] = float(match.group(1))

    # Character Error Rate
    match = re.search(r'Character Error Rate:\s+([\d.]+)', content)
    if match:
        metrics['cer'] = float(match.group(1))

    # Exact Match Rate
    match = re.search(r'Exact Match Rate:\s+([\d.]+)%', content)
    if match:
        metrics['exact_match'] = float(match.group(1))

    # Error ≤ 1
    match = re.search(r'Error ≤ 1:\s+([\d.]+)%', content)
    if match:
        metrics['error_le_1'] = float(match.group(1))

    # Error ≤ 3
    match = re.search(r'Error ≤ 3:\s+([\d.]+)%', content)
    if match:
        metrics['error_le_3'] = float(match.group(1))

    if not metrics:
        LOGGER.warning(f"No metrics found in {base_results_file}")
        return None

    return metrics


def parse_finetuned_results(results_file: Path) -> Optional[Dict]:
    """
    Parse fine-tuned model results file to extract metrics.

    Args:
        results_file: Path to fine-tuned model results text file

    Returns:
        Dictionary with metrics or None if file not found
    """
    return parse_base_results(results_file)  # Same format


def create_comparison_table(
    all_results: List[Dict],
    output_file: Path,
    paper_results: Optional[Dict] = None,
) -> None:
    """
    Create comparison table between base and fine-tuned models.

    Args:
        all_results: List of result dictionaries for each dataset
        output_file: Path to output comparison table
        paper_results: Optional paper benchmark results
    """
    lines = []
    lines.append("=" * 80)
    lines.append("         COMPARISON: Base Model vs Fine-tuned Model vs Paper")
    lines.append("=" * 80)

    # Per-dataset comparison
    for result in all_results:
        dataset = result['dataset']
        base = result.get('base')
        ft = result.get('finetuned')
        paper = paper_results.get(dataset) if paper_results else None

        lines.append("")
        lines.append(f"Dataset: {dataset.upper()} ({result.get('samples', 'N/A')} samples)")
        lines.append("-" * 80)

        if base and ft:
            lines.append(f"{'Metric':<30} {'Base':<12} {'Fine-tuned':<12} {'Delta':<10} {'Paper':<10}")
        elif ft:
            lines.append(f"{'Metric':<30} {'Fine-tuned':<12} {'Paper':<10}")
        else:
            continue

        lines.append("-" * 80)

        # Mean Edit Score
        if ft:
            ft_mes = ft.get('mean_edit_score', 0)
            if base:
                base_mes = base.get('mean_edit_score', 0)
                delta = ft_mes - base_mes
                paper_mes = paper.get('mean_edit_score', 'N/A') if paper else 'N/A'
                lines.append(
                    f"{'Mean Edit Score':<30} "
                    f"{base_mes:>6.2f}%     "
                    f"{ft_mes:>6.2f}%     "
                    f"{delta:>+6.2f}%   "
                    f"{paper_mes if isinstance(paper_mes, str) else f'{paper_mes:.2f}%':>10}"
                )
            else:
                paper_mes = paper.get('mean_edit_score', 'N/A') if paper else 'N/A'
                lines.append(
                    f"{'Mean Edit Score':<30} "
                    f"{ft_mes:>6.2f}%     "
                    f"{paper_mes if isinstance(paper_mes, str) else f'{paper_mes:.2f}%':>10}"
                )

        # BLEU-4 Score
        if ft:
            ft_bleu = ft.get('bleu4', 0)
            if base:
                base_bleu = base.get('bleu4', 0)
                delta = ft_bleu - base_bleu
                paper_bleu = paper.get('bleu4', 'N/A') if paper else 'N/A'
                lines.append(
                    f"{'BLEU-4 Score':<30} "
                    f"{base_bleu:>6.2f}%     "
                    f"{ft_bleu:>6.2f}%     "
                    f"{delta:>+6.2f}%   "
                    f"{paper_bleu if isinstance(paper_bleu, str) else f'{paper_bleu:.2f}%':>10}"
                )
            else:
                paper_bleu = paper.get('bleu4', 'N/A') if paper else 'N/A'
                lines.append(
                    f"{'BLEU-4 Score':<30} "
                    f"{ft_bleu:>6.2f}%     "
                    f"{paper_bleu if isinstance(paper_bleu, str) else f'{paper_bleu:.2f}%':>10}"
                )

        # Character Error Rate
        if ft:
            ft_cer = ft.get('cer', 0)
            if base:
                base_cer = base.get('cer', 0)
                delta = ft_cer - base_cer
                paper_cer = paper.get('cer', 'N/A') if paper else 'N/A'
                lines.append(
                    f"{'Character Error Rate':<30} "
                    f"{base_cer:>8.4f}   "
                    f"{ft_cer:>8.4f}   "
                    f"{delta:>+8.4f} "
                    f"{paper_cer if isinstance(paper_cer, str) else f'{paper_cer:.4f}':>10}"
                )
            else:
                paper_cer = paper.get('cer', 'N/A') if paper else 'N/A'
                lines.append(
                    f"{'Character Error Rate':<30} "
                    f"{ft_cer:>8.4f}   "
                    f"{paper_cer if isinstance(paper_cer, str) else f'{paper_cer:.4f}':>10}"
                )

        # Exact Match Rate
        if ft:
            ft_em = ft.get('exact_match', 0)
            if base:
                base_em = base.get('exact_match', 0)
                delta = ft_em - base_em
                paper_em = paper.get('exact_match', 'N/A') if paper else 'N/A'
                lines.append(
                    f"{'Exact Match Rate':<30} "
                    f"{base_em:>6.2f}%     "
                    f"{ft_em:>6.2f}%     "
                    f"{delta:>+6.2f}%   "
                    f"{paper_em if isinstance(paper_em, str) else f'{paper_em:.2f}%':>10}"
                )
            else:
                paper_em = paper.get('exact_match', 'N/A') if paper else 'N/A'
                lines.append(
                    f"{'Exact Match Rate':<30} "
                    f"{ft_em:>6.2f}%     "
                    f"{paper_em if isinstance(paper_em, str) else f'{paper_em:.2f}%':>10}"
                )

    # Overall summary
    lines.append("")
    lines.append("=" * 80)
    lines.append("OVERALL SUMMARY (All 3 CROHME datasets)")
    lines.append("=" * 80)

    # Calculate averages
    avg_metrics = {'base': {}, 'finetuned': {}, 'paper': {}}

    for metric in ['mean_edit_score', 'bleu4', 'cer', 'exact_match']:
        # Fine-tuned average
        ft_values = [r['finetuned'][metric] for r in all_results if 'finetuned' in r and metric in r['finetuned']]
        if ft_values:
            avg_metrics['finetuned'][metric] = sum(ft_values) / len(ft_values)

        # Base average
        base_values = [r['base'][metric] for r in all_results if 'base' in r and r['base'] and metric in r['base']]
        if base_values:
            avg_metrics['base'][metric] = sum(base_values) / len(base_values)

        # Paper average (if provided)
        if paper_results:
            paper_values = [paper_results[r['dataset']][metric] for r in all_results if r['dataset'] in paper_results and metric in paper_results[r['dataset']]]
            if paper_values:
                avg_metrics['paper'][metric] = sum(paper_values) / len(paper_values)

    lines.append("")
    if avg_metrics['base'] and avg_metrics['finetuned']:
        lines.append(f"{'Metric':<30} {'Base (avg)':<12} {'Fine-tuned (avg)':<16} {'Delta':<10} {'Paper (avg)':<10}")
    elif avg_metrics['finetuned']:
        lines.append(f"{'Metric':<30} {'Fine-tuned (avg)':<16} {'Paper (avg)':<10}")
    lines.append("-" * 80)

    for metric, label in [
        ('mean_edit_score', 'Mean Edit Score'),
        ('bleu4', 'BLEU-4 Score'),
        ('cer', 'Character Error Rate'),
        ('exact_match', 'Exact Match Rate'),
    ]:
        ft_val = avg_metrics['finetuned'].get(metric)
        base_val = avg_metrics['base'].get(metric)
        paper_val = avg_metrics['paper'].get(metric)

        if ft_val is not None:
            if base_val is not None:
                delta = ft_val - base_val
                if metric == 'cer':
                    lines.append(
                        f"{label:<30} {base_val:>8.4f}   {ft_val:>8.4f}        "
                        f"{delta:>+8.4f} {paper_val if paper_val else 'N/A':>10}"
                    )
                else:
                    lines.append(
                        f"{label:<30} {base_val:>6.2f}%     {ft_val:>6.2f}%        "
                        f"{delta:>+6.2f}%   {f'{paper_val:.2f}%' if paper_val else 'N/A':>10}"
                    )
            else:
                if metric == 'cer':
                    lines.append(f"{label:<30} {ft_val:>8.4f}        {f'{paper_val:.4f}' if paper_val else 'N/A':>10}")
                else:
                    lines.append(f"{label:<30} {ft_val:>6.2f}%        {f'{paper_val:.2f}%' if paper_val else 'N/A':>10}")

    lines.append("")
    lines.append("=" * 80)

    # Conclusion
    if avg_metrics['base'] and avg_metrics['finetuned']:
        improvement = avg_metrics['finetuned'].get('mean_edit_score', 0) - avg_metrics['base'].get('mean_edit_score', 0)
        lines.append(f"Conclusion: Fine-tuned model shows {improvement:+.2f}% improvement")
        lines.append(f"            in Mean Edit Score compared to base model.")
    elif avg_metrics['finetuned'] and avg_metrics['paper']:
        paper_avg = avg_metrics['paper'].get('mean_edit_score', 0)
        ft_avg = avg_metrics['finetuned'].get('mean_edit_score', 0)
        diff = ft_avg - paper_avg
        lines.append(f"Comparison with paper: {diff:+.2f}% difference in Mean Edit Score")

    lines.append("=" * 80)

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    LOGGER.info(f"✓ Comparison table saved → {output_file}")

    # Also print to console
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Full test on 3 CROHME datasets with comparison"
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model path or HF repo"
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="LoRA adapter checkpoint path"
    )
    parser.add_argument(
        "--test-datasets",
        nargs="+",
        default=["crohme_2014", "crohme_2016", "crohme_2019"],
        help="List of dataset names to test"
    )
    parser.add_argument(
        "--backup-dir",
        default="example_data/backup",
        help="Directory containing backup JSON files"
    )
    parser.add_argument(
        "--base-results-dir",
        default="example_data/CROHME/results",
        help="Directory containing base model results (optional)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for all results"
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project root directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference"
    )

    args = parser.parse_args()

    # Auto-detect project dir
    if args.project_dir is None:
        args.project_dir = Path(__file__).resolve().parents[1]

    project_dir = Path(args.project_dir)
    backup_dir = project_dir / args.backup_dir
    base_results_dir = project_dir / args.base_results_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paper benchmark results
    paper_results = {
        'crohme_2014': {
            'mean_edit_score': 96.31,
            'bleu4': 91.92,
            'cer': 0.0273,
            'exact_match': 82.05,
        },
        'crohme_2016': {
            'mean_edit_score': 96.35,
            'bleu4': 93.76,
            'cer': 0.0150,
            'exact_match': 77.94,
        },
        'crohme_2019': {
            'mean_edit_score': 96.74,
            'bleu4': 94.91,
            'cer': 0.0127,
            'exact_match': 79.23,
        },
    }

    # Sample counts
    dataset_samples = {
        'crohme_2014': 986,
        'crohme_2016': 1147,
        'crohme_2019': 1199,
    }

    all_results = []

    LOGGER.info("=" * 80)
    LOGGER.info("Starting full benchmark test on 3 CROHME datasets")
    LOGGER.info("=" * 80)

    # Loop through datasets
    for dataset in args.test_datasets:
        LOGGER.info("")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Testing {dataset.upper()}...")
        LOGGER.info("=" * 80)

        test_data_path = backup_dir / f"{dataset}.json"
        if not test_data_path.exists():
            LOGGER.error(f"Test data not found: {test_data_path}")
            continue

        # Run kaggle_test_adapter.py
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "kaggle_test_adapter.py"),
            "--base-model", args.base_model,
            "--adapter-path", args.adapter_path,
            "--test-data", str(test_data_path),
            "--output-dir", str(output_dir),
            "--project-dir", str(project_dir),
            "--batch-size", str(args.batch_size),
        ]

        LOGGER.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            LOGGER.error(f"Failed to test {dataset}")
            continue

        # Parse results
        finetuned_results = parse_finetuned_results(output_dir / f"{dataset}_results.txt")
        base_results = parse_base_results(base_results_dir / f"{dataset}_results.txt")

        all_results.append({
            'dataset': dataset,
            'samples': dataset_samples.get(dataset, 'N/A'),
            'base': base_results,
            'finetuned': finetuned_results,
        })

    # Create comparison table
    if all_results:
        comparison_file = output_dir / "comparison_table.txt"
        create_comparison_table(
            all_results=all_results,
            output_file=comparison_file,
            paper_results=paper_results,
        )

        LOGGER.info("")
        LOGGER.info("=" * 80)
        LOGGER.info("✓ Full benchmark test complete!")
        LOGGER.info(f"  Results directory: {output_dir}")
        LOGGER.info(f"  Comparison table: {comparison_file}")
        LOGGER.info("=" * 80)
    else:
        LOGGER.error("No results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
