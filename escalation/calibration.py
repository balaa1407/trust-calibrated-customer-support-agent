"""
Risk-coverage curve analysis — the headline artifact.

Sweeps confidence thresholds to answer:
"At threshold T, we auto-handle X% of volume with Y% quality."

This is what makes the escalation decision commercially meaningful:
it's not just "escalate or not" but "here's the trade-off you're choosing."

Usage: python escalation/calibration.py --results PATH --output PATH
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR


def compute_risk_coverage(eval_results: list[dict], quality_key: str = 'quality_score') -> dict:
    """
    Compute risk-coverage curve from evaluation results.

    Each eval_result should have:
      - confidence: float [0,1]
      - quality_score: float (or whatever quality_key points to)
      - is_correct: bool (for intent accuracy)

    Returns dict with arrays for plotting.
    """
    if not eval_results:
        return {}

    confidences = np.array([r['confidence'] for r in eval_results])
    qualities = np.array([r.get(quality_key, 0) for r in eval_results])

    thresholds = np.linspace(0, 1, 51)  # 0.00, 0.02, 0.04, ..., 1.00
    coverages = []
    avg_qualities = []
    accuracies = []

    for t in thresholds:
        # Auto-handle = confidence >= threshold
        mask = confidences >= t
        coverage = mask.sum() / len(confidences) if len(confidences) > 0 else 0
        coverages.append(coverage)

        if mask.sum() > 0:
            avg_q = qualities[mask].mean()
            # Intent accuracy among auto-handled
            correct = np.array([r.get('is_correct', True) for r in eval_results])
            acc = correct[mask].mean()
        else:
            avg_q = 0
            acc = 0
        avg_qualities.append(avg_q)
        accuracies.append(acc)

    return {
        'thresholds': thresholds.tolist(),
        'coverages': coverages,
        'avg_qualities': avg_qualities,
        'accuracies': accuracies,
    }


def find_operating_point(curve_data: dict, quality_floor: float = 0.8) -> dict:
    """
    Find the optimal operating point: maximum coverage while maintaining quality above floor.
    """
    thresholds = curve_data['thresholds']
    coverages = curve_data['coverages']
    avg_qualities = curve_data['avg_qualities']

    best_threshold = 1.0
    best_coverage = 0.0
    best_quality = 0.0

    for t, cov, q in zip(thresholds, coverages, avg_qualities):
        if q >= quality_floor and cov > best_coverage:
            best_threshold = t
            best_coverage = cov
            best_quality = q

    return {
        'threshold': best_threshold,
        'coverage': best_coverage,
        'quality': best_quality,
        'quality_floor': quality_floor,
    }


def plot_risk_coverage(curve_data: dict, operating_point: dict = None,
                       output_path: Path = None, title: str = "Risk-Coverage Curve"):
    """
    Plot the risk-coverage curve — the headline visualization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    thresholds = curve_data['thresholds']
    coverages = curve_data['coverages']
    avg_qualities = curve_data['avg_qualities']
    accuracies = curve_data['accuracies']

    # Plot 1: Coverage vs Quality
    ax1.plot(coverages, avg_qualities, 'b-', linewidth=2, label='Avg Quality')
    ax1.plot(coverages, accuracies, 'g--', linewidth=2, label='Intent Accuracy')
    if operating_point:
        ax1.axhline(y=operating_point['quality_floor'], color='r', linestyle=':',
                     alpha=0.7, label=f"Quality Floor ({operating_point['quality_floor']:.0%})")
        ax1.plot(operating_point['coverage'], operating_point['quality'], 'r*',
                 markersize=15, label=f"Operating Point (cov={operating_point['coverage']:.0%})")
    ax1.set_xlabel('Coverage (% auto-handled)', fontsize=12)
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_title('Coverage vs Quality Trade-off', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Threshold vs Coverage + Quality
    ax2.plot(thresholds, coverages, 'b-', linewidth=2, label='Coverage')
    ax2.plot(thresholds, avg_qualities, 'orange', linewidth=2, label='Avg Quality')
    if operating_point:
        ax2.axvline(x=operating_point['threshold'], color='r', linestyle=':',
                     alpha=0.7, label=f"Threshold = {operating_point['threshold']:.2f}")
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Threshold Sweep', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = RESULTS_DIR / "risk_coverage_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved risk-coverage curve to {output_path}")
    return output_path


def run_calibration_analysis(eval_results_path: Path = None, eval_results: list = None):
    """Full calibration analysis from eval results."""
    if eval_results is None:
        if eval_results_path is None:
            eval_results_path = RESULTS_DIR / "full_eval_results.jsonl"
        eval_results = []
        with open(eval_results_path, 'r') as f:
            for line in f:
                eval_results.append(json.loads(line))

    print(f"\nCalibration analysis on {len(eval_results)} examples...")

    curve = compute_risk_coverage(eval_results)
    op_80 = find_operating_point(curve, quality_floor=0.8)
    op_90 = find_operating_point(curve, quality_floor=0.9)

    print(f"\nOperating Points:")
    print(f"  At 80% quality floor: threshold={op_80['threshold']:.2f}, "
          f"coverage={op_80['coverage']:.1%}, quality={op_80['quality']:.3f}")
    print(f"  At 90% quality floor: threshold={op_90['threshold']:.2f}, "
          f"coverage={op_90['coverage']:.1%}, quality={op_90['quality']:.3f}")

    plot_path = plot_risk_coverage(curve, op_80,
                                   title="Trust Calibration: Risk-Coverage Analysis")

    # Save analysis
    analysis = {
        'n_examples': len(eval_results),
        'operating_point_80': op_80,
        'operating_point_90': op_90,
        'curve_data': curve,
    }
    analysis_path = RESULTS_DIR / "calibration_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved analysis to {analysis_path}")

    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, help="Path to eval results JSONL")
    args = parser.parse_args()

    path = Path(args.results) if args.results else None
    run_calibration_analysis(path)
