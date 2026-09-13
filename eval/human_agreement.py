"""
Human-LLM Judge Agreement Analysis.

Calculates Cohen's kappa between human scores and LLM judge scores.
Extracts examples where they disagree.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_agreement(human_scores: list[float], llm_scores: list[float]) -> dict:
    """
    Compare human 1-5 scores against LLM 1-5 scores.
    """
    if not human_scores or not llm_scores or len(human_scores) != len(llm_scores):
        return {}
        
    # Round LLM scores to nearest integer for kappa
    y_true = np.round(human_scores).astype(int)
    y_pred = np.round(llm_scores).astype(int)
    
    # Exact agreement
    exact_match = np.mean(y_true == y_pred)
    
    # Within 1 point agreement
    within_one = np.mean(np.abs(y_true - y_pred) <= 1)
    
    # Cohen's Kappa (quadratic weighted to penalize larger disagreements more)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Mean error (shows if LLM systematically over/under scores)
    mean_error = np.mean(y_pred - y_true)
    
    return {
        'exact_agreement': float(exact_match),
        'within_one_point': float(within_one),
        'kappa_quadratic': float(kappa),
        'mean_error': float(mean_error),  # >0 means LLM scores higher than human
        'n_samples': len(y_true)
    }


def get_disagreement_examples(results_df: pd.DataFrame, diff_threshold: float = 1.5) -> pd.DataFrame:
    """
    Extract examples where human and LLM score differ by more than threshold.
    """
    if 'human_quality' not in results_df.columns or 'llm_quality' not in results_df.columns:
        return pd.DataFrame()
        
    # Drop rows missing human labels
    df = results_df.dropna(subset=['human_quality']).copy()
    
    # Calculate difference
    df['score_diff'] = df['llm_quality'] - df['human_quality']
    df['abs_diff'] = df['score_diff'].abs()
    
    # Filter for significant disagreements
    disagreements = df[df['abs_diff'] >= diff_threshold].copy()
    
    # Sort by absolute difference
    disagreements = disagreements.sort_values('abs_diff', ascending=False)
    
    return disagreements
