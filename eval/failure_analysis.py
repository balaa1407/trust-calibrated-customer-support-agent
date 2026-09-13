"""
Failure Analysis.
Extracts the worst-performing examples and helps cluster them into failure modes.
"""
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR


def extract_worst_failures(results_jsonl: Path, n_examples: int = 20) -> pd.DataFrame:
    """Extract worst examples based on quality score and intent accuracy."""
    try:
        df = pd.read_json(results_jsonl, lines=True)
    except Exception as e:
        print(f"Error loading {results_jsonl}: {e}")
        return pd.DataFrame()
        
    if len(df) == 0:
        return df

    # We want examples where intent was wrong OR quality score was low (< 3.0)
    # Give priority to examples that failed both
    df['failure_score'] = 0.0
    
    if 'intent_correct' in df.columns:
        df['failure_score'] += (~df['intent_correct']).astype(float) * 2.0
        
    if 'llm_overall_score' in df.columns:
        # Lower score = higher failure
        df['failure_score'] += (5.0 - df['llm_overall_score'])
        
    # Sort by failure score descending
    worst = df.sort_values('failure_score', ascending=False).head(n_examples)
    
    # Save to CSV for manual review
    out_path = RESULTS_DIR / "worst_failures_for_review.csv"
    
    cols_to_export = ['thread_id', 'customer_msg', 'predicted_intent']
    if 'true_intent' in worst.columns: cols_to_export.append('true_intent')
    if 'llm_overall_score' in worst.columns: cols_to_export.append('llm_overall_score')
    if 'draft_reply' in worst.columns: cols_to_export.append('draft_reply')
    if 'confidence' in worst.columns: cols_to_export.append('confidence')
    
    # Keep only columns that exist
    export_df = worst[[c for c in cols_to_export if c in worst.columns]]
    export_df.to_csv(out_path, index=False)
    
    print(f"Extracted {len(worst)} worst failures to {out_path}")
    return worst


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR / "eval_results.jsonl"))
    args = parser.parse_args()
    
    extract_worst_failures(Path(args.results))
