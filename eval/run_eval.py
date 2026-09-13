"""
Master Evaluation Script.
Runs the entire pipeline on the golden set, computes metrics, and generates plots.
"""
import sys
import json
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR, DEFAULT_ESCALATION_THRESHOLD
from pipeline import SupportAgent
from eval.metrics import evaluate_intents, evaluate_retrieval, evaluate_escalation
from eval.llm_judge import evaluate_reply
from escalation.calibration import run_calibration_analysis


def run_full_eval(brand_id: str, golden_set_csv: Path, subsample: int = None):
    print(f"\n{'='*60}")
    print(f"RUNNING FULL EVALUATION FOR {brand_id}")
    print(f"{'='*60}")
    
    # Load Golden Set
    df = pd.read_csv(golden_set_csv)
    if subsample and subsample < len(df):
        print(f"Subsampling to {subsample} examples for speed...")
        # Make sure we don't just take the easy ones
        df = df.sample(subsample, random_state=42)
        
    records = df.to_dict('records')
    
    agent = SupportAgent(brand_id)
    
    results = []
    from tqdm import tqdm
    for row in tqdm(records, desc="Processing Pipeline"):
        msg = row['customer_msg']
        
        try:
            # 1. Run Pipeline
            pip_res = agent.handle_message(msg)
            
            # 2. Run LLM Judge
            judge_res = evaluate_reply(
                customer_msg=msg,
                draft_reply=pip_res['draft']['reply'],
                precedents=pip_res['precedents'],
                brand_id=brand_id
            )
            
            results.append({
                'thread_id': row['id'],
                'customer_msg': msg,
                'true_intent': row.get('true_intent', ''),
                'predicted_intent': pip_res['intent']['intent'],
                'intent_confidence': pip_res['intent']['confidence'],
                
                'retrieved_ids': [p['id'] for p in pip_res['precedents']],
                'actual_brand_reply': row.get('actual_brand_reply', ''),
                
                'draft_reply': pip_res['draft']['reply'],
                'llm_overall_score': judge_res['overall_score'],
                'judge_details': judge_res,
                
                'confidence': pip_res['escalation']['confidence'],
                'should_escalate_pred': pip_res['escalation']['should_escalate'],
                'should_escalate_true': row.get('should_escalate_y_n', ''),
            })
        except Exception as e:
            print(f"Error processing thread {row['id']}: {e}")
            
    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_results.jsonl"
    with open(out_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
            
    print(f"\n  Processed {len(results)} examples. Computing metrics...")
    
    # Calculate Metrics
    # 1. Intent Accuracy
    true_intents = [r['true_intent'] for r in results]
    pred_intents = [r['predicted_intent'] for r in results]
    intent_metrics = evaluate_intents(true_intents, pred_intents)
    if intent_metrics:
        print(f"\nIntent Accuracy: {intent_metrics['accuracy']:.1%}")
        print(f"Intent Macro-F1: {intent_metrics['f1_macro']:.3f}")
        
    # 2. Reply Quality (Judge)
    avg_quality = sum(r['llm_overall_score'] for r in results) / len(results)
    print(f"\nAverage LLM Judge Quality: {avg_quality:.2f} / 5.0")
    
    # 3. Calibration & Risk Coverage
    # Add is_correct flag for calibration analysis
    for r in results:
        r['is_correct'] = (str(r['true_intent']).strip().lower() == str(r['predicted_intent']).strip().lower())
        r['quality_score'] = r['llm_overall_score'] / 5.0  # Normalize to 0-1
        
    calib_analysis = run_calibration_analysis(eval_results=results)
    
    print("\n  Evaluation complete. See eval/results/ for details and plots.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    parser.add_argument("--golden", required=True, help="Path to labeled_golden_set.csv")
    parser.add_argument("--subsample", type=int, default=None)
    args = parser.parse_args()
    
    run_full_eval(args.brand, Path(args.golden), args.subsample)
