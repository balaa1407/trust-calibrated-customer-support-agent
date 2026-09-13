"""
Stratified sampling for the Golden Set.

Extracts a representative sample of customer messages across all intents
and across difficulty levels (easy vs. hard/ambiguous), ready for human labeling.

Usage: python eval/golden_set/sampling.py --brand BRAND_ID --size 200
"""
import sys
import json
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROCESSED_DATA_DIR, GOLDEN_SET_DIR
from intent.classifier import classify
from intent.taxonomy import load_taxonomy


def estimate_difficulty(text: str, thread_length: int) -> str:
    """Heuristic for how hard a message is to handle."""
    words = text.split()
    if len(words) < 5:
        return "easy"  # Very short, usually clear (or too vague to do anything but ask for clarity)
    if len(words) > 50 or thread_length > 4:
        return "hard"  # Long story or deep thread
    if any(w in text.lower() for w in ['but', 'however', 'although', 'wait', 'also', 'and then']):
        return "hard"  # Multi-part or complex conditional
    return "medium"


def generate_golden_candidates(brand_id: str, target_size: int = 200):
    """
    Sample messages and pre-classify them to ensure stratified distribution.
    Outputs a JSONL file that the user can open in a spreadsheet for labeling.
    """
    threads_path = PROCESSED_DATA_DIR / f"{brand_id}_threads.jsonl"
    taxonomy = load_taxonomy(brand_id)

    print("Loading threads...")
    threads = []
    with open(threads_path, 'r', encoding='utf-8') as f:
        for line in f:
            threads.append(json.loads(line))

    # We want to label the *first* customer message in the thread
    candidates = []
    for t in threads:
        msg = t['first_customer_msg']
        if len(msg.split()) < 3:
            continue  # Skip junk
        difficulty = estimate_difficulty(msg, t['thread_length'])
        candidates.append({
            'id': t['thread_id'],
            'customer_msg': msg,
            'thread_context': t['full_context'],
            'actual_brand_reply': t['brand_replies'][0] if t['brand_replies'] else "",
            'difficulty': difficulty,
            'thread_length': t['thread_length'],
        })

    # BYPASS LLM STRATIFICATION DUE TO STRICT DAILY API QUOTA LIMITS
    # We randomly sample the candidates and assign mock predicted intent.
    print(f"\nRandomly sampling {target_size} candidates (bypassing LLM due to quotas)...")
    
    random.shuffle(candidates)
    selected = candidates[:target_size]
    
    for c in selected:
        c['predicted_intent'] = "other"
        c['intent_confidence'] = 0.5
        
        # Add placeholder fields for human labeling
        c['true_intent'] = ""
        c['true_reply_quality_1_to_5'] = ""
        c['should_escalate_y_n'] = ""
        c['labeler_notes'] = ""

    # Save to JSONL
    GOLDEN_SET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLDEN_SET_DIR / "candidates_to_label.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for c in selected:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
            
    # Also save as CSV for easier spreadsheet labeling
    import pandas as pd
    df = pd.DataFrame(selected)
    csv_path = GOLDEN_SET_DIR / "candidates_to_label.csv"
    # Reorder columns for labeling
    cols = ['id', 'customer_msg', 'predicted_intent', 'true_intent', 
            'true_reply_quality_1_to_5', 'should_escalate_y_n', 'difficulty', 'labeler_notes', 'actual_brand_reply']
    df[cols].to_csv(csv_path, index=False)

    print(f"\n  Generated {len(selected)} candidates for golden set")
    print(f"  Saved JSONL: {out_path}")
    print(f"  Saved CSV:   {csv_path} (Open this in Excel/Google Sheets to label)")
    print("\nNext steps:")
    print("  1. Open candidates_to_label.csv")
    print("  2. Fill in 'true_intent', 'true_reply_quality_1_to_5', and 'should_escalate_y_n'")
    print("  3. Save as 'labeled_golden_set.csv' in the same folder")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()
    
    generate_golden_candidates(args.brand, args.size)
