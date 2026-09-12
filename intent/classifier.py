"""
Intent classifier — few-shot LLM classification with confidence scores.
Also includes baseline classifiers for comparison.

Usage:
    python intent/classifier.py --brand BRAND_ID --message "I can't log in"
    python intent/classifier.py --brand BRAND_ID --eval  # Run on golden set
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR, CLASSIFIER_FEW_SHOT_K, CLASSIFIER_TEMPERATURE
from llm_utils import call_llm_json
from intent.taxonomy import load_taxonomy


def build_classification_prompt(taxonomy: dict, message: str) -> str:
    """Build a few-shot classification prompt from the taxonomy."""
    # Build intent descriptions with exemplars
    intent_block = ""
    intent_names = list(taxonomy.keys())
    for name, info in taxonomy.items():
        intent_block += f"\n- **{name}**: {info['description']}"
        exemplars = info.get('exemplars', [])[:CLASSIFIER_FEW_SHOT_K]
        for ex in exemplars:
            ex_short = ex[:100] + "..." if len(ex) > 100 else ex
            intent_block += f'\n  Example: "{ex_short}"'

    prompt = f"""Classify this customer support message into exactly one intent category.

Available intent categories:{intent_block}

Customer message: "{message}"

Return a JSON object:
{{
  "intent": "the_intent_label",
  "confidence": 0.XX,
  "reasoning": "brief 1-sentence explanation"
}}

Rules:
- "intent" must be one of: {intent_names}
- "confidence" is your probability estimate (0.0 to 1.0) that this is the correct label
- Be honest about confidence — low confidence for ambiguous messages is better than false certainty
- If the message could fit multiple categories, pick the strongest fit and lower your confidence"""

    return prompt


def classify(message: str, taxonomy: dict) -> dict:
    """
    Classify a customer message into an intent.
    Returns: {intent, confidence, reasoning}
    """
    prompt = build_classification_prompt(taxonomy, message)
    result = call_llm_json(prompt, temperature=CLASSIFIER_TEMPERATURE, max_tokens=200)

    # Validate
    intent_names = list(taxonomy.keys())
    if result.get('intent') not in intent_names:
        result['intent'] = 'other'
        result['confidence'] = max(result.get('confidence', 0.3) * 0.5, 0.1)

    # Clamp confidence
    result['confidence'] = max(0.0, min(1.0, result.get('confidence', 0.5)))

    return result


def classify_batch(messages: list[str], taxonomy: dict, progress: bool = True) -> list[dict]:
    """Classify a batch of messages."""
    from tqdm import tqdm
    results = []
    iterator = tqdm(messages, desc="Classifying") if progress else messages
    for msg in iterator:
        try:
            result = classify(msg, taxonomy)
            result['message'] = msg
            results.append(result)
        except Exception as e:
            results.append({
                'message': msg,
                'intent': 'other',
                'confidence': 0.0,
                'reasoning': f'Error: {e}',
                'error': True,
            })
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify customer messages")
    parser.add_argument("--brand", required=True)
    parser.add_argument("--message", type=str, help="Single message to classify")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on golden set")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.brand)
    print(f"Loaded taxonomy with {len(taxonomy)} intents: {list(taxonomy.keys())}")

    if args.message:
        result = classify(args.message, taxonomy)
        print(f"\nMessage: \"{args.message}\"")
        print(f"Intent:  {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Reason:  {result['reasoning']}")
    elif args.eval:
        print("Use eval/run_eval.py for full evaluation")
