"""
Confidence scoring for escalation decisions — the core novelty.

Instead of a binary "auto-handle or escalate" classifier, we compute a
calibrated confidence score from three signals:
  1. Intent classification confidence
  2. Retrieval grounding strength (max similarity of top-k precedents)
  3. LLM self-consistency (agreement between multiple reply samples)

The confidence score feeds into a risk-coverage curve that answers:
"At what threshold can we safely automate X% of volume?"

Usage: python escalation/confidence.py --demo
"""
import sys
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (CONFIDENCE_WEIGHTS, DEFAULT_ESCALATION_THRESHOLD,
                     SELF_CONSISTENCY_SAMPLES, GENERATION_TEMPERATURE)
from llm_utils import call_llm


def compute_self_consistency(brand_id: str, customer_msg: str,
                             precedents: list[dict], n_samples: int = None) -> float:
    """
    Generate the reply multiple times and measure agreement.
    High agreement = model is "sure"; low agreement = unstable/risky.

    Returns a score in [0, 1].
    """
    if n_samples is None:
        n_samples = SELF_CONSISTENCY_SAMPLES

    from generation.draft import build_grounded_prompt

    prompt = build_grounded_prompt(brand_id, customer_msg, precedents)

    replies = []
    for _ in range(n_samples):
        try:
            raw = call_llm(prompt, temperature=0.7, max_tokens=300)  # Higher temp for diversity
            # Extract just the reply part
            reply = ""
            for line in raw.split('\n'):
                if line.strip().startswith('REPLY:'):
                    reply = line.strip()[len('REPLY:'):].strip()
                    break
            if not reply:
                reply = raw.strip()[:300]
            replies.append(reply.lower())
        except Exception:
            replies.append("")

    if len(replies) < 2:
        return 0.5

    # Compute pairwise word-overlap similarity (Jaccard)
    similarities = []
    for i in range(len(replies)):
        for j in range(i + 1, len(replies)):
            words_i = set(replies[i].split())
            words_j = set(replies[j].split())
            if not words_i or not words_j:
                similarities.append(0.0)
                continue
            intersection = words_i & words_j
            union = words_i | words_j
            similarities.append(len(intersection) / len(union))

    return float(np.mean(similarities)) if similarities else 0.5


def compute_confidence(intent_result: dict, retrieval_results: list[dict],
                       self_consistency: float = None,
                       brand_id: str = None, customer_msg: str = None) -> dict:
    """
    Compute composite confidence score from three signals.

    Args:
        intent_result: {intent, confidence, reasoning} from classifier
        retrieval_results: list of {similarity, ...} from retriever
        self_consistency: pre-computed self-consistency score (optional)
        brand_id: needed if self_consistency not pre-computed
        customer_msg: needed if self_consistency not pre-computed

    Returns: {
        confidence: float [0,1],
        signals: {intent_confidence, retrieval_strength, self_consistency},
        should_escalate: bool,
        escalation_reason: str,
    }
    """
    # Signal 1: Intent classification confidence
    intent_conf = intent_result.get('confidence', 0.5)

    # Signal 2: Retrieval grounding strength (max similarity of top results)
    if retrieval_results:
        retrieval_strength = max(r['similarity'] for r in retrieval_results)
    else:
        retrieval_strength = 0.0

    # Signal 3: Self-consistency (compute if not provided)
    if self_consistency is None:
        if brand_id and customer_msg and retrieval_results:
            self_consistency = compute_self_consistency(
                brand_id, customer_msg, retrieval_results
            )
        else:
            self_consistency = 0.5

    # Weighted combination
    w = CONFIDENCE_WEIGHTS
    confidence = (
        w['intent_confidence'] * intent_conf +
        w['retrieval_strength'] * retrieval_strength +
        w['self_consistency'] * self_consistency
    )
    confidence = max(0.0, min(1.0, confidence))

    # Escalation decision
    should_escalate = confidence < DEFAULT_ESCALATION_THRESHOLD

    # Generate reason
    reasons = []
    if intent_conf < 0.5:
        reasons.append(f"low intent confidence ({intent_conf:.2f})")
    if retrieval_strength < 0.4:
        reasons.append(f"weak retrieval match ({retrieval_strength:.2f})")
    if self_consistency < 0.4:
        reasons.append(f"inconsistent reply generation ({self_consistency:.2f})")

    if should_escalate:
        escalation_reason = "ESCALATE: " + ("; ".join(reasons) if reasons else "confidence below threshold")
    else:
        escalation_reason = "AUTO-HANDLE: confidence sufficient for automated response"

    return {
        'confidence': confidence,
        'signals': {
            'intent_confidence': intent_conf,
            'retrieval_strength': retrieval_strength,
            'self_consistency': self_consistency,
        },
        'should_escalate': should_escalate,
        'escalation_reason': escalation_reason,
        'threshold_used': DEFAULT_ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    # Demo with synthetic data
    print("Confidence Scoring Demo")
    print("=" * 60)

    # High confidence case
    result = compute_confidence(
        intent_result={'intent': 'account_access', 'confidence': 0.92},
        retrieval_results=[{'similarity': 0.88}, {'similarity': 0.75}],
        self_consistency=0.85,
    )
    print(f"\nHigh-confidence case:")
    print(f"  Score: {result['confidence']:.3f}")
    print(f"  Signals: {result['signals']}")
    print(f"  Decision: {result['escalation_reason']}")

    # Low confidence case
    result = compute_confidence(
        intent_result={'intent': 'other', 'confidence': 0.3},
        retrieval_results=[{'similarity': 0.35}],
        self_consistency=0.25,
    )
    print(f"\nLow-confidence case:")
    print(f"  Score: {result['confidence']:.3f}")
    print(f"  Signals: {result['signals']}")
    print(f"  Decision: {result['escalation_reason']}")
