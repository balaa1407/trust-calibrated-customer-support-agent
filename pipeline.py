"""
End-to-End Execution Pipeline.

Orchestrates the full flow:
Classify -> Retrieve -> Draft -> Score Confidence -> Decide Escalation
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_ESCALATION_THRESHOLD
from intent.classifier import classify
from intent.taxonomy import load_taxonomy
from retrieval.retrieve import retrieve, load_index
from generation.draft import generate_reply
from escalation.confidence import compute_confidence

class SupportAgent:
    def __init__(self, brand_id: str):
        self.brand_id = brand_id
        print(f"Loading Support Agent for {brand_id}...")
        self.taxonomy = load_taxonomy(brand_id)
        self.index_data = load_index(brand_id)
        print("  Agent ready.")
        
    def handle_message(self, customer_msg: str, threshold: float = DEFAULT_ESCALATION_THRESHOLD) -> dict:
        """Process a single message through the full pipeline."""
        
        # 1. Classify Intent
        intent_res = classify(customer_msg, self.taxonomy)
        
        # 2. Retrieve Precedents
        precedents = retrieve(customer_msg, self.brand_id, index_data=self.index_data)
        
        # 3. Draft Reply
        draft_res = generate_reply(self.brand_id, customer_msg, precedents, intent_res['intent'])
        
        # 4. Confidence & Escalation
        conf_res = compute_confidence(
            intent_result=intent_res,
            retrieval_results=precedents,
            brand_id=self.brand_id,
            customer_msg=customer_msg
        )
        
        # Override threshold if requested
        should_escalate = conf_res['confidence'] < threshold
        if should_escalate != conf_res['should_escalate']:
            conf_res['should_escalate'] = should_escalate
            conf_res['escalation_reason'] = f"{'ESCALATE' if should_escalate else 'AUTO-HANDLE'} based on custom threshold {threshold}"
            
        return {
            'customer_msg': customer_msg,
            'intent': intent_res,
            'draft': draft_res,
            'escalation': conf_res,
            'precedents': [{'id': p['thread_id'], 'sim': p['similarity']} for p in precedents]
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run end-to-end support pipeline")
    parser.add_argument("--brand", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    
    agent = SupportAgent(args.brand)
    result = agent.handle_message(args.message)
    
    print("\n" + "="*60)
    print("PIPELINE RESULT")
    print("="*60)
    print(f"Customer:  \"{result['customer_msg']}\"")
    print(f"\nIntent:    {result['intent']['intent']} (conf: {result['intent']['confidence']:.2f})")
    print(f"Reasoning: {result['intent']['reasoning']}")
    
    print(f"\nPrecedents Retrieved: {len(result['precedents'])}")
    for p in result['precedents']:
        print(f"  - Thread {p['id']} (sim: {p['sim']:.3f})")
        
    print(f"\nDraft:     {result['draft']['reply']}")
    print(f"Citations: {result['draft']['cited_thread_ids']}")
    
    print(f"\nEscalation: {result['escalation']['should_escalate']}")
    print(f"Confidence: {result['escalation']['confidence']:.3f}")
    print(f"Signals:    {result['escalation']['signals']}")
    print(f"Reasoning:  {result['escalation']['escalation_reason']}")
