"""
LLM-as-a-judge for evaluating reply quality.

Scores replies across 4 dimensions (1-5 scale):
1. Clarity
2. Correctness (does it address the issue?)
3. Tone match
4. Grounding faithfulness (does it align with the precedents?)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LLM_JUDGE_TEMPERATURE
from llm_utils import call_llm_json


def evaluate_reply(customer_msg: str, draft_reply: str, 
                   precedents: list[dict], brand_id: str) -> dict:
    """Evaluate a generated reply using an LLM judge."""
    
    precedent_text = ""
    for i, p in enumerate(precedents):
        precedent_text += f"\n[Precedent {i+1}] {p['brand_reply']}"
        
    prompt = f"""You are an expert customer support QA manager for {brand_id}.
Evaluate the following drafted reply to a customer message.

Customer message: "{customer_msg}"
Drafted reply: "{draft_reply}"

The reply was grounded in these past resolutions:
{precedent_text}

Rate the drafted reply on a scale of 1-5 for each of these 4 dimensions:
1. Clarity (is it easy to understand and well-written?)
2. Correctness (does it actually address the customer's specific issue?)
3. Tone match (does it match the professional/casual tone of the precedents?)
4. Grounding (does it follow the resolution pattern shown in the precedents?)

Return a JSON object:
{{
  "clarity": {{"score": X, "reason": "..."}},
  "correctness": {{"score": X, "reason": "..."}},
  "tone": {{"score": X, "reason": "..."}},
  "grounding": {{"score": X, "reason": "..."}},
  "overall_score": X.X  // Average of the 4 scores
}}"""

    try:
        result = call_llm_json(prompt, temperature=LLM_JUDGE_TEMPERATURE, max_tokens=400)
        
        # Ensure overall_score exists and is a float
        if 'overall_score' not in result:
            scores = [
                result.get('clarity', {}).get('score', 3),
                result.get('correctness', {}).get('score', 3),
                result.get('tone', {}).get('score', 3),
                result.get('grounding', {}).get('score', 3)
            ]
            result['overall_score'] = sum(scores) / len(scores)
            
        return result
    except Exception as e:
        print(f"  [Judge error] {e}")
        return {
            "clarity": {"score": 3, "reason": "Error parsing judge response"},
            "correctness": {"score": 3, "reason": "Error parsing judge response"},
            "tone": {"score": 3, "reason": "Error parsing judge response"},
            "grounding": {"score": 3, "reason": "Error parsing judge response"},
            "overall_score": 3.0
        }
