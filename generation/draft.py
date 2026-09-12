"""
RAG reply generation with provenance — every draft cites which past threads inform it.

This is the one real technical novelty in the agent: the system shows its receipts.
It helps both with trust (the user sees why the agent said what it said) and with
debugging (you can trace a bad reply back to bad retrieval vs. bad generation).

Usage: python generation/draft.py --brand BRAND_ID --message "I can't log in"
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GENERATION_TEMPERATURE, GENERATION_MAX_TOKENS
from llm_utils import call_llm


def build_grounded_prompt(brand_id: str, customer_msg: str,
                          precedents: list[dict], intent: str = None) -> str:
    """Build a reply-generation prompt grounded in retrieved precedents."""

    precedent_block = ""
    for i, p in enumerate(precedents):
        precedent_block += f"""
--- Precedent {i+1} (similarity: {p['similarity']:.2f}, thread: {p['thread_id']}) ---
Customer asked: "{p['customer_msg'][:200]}"
Brand resolved with: "{p['brand_reply'][:300]}"
"""

    intent_hint = f"\nDetected intent: {intent}" if intent else ""

    prompt = f"""You are a support agent for {brand_id} on Twitter. Draft a helpful reply to this customer message.

IMPORTANT: Ground your reply in the precedent resolutions below. These are real past interactions showing how this brand handles similar issues. Match the brand's tone and resolution approach.

Customer message: "{customer_msg}"{intent_hint}

{precedent_block}

Instructions:
1. Write a concise, helpful Twitter-length reply (under 280 chars ideal, max 500 chars)
2. Match the brand's actual tone from the precedents (professional, casual, empathetic — whatever they use)
3. Address the specific issue, don't be generic
4. If the precedents suggest a specific resolution step (like a link, DM request, or action), include it
5. At the end, note which precedent(s) most informed your reply

Format your response as:
REPLY: [your draft reply]
CITED_PRECEDENTS: [comma-separated precedent numbers, e.g. "1, 3"]
GROUNDING_NOTES: [1-sentence note on how you used the precedents]"""

    return prompt


def generate_reply(brand_id: str, customer_msg: str, precedents: list[dict],
                   intent: str = None) -> dict:
    """
    Generate a grounded reply with provenance.
    Returns: {reply, cited_thread_ids, cited_similarities, grounding_notes}
    """
    prompt = build_grounded_prompt(brand_id, customer_msg, precedents, intent)
    raw = call_llm(prompt, temperature=GENERATION_TEMPERATURE,
                   max_tokens=GENERATION_MAX_TOKENS)

    # Parse structured response
    reply = ""
    cited = []
    notes = ""

    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith('REPLY:'):
            reply = line[len('REPLY:'):].strip()
        elif line.startswith('CITED_PRECEDENTS:'):
            cited_str = line[len('CITED_PRECEDENTS:'):].strip()
            try:
                cited = [int(x.strip()) - 1 for x in cited_str.split(',') if x.strip().isdigit()]
            except ValueError:
                cited = [0]
        elif line.startswith('GROUNDING_NOTES:'):
            notes = line[len('GROUNDING_NOTES:'):].strip()

    if not reply:
        # Fallback: use the whole response as the reply
        reply = raw.strip()[:500]

    # Build provenance
    cited_thread_ids = []
    cited_similarities = []
    for idx in cited:
        if 0 <= idx < len(precedents):
            cited_thread_ids.append(precedents[idx]['thread_id'])
            cited_similarities.append(precedents[idx]['similarity'])

    if not cited_thread_ids and precedents:
        # Default: cite the top precedent
        cited_thread_ids = [precedents[0]['thread_id']]
        cited_similarities = [precedents[0]['similarity']]

    return {
        'reply': reply,
        'cited_thread_ids': cited_thread_ids,
        'cited_similarities': cited_similarities,
        'grounding_notes': notes,
        'raw_response': raw,
    }


def generate_reply_generic(brand_id: str, customer_msg: str, intent: str = None) -> dict:
    """
    BASELINE 1: Generic LLM reply with NO retrieved precedents.
    Tests whether RAG actually helps vs. pure parametric knowledge.
    """
    intent_hint = f"\nDetected intent: {intent}" if intent else ""
    prompt = f"""You are a support agent for {brand_id} on Twitter. Draft a helpful reply.

Customer message: "{customer_msg}"{intent_hint}

Write a concise, helpful Twitter-length reply (under 280 chars ideal).
Be professional and address the specific issue."""

    raw = call_llm(prompt, temperature=GENERATION_TEMPERATURE, max_tokens=GENERATION_MAX_TOKENS)
    return {
        'reply': raw.strip()[:500],
        'cited_thread_ids': [],
        'cited_similarities': [],
        'grounding_notes': 'Generic LLM baseline — no retrieval grounding',
        'raw_response': raw,
    }


def generate_reply_template(precedents: list[dict]) -> dict:
    """
    BASELINE 2: Template-only — return the best-matching precedent's brand reply verbatim.
    Tests whether generation adds value over pure retrieval.
    """
    if not precedents:
        return {
            'reply': "We're sorry to hear about your issue. Please DM us for assistance.",
            'cited_thread_ids': [],
            'cited_similarities': [],
            'grounding_notes': 'Template baseline — no precedents found, using fallback',
            'raw_response': '',
        }

    best = precedents[0]
    return {
        'reply': best['brand_reply'],
        'cited_thread_ids': [best['thread_id']],
        'cited_similarities': [best['similarity']],
        'grounding_notes': f'Template baseline — verbatim reply from thread {best["thread_id"]}',
        'raw_response': best['brand_reply'],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate grounded replies")
    parser.add_argument("--brand", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    from retrieval.retrieve import retrieve
    precedents = retrieve(args.message, args.brand)

    print(f"\nCustomer: \"{args.message}\"")
    print(f"\nRetrieved {len(precedents)} precedents:")
    for i, p in enumerate(precedents):
        print(f"  [{i+1}] sim={p['similarity']:.3f}: \"{p['customer_msg'][:80]}\"")

    result = generate_reply(args.brand, args.message, precedents)
    print(f"\n{'='*60}")
    print(f"DRAFT REPLY: {result['reply']}")
    print(f"CITED: threads {result['cited_thread_ids']}")
    print(f"NOTES: {result['grounding_notes']}")
