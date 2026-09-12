"""
Intent taxonomy builder — discover and define intents from the data.

Uses few-shot LLM labeling on a sample, then manual deduplication into a
clean 6-10 intent taxonomy. We intentionally keep this boring: it's a solved
problem, and novelty budget goes to the evaluation harness instead.

Usage: python intent/taxonomy.py --brand BRAND_ID
"""
import sys
import json
import random
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR, MAX_INTENTS
from llm_utils import call_llm_json


# Default taxonomy — updated after data-driven discovery
DEFAULT_TAXONOMY = {
    "account_access": {
        "description": "Login issues, password resets, locked accounts, verification problems",
        "exemplars": []
    },
    "billing_payment": {
        "description": "Charges, refunds, payment failures, subscription billing questions",
        "exemplars": []
    },
    "service_outage": {
        "description": "Service down, app not working, connectivity issues, error messages",
        "exemplars": []
    },
    "product_issue": {
        "description": "Bugs, quality problems, feature not working as expected",
        "exemplars": []
    },
    "information_request": {
        "description": "How-to questions, feature inquiries, general information seeking",
        "exemplars": []
    },
    "complaint": {
        "description": "General dissatisfaction, poor experience, frustrated venting",
        "exemplars": []
    },
    "cancellation": {
        "description": "Want to cancel service/subscription, threatening to leave",
        "exemplars": []
    },
    "feedback_positive": {
        "description": "Compliments, thanks, positive experience sharing",
        "exemplars": []
    },
    "other": {
        "description": "Doesn't fit other categories, unclear intent, off-topic",
        "exemplars": []
    },
}


def discover_intents_from_data(brand_id: str, sample_size: int = 300) -> dict:
    """
    Use LLM to discover intent categories from a sample of customer messages.
    Then merge with default taxonomy and clean up.
    """
    # Load threads
    threads_path = PROCESSED_DATA_DIR / f"{brand_id}_threads.jsonl"
    threads = []
    with open(threads_path, 'r', encoding='utf-8') as f:
        for line in f:
            threads.append(json.loads(line))

    # Sample customer messages
    all_msgs = []
    for t in threads:
        for msg in t['customer_messages']:
            if len(msg.strip()) > 10:  # Skip very short messages
                all_msgs.append(msg)

    sample = random.sample(all_msgs, min(sample_size, len(all_msgs)))
    print(f"  Sampled {len(sample)} customer messages for intent discovery")

    # Batch LLM labeling — send chunks of 20 messages
    discovered_labels = []
    batch_size = 20

    for i in range(0, len(sample), batch_size):
        batch = sample[i:i + batch_size]
        messages_text = "\n".join([f"{j+1}. \"{msg}\"" for j, msg in enumerate(batch)])

        prompt = f"""Analyze these customer support messages and assign each an intent category.
Use short, snake_case labels (e.g., account_access, billing_payment, service_outage).

Messages:
{messages_text}

Return a JSON object with:
{{
  "labels": ["label_for_msg_1", "label_for_msg_2", ...],
  "new_categories": [
    {{"name": "category_name", "description": "what this category covers"}}
  ]
}}

Only include truly new categories in new_categories (not standard ones like account, billing, etc.)."""

        try:
            result = call_llm_json(prompt, temperature=0.0, max_tokens=1000)
            labels = result.get('labels', [])
            discovered_labels.extend(labels)
            new_cats = result.get('new_categories', [])
            if new_cats:
                for cat in new_cats:
                    print(f"    Discovered new category: {cat['name']} — {cat.get('description', '')}")
        except Exception as e:
            err_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            print(f"  [Warning] LLM labeling failed for batch {i}: {err_msg}")
            continue

        # Progress
        if (i // batch_size) % 3 == 0:
            print(f"  Processed {min(i + batch_size, len(sample))}/{len(sample)} messages...")

    # Count discovered labels
    label_counts = Counter(discovered_labels)
    print(f"\n  Discovered {len(label_counts)} unique labels from {len(discovered_labels)} messages:")
    for label, count in label_counts.most_common(20):
        print(f"    {label}: {count}")

    # Merge with default taxonomy
    taxonomy = dict(DEFAULT_TAXONOMY)

    # Map discovered labels to taxonomy entries
    # This is a manual-audit step in practice; here we do a best-effort merge
    label_mapping = {}
    for label in label_counts:
        normalized = label.lower().replace(' ', '_').replace('-', '_')
        if normalized in taxonomy:
            label_mapping[label] = normalized
        else:
            # Try fuzzy matching
            matched = False
            for tax_key in taxonomy:
                if tax_key in normalized or normalized in tax_key:
                    label_mapping[label] = tax_key
                    matched = True
                    break
            if not matched and label_counts[label] >= 3:
                # Add as new category if it appears enough times
                taxonomy[normalized] = {
                    "description": f"Auto-discovered: {label}",
                    "exemplars": []
                }
                label_mapping[label] = normalized

    # Add exemplars to each intent
    for msg, label in zip(sample[:len(discovered_labels)], discovered_labels):
        mapped = label_mapping.get(label, 'other')
        if mapped in taxonomy and len(taxonomy[mapped]['exemplars']) < 5:
            taxonomy[mapped]['exemplars'].append(msg)

    # Prune empty categories (keep at most MAX_INTENTS)
    non_empty = {k: v for k, v in taxonomy.items() if v['exemplars']}
    if len(non_empty) > MAX_INTENTS:
        # Keep top by exemplar count
        sorted_cats = sorted(non_empty.items(), key=lambda x: len(x[1]['exemplars']), reverse=True)
        taxonomy = dict(sorted_cats[:MAX_INTENTS])
    else:
        taxonomy = non_empty if non_empty else DEFAULT_TAXONOMY

    # Always ensure 'other' exists
    if 'other' not in taxonomy:
        taxonomy['other'] = DEFAULT_TAXONOMY['other']

    return taxonomy


def save_taxonomy(taxonomy: dict, brand_id: str):
    """Save taxonomy to disk."""
    output_path = PROCESSED_DATA_DIR / f"{brand_id}_taxonomy.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved taxonomy ({len(taxonomy)} intents) to {output_path}")
    return output_path


def load_taxonomy(brand_id: str) -> dict:
    """Load saved taxonomy."""
    path = PROCESSED_DATA_DIR / f"{brand_id}_taxonomy.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build intent taxonomy")
    parser.add_argument("--brand", required=True, help="Brand ID")
    parser.add_argument("--sample-size", type=int, default=300)
    args = parser.parse_args()

    print(f"Building intent taxonomy for {args.brand}...")
    taxonomy = discover_intents_from_data(args.brand, args.sample_size)

    print(f"\nFinal taxonomy ({len(taxonomy)} intents):")
    for name, info in taxonomy.items():
        desc = info['description'].encode('ascii', 'ignore').decode('ascii')
        print(f"  {name}: {desc}")
        for ex in info['exemplars'][:2]:
            clean_ex = ex.encode('ascii', 'ignore').decode('ascii')
            print(f"    → \"{clean_ex[:80]}...\"" if len(clean_ex) > 80 else f"    → \"{clean_ex}\"")

    save_taxonomy(taxonomy, args.brand)
