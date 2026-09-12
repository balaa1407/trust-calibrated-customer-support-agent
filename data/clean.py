"""
Clean raw Twitter data and reconstruct conversation threads for a selected brand.

Produces structured JSONL records:
  {thread_id, customer_messages[], brand_replies[], full_context, is_resolved}

Usage: python data/clean.py --brand BRAND_ID [--csv PATH]
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import re
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CSV_FILENAME


def clean_tweet_text(text: str) -> str:
    """Clean a single tweet: remove @mentions prefix, normalize whitespace."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove leading @mention (the brand tag) but keep inline mentions
    text = re.sub(r'^@\w+\s*', '', text)
    # Normalize URLs to [URL]
    text = re.sub(r'https?://\S+', '[URL]', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def reconstruct_threads(df: pd.DataFrame, brand_id: str) -> list[dict]:
    """
    Reconstruct conversation threads for a specific brand.
    A thread = a chain of tweets linked by response_tweet_id / in_response_to_tweet_id.
    """
    # Pre-process IDs to string to avoid float `.0` mismatches
    df['tweet_id'] = df['tweet_id'].astype(str)
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna(-1).astype(int).astype(str)
    df.loc[df['in_response_to_tweet_id'] == '-1', 'in_response_to_tweet_id'] = ''

    # Filter to brand's outbound tweets
    brand_outbound = df[(df['author_id'] == brand_id) & (df['inbound'] == False)]
    # All inbound tweets
    inbound = df[df['inbound'] == True]

    # Build lookup: tweet_id -> row
    tweet_lookup = df.set_index(df['tweet_id'].astype(str)).to_dict('index')

    # Build response graph: tweet_id -> list of response tweet_ids
    response_graph = {}
    for _, row in df.iterrows():
        resp_to = str(row.get('in_response_to_tweet_id', ''))
        if resp_to and resp_to != 'nan':
            if resp_to not in response_graph:
                response_graph[resp_to] = []
            response_graph[resp_to].append(str(row['tweet_id']))

    # Find thread roots: inbound tweets that are responded to by the brand
    brand_response_targets = brand_outbound['in_response_to_tweet_id'].dropna().astype(str).unique()
    # Thread roots = inbound tweets that have no in_response_to (they started the conversation)
    # OR inbound tweets that the brand directly replied to
    root_candidates = set()
    for target_id in brand_response_targets:
        # Walk up to find the root
        current = target_id
        visited = set()
        while current in tweet_lookup and current not in visited:
            visited.add(current)
            row = tweet_lookup[current]
            parent = str(row.get('in_response_to_tweet_id', ''))
            if parent == 'nan' or parent == '' or parent not in tweet_lookup:
                root_candidates.add(current)
                break
            current = parent

    print(f"  Found {len(root_candidates)} thread roots for brand {brand_id}")

    # Build threads from roots
    threads = []
    seen_tweets = set()

    for root_id in root_candidates:
        thread_msgs = []
        queue = [root_id]
        visited = set()

        while queue:
            tid = queue.pop(0)
            if tid in visited or tid not in tweet_lookup:
                continue
            visited.add(tid)
            seen_tweets.add(tid)

            row = tweet_lookup[tid]
            thread_msgs.append({
                'tweet_id': tid,
                'author_id': str(row.get('author_id', '')),
                'text': str(row.get('text', '')),
                'text_clean': clean_tweet_text(row.get('text', '')),
                'inbound': bool(row.get('inbound', True)),
                'created_at': str(row.get('created_at', '')),
            })

            # Follow responses
            if tid in response_graph:
                for resp_id in response_graph[tid]:
                    if resp_id not in visited:
                        queue.append(resp_id)

        if not thread_msgs:
            continue

        # Sort by created_at (or tweet_id as proxy for order)
        thread_msgs.sort(key=lambda m: m.get('created_at', ''))

        # Separate customer and brand messages
        customer_msgs = [m for m in thread_msgs if m['inbound']]
        brand_msgs = [m for m in thread_msgs if not m['inbound']]

        if not customer_msgs:
            continue

        # Determine if "resolved" — heuristic: thread ends with brand reply
        last_msg = thread_msgs[-1]
        is_resolved = not last_msg['inbound']  # Last message is from brand

        thread = {
            'thread_id': root_id,
            'brand_id': brand_id,
            'customer_messages': [m['text_clean'] for m in customer_msgs],
            'brand_replies': [m['text_clean'] for m in brand_msgs],
            'full_context': [
                {'role': 'customer' if m['inbound'] else 'brand', 'text': m['text_clean']}
                for m in thread_msgs
            ],
            'first_customer_msg': customer_msgs[0]['text_clean'],
            'thread_length': len(thread_msgs),
            'is_resolved': is_resolved,
        }
        threads.append(thread)

    return threads


def run_cleaning(brand_id: str, csv_path: Path = None):
    """Main cleaning pipeline."""
    if csv_path is None:
        csv_path = RAW_DATA_DIR / CSV_FILENAME

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} total tweets")

    # Convert inbound column
    df['inbound'] = df['inbound'].astype(bool)

    print(f"\nReconstructing threads for brand: {brand_id}")
    threads = reconstruct_threads(df, brand_id)

    # Stats
    resolved = [t for t in threads if t['is_resolved']]
    multi_turn = [t for t in threads if t['thread_length'] > 2]
    avg_len = np.mean([t['thread_length'] for t in threads]) if threads else 0

    print(f"\n{'='*60}")
    print(f"CLEANING RESULTS for {brand_id}")
    print(f"{'='*60}")
    print(f"  Total threads:     {len(threads):,}")
    print(f"  Resolved threads:  {len(resolved):,} ({100*len(resolved)/max(len(threads),1):.1f}%)")
    print(f"  Multi-turn (>2):   {len(multi_turn):,} ({100*len(multi_turn)/max(len(threads),1):.1f}%)")
    print(f"  Avg thread length: {avg_len:.1f} messages")

    # Save
    output_path = PROCESSED_DATA_DIR / f"{brand_id}_threads.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for thread in threads:
            f.write(json.dumps(thread, ensure_ascii=False) + '\n')
    print(f"\n Saved {len(threads)} threads to {output_path}")

    # Also save resolved threads separately (for RAG index)
    resolved_path = PROCESSED_DATA_DIR / f"{brand_id}_resolved.jsonl"
    with open(resolved_path, 'w', encoding='utf-8') as f:
        for thread in resolved:
            f.write(json.dumps(thread, ensure_ascii=False) + '\n')
    print(f" Saved {len(resolved)} resolved threads to {resolved_path}")

    return threads


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean data for a brand")
    parser.add_argument("--brand", required=True, help="Brand author_id")
    parser.add_argument("--csv", type=str, help="Path to twcs.csv")
    args = parser.parse_args()
    csv_path = Path(args.csv) if args.csv else None
    run_cleaning(args.brand, csv_path)
