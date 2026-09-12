"""
Data-driven brand selection — the "20-minute rigor signal" script.

Computes per-brand metrics and recommends a brand based on:
- Volume (10k-50k inbound tweets sweet spot)
- Reply rate (brand responsiveness)
- Multi-turn thread depth
- Vocabulary diversity (proxy for intent complexity)

Usage: python data/brand_selection.py [--csv PATH]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from collections import Counter
from config import RAW_DATA_DIR, CSV_FILENAME


def load_raw_data(csv_path: Path = None) -> pd.DataFrame:
    """Load the raw Twitter support CSV."""
    if csv_path is None:
        csv_path = RAW_DATA_DIR / CSV_FILENAME
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} tweets, columns: {list(df.columns)}")
    return df


def extract_brand(author_id: str, text: str, inbound: bool) -> str:
    """Extract brand name from outbound tweets (author_id for brands)."""
    if not inbound:
        return str(author_id)
    return None


def compute_brand_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute selection metrics for each brand."""
    # Pre-process IDs
    df['tweet_id'] = df['tweet_id'].astype(str)
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna(-1).astype(int).astype(str)
    df.loc[df['in_response_to_tweet_id'] == '-1', 'in_response_to_tweet_id'] = ''

    # Separate inbound (customer) and outbound (brand) tweets
    outbound = df[df['inbound'] == False].copy()
    inbound = df[df['inbound'] == True].copy()
    
    print("  Building response graph lookup...")
    response_graph = {}
    for tid, resp_to in zip(df['tweet_id'], df['in_response_to_tweet_id']):
        if resp_to != '':
            if resp_to not in response_graph:
                response_graph[resp_to] = []
            response_graph[resp_to].append(tid)

    # Get unique brand author_ids (those who send outbound tweets)
    brand_ids = outbound['author_id'].unique()
    print(f"  Found {len(brand_ids)} unique brand accounts")

    metrics = []
    for brand_id in brand_ids:
        brand_tweets = outbound[outbound['author_id'] == brand_id]
        brand_name = str(brand_id)

        # Volume metrics
        n_outbound = len(brand_tweets)

        # Find inbound tweets that these brand tweets respond to
        brand_response_targets = brand_tweets[brand_tweets['in_response_to_tweet_id'] != '']['in_response_to_tweet_id']
        # Find inbound tweets responded to by this brand
        inbound_to_brand = inbound[inbound['tweet_id'].isin(brand_response_targets)]
        n_inbound = len(inbound_to_brand)

        if n_inbound < 500:  # Skip tiny brands
            continue

        # Reply rate
        reply_rate = n_outbound / max(n_inbound, 1)

        # Thread depth — count chains of responses
        thread_depths = []
        # Sample to avoid huge computation
        sample_ids = brand_tweets['tweet_id'].sample(min(200, len(brand_tweets))).values
        for tid in sample_ids:
            depth = 1
            current = str(tid)
            visited = set()
            while True:
                responses = response_graph.get(current, [])
                if len(responses) == 0 or current in visited:
                    break
                visited.add(current)
                current = responses[0]
                depth += 1
                if depth > 10:
                    break
            thread_depths.append(depth)
        avg_depth = np.mean(thread_depths) if thread_depths else 1.0

        # Vocabulary diversity (unique words / total words in customer messages)
        if len(inbound_to_brand) > 0:
            sample_texts = inbound_to_brand['text'].dropna().sample(min(500, len(inbound_to_brand)))
            all_words = ' '.join(sample_texts.str.lower()).split()
            vocab_size = len(set(all_words))
            vocab_diversity = vocab_size / max(len(all_words), 1)
        else:
            vocab_diversity = 0.0

        # Unique customers
        n_customers = inbound_to_brand['author_id'].nunique()

        metrics.append({
            'brand_id': brand_id,
            'n_inbound': n_inbound,
            'n_outbound': n_outbound,
            'reply_rate': round(reply_rate, 2),
            'avg_thread_depth': round(avg_depth, 2),
            'vocab_diversity': round(vocab_diversity, 4),
            'n_customers': n_customers,
        })

    metrics_df = pd.DataFrame(metrics)
    return metrics_df


def score_and_rank(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score brands on suitability for the assignment.
    Sweet spot: moderate volume, high reply rate, good depth, diverse vocabulary.
    """
    df = metrics_df.copy()

    # Volume score: penalize too small (<2000) and too large (>100000)
    df['volume_score'] = df['n_inbound'].apply(
        lambda x: 1.0 if 5000 <= x <= 50000
        else 0.7 if 2000 <= x < 5000 or 50000 < x <= 100000
        else 0.3
    )

    # Reply rate score (higher = better, capped at 1.5)
    df['reply_score'] = df['reply_rate'].clip(upper=1.5) / 1.5

    # Thread depth score (deeper = more interesting)
    max_depth = df['avg_thread_depth'].max()
    df['depth_score'] = df['avg_thread_depth'] / max(max_depth, 1)

    # Vocab diversity score (higher = more diverse intents)
    max_vocab = df['vocab_diversity'].max()
    df['vocab_score'] = df['vocab_diversity'] / max(max_vocab, 1)

    # Composite score (weighted)
    df['total_score'] = (
        0.30 * df['volume_score'] +
        0.25 * df['reply_score'] +
        0.20 * df['depth_score'] +
        0.25 * df['vocab_score']
    )

    return df.sort_values('total_score', ascending=False)


def select_brand(csv_path: Path = None) -> dict:
    """Run brand selection and return the recommended brand + reasoning."""
    df = load_raw_data(csv_path)
    print("\nComputing per-brand metrics (this may take 1-2 minutes)...")
    metrics = compute_brand_metrics(df)
    ranked = score_and_rank(metrics)

    print("\n" + "=" * 80)
    print("BRAND RANKING (top 15)")
    print("=" * 80)
    display_cols = ['brand_id', 'n_inbound', 'n_outbound', 'reply_rate',
                    'avg_thread_depth', 'vocab_diversity', 'total_score']
    print(ranked[display_cols].head(15).to_string(index=False))

    # Pick the top brand
    top = ranked.iloc[0]
    result = {
        'brand_id': str(top['brand_id']),
        'n_inbound': int(top['n_inbound']),
        'n_outbound': int(top['n_outbound']),
        'reply_rate': float(top['reply_rate']),
        'avg_thread_depth': float(top['avg_thread_depth']),
        'vocab_diversity': float(top['vocab_diversity']),
        'total_score': float(top['total_score']),
    }

    print(f"\n RECOMMENDED BRAND: {result['brand_id']}")
    print(f"  Inbound volume:  {result['n_inbound']:,}")
    print(f"  Reply rate:      {result['reply_rate']:.2f}")
    print(f"  Avg thread depth: {result['avg_thread_depth']:.2f}")
    print(f"  Vocab diversity: {result['vocab_diversity']:.4f}")

    # Save metrics for the report
    ranked.to_csv(RAW_DATA_DIR.parent / "processed" / "brand_metrics.csv", index=False)
    print(f"\n  Saved full metrics to data/processed/brand_metrics.csv")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data-driven brand selection")
    parser.add_argument("--csv", type=str, help="Path to twcs.csv")
    args = parser.parse_args()
    csv_path = Path(args.csv) if args.csv else None
    select_brand(csv_path)
