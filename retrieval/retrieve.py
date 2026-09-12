"""
Retrieve top-k similar resolved threads for a customer query.
Returns precedent threads with similarity scores for provenance tracking.
"""
import sys
import pickle
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INDEX_DIR, RETRIEVAL_TOP_K
from llm_utils import embed_query


def load_index(brand_id: str) -> dict:
    """Load the pre-built vector index."""
    index_path = INDEX_DIR / f"{brand_id}_index.pkl"
    if not index_path.exists():
        raise FileNotFoundError(f"No index found at {index_path}. Run retrieval/index.py first.")
    with open(index_path, 'rb') as f:
        return pickle.load(f)


def retrieve(query: str, brand_id: str, top_k: int = None, index_data: dict = None) -> list[dict]:
    """
    Retrieve top-k similar resolved threads for a customer query.

    Returns list of:
      {thread_id, similarity, customer_msg, brand_reply, all_customer_msgs, all_brand_replies}
    """
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    if index_data is None:
        index_data = load_index(brand_id)

    embeddings = index_data['embeddings']
    metadata = index_data['metadata']

    # Embed the query
    query_vec = np.array(embed_query(query), dtype=np.float32)
    # Normalize
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    # Cosine similarity (embeddings are already normalized)
    similarities = embeddings @ query_vec

    # Top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        meta = metadata[idx]
        results.append({
            'thread_id': meta['thread_id'],
            'similarity': float(similarities[idx]),
            'customer_msg': meta['customer_msg'],
            'brand_reply': meta['brand_reply'],
            'all_customer_msgs': meta['all_customer_msgs'],
            'all_brand_replies': meta['all_brand_replies'],
            'thread_length': meta['thread_length'],
        })

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrieve similar threads")
    parser.add_argument("--brand", required=True)
    parser.add_argument("--query", required=True, help="Customer message to find precedents for")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    args = parser.parse_args()

    results = retrieve(args.query, args.brand, args.top_k)
    print(f"\nTop {len(results)} precedents for: \"{args.query}\"\n")
    for i, r in enumerate(results):
        print(f"  [{i+1}] Similarity: {r['similarity']:.4f} | Thread: {r['thread_id']}")
        print(f"      Customer: \"{r['customer_msg'][:100]}\"")
        print(f"      Brand:    \"{r['brand_reply'][:100]}\"")
        print()
