"""
Build and query a vector index over historically resolved brand threads.
Uses Gemini embeddings + numpy cosine similarity (no heavy dependencies).

Usage:
    python retrieval/index.py --brand BRAND_ID        # Build index
    python retrieval/retrieve.py --brand BRAND_ID --query "..."  # Query
"""
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR, INDEX_DIR, RETRIEVAL_TOP_K, EMBEDDING_BATCH_SIZE
from llm_utils import embed_texts


def build_index(brand_id: str):
    """
    Build vector index over resolved threads.
    Each document = first customer message + brand resolution.
    """
    resolved_path = PROCESSED_DATA_DIR / f"{brand_id}_resolved.jsonl"
    if not resolved_path.exists():
        print(f"✗ No resolved threads found at {resolved_path}")
        print("  Run data/clean.py first.")
        return

    # Load resolved threads
    threads = []
    with open(resolved_path, 'r', encoding='utf-8') as f:
        for line in f:
            threads.append(json.loads(line))

    import random
    random.seed(42)
    if len(threads) > 1500:
        threads = random.sample(threads, 1500)

    print(f"Building index over {len(threads)} resolved threads (sampled for API limits)...")

    # Prepare documents for embedding
    documents = []
    metadata = []
    for t in threads:
        customer_msg = t['first_customer_msg']
        brand_reply = t['brand_replies'][0] if t['brand_replies'] else ""

        if not customer_msg.strip() or len(customer_msg) < 5:
            continue

        # Document = customer query for semantic matching
        doc_text = customer_msg
        documents.append(doc_text)
        metadata.append({
            'thread_id': t['thread_id'],
            'customer_msg': customer_msg,
            'brand_reply': brand_reply,
            'all_customer_msgs': t['customer_messages'],
            'all_brand_replies': t['brand_replies'],
            'thread_length': t['thread_length'],
        })

    print(f"  {len(documents)} documents to embed")

    # Embed in batches
    print("  Embedding documents...")
    all_embeddings = []
    for i in tqdm(range(0, len(documents), EMBEDDING_BATCH_SIZE), desc="Embedding"):
        batch = documents[i:i + EMBEDDING_BATCH_SIZE]
        embeddings = embed_texts(batch, batch_size=len(batch))
        all_embeddings.extend(embeddings)

    # Convert to numpy
    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_matrix = embeddings_matrix / norms

    # Save index
    index_path = INDEX_DIR / f"{brand_id}_index.pkl"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings_matrix,
            'metadata': metadata,
            'documents': documents,
        }, f)

    print(f"\n  Built index: {embeddings_matrix.shape[0]} vectors, dim={embeddings_matrix.shape[1]}")
    print(f"  Saved to {index_path}")

    return index_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build retrieval index")
    parser.add_argument("--brand", required=True)
    args = parser.parse_args()
    build_index(args.brand)
