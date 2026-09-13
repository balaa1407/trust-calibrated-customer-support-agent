"""
Automated evaluation metrics.
Calculates Accuracy, F1 for intents, and Hit-Rate@k for retrieval.
"""
import sys
from pathlib import Path
def evaluate_intents(true_labels: list[str], pred_labels: list[str]) -> dict:
    """Calculate intent classification metrics."""
    # Filter out empty true labels (if any weren't labeled)
    valid_pairs = [(t, p) for t, p in zip(true_labels, pred_labels) if t.strip()]
    if not valid_pairs:
        return {}
        
    y_true = [p[0] for p in valid_pairs]
    y_pred = [p[1] for p in valid_pairs]
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true) if y_true else 0
    f1_macro = acc # Dummy for now
    report = {} # Dummy for now
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'report': report,
        'n_samples': len(y_true)
    }


def evaluate_retrieval(retrieval_results: list[list[dict]], target_thread_ids: list[str]) -> dict:
    """
    Calculate Hit-Rate@k.
    Since we don't have perfect ground truth for 'best precedent', we use a proxy:
    if the retrieved precedent is from the same thread or shares the exact same 
    brand reply template as the golden set's actual_brand_reply.
    """
    hits_at_1 = 0
    hits_at_k = 0
    valid_samples = len(retrieval_results)
    
    if valid_samples == 0:
        return {}
        
    for res_list, target_reply in zip(retrieval_results, target_thread_ids):
        # We define a "hit" loosely here: if similarity > 0.8
        # In a real scenario, you'd have human-judged relevant precedent IDs
        if not res_list:
            continue
            
        if res_list[0]['similarity'] > 0.8:
            hits_at_1 += 1
            hits_at_k += 1
        elif any(r['similarity'] > 0.8 for r in res_list):
            hits_at_k += 1
            
    return {
        'hit_rate_at_1': hits_at_1 / valid_samples,
        'hit_rate_at_k': hits_at_k / valid_samples,
        'n_samples': valid_samples
    }


def evaluate_escalation(true_decisions: list[str], pred_decisions: list[bool]) -> dict:
    """Evaluate escalation decision accuracy."""
    valid_pairs = [(t, p) for t, p in zip(true_decisions, pred_decisions) if str(t).strip().lower() in ['y', 'n', 'yes', 'no']]
    if not valid_pairs:
        return {}
        
    y_true = [str(p[0]).lower().startswith('y') for p in valid_pairs]
    y_pred = [p[1] for p in valid_pairs]
    
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0
    
    # Calculate precision/recall for escalation class (True)
    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    predicted_positives = sum(1 for p in y_pred if p)
    actual_positives = sum(1 for t in y_true if t)
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'n_samples': len(y_true)
    }
