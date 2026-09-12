"""
Baseline intent classifiers for comparison:
  1. Majority-class predictor (trivial baseline)
  2. TF-IDF + Logistic Regression (simple ML baseline)

These exist to make the evaluation rigorous — showing the LLM classifier
actually adds value over trivial approaches.
"""
import sys
import json
import pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from config import PROCESSED_DATA_DIR


class MajorityClassBaseline:
    """Always predicts the most frequent intent. The bar that must be cleared."""

    def __init__(self):
        self.majority_class = None
        self.class_distribution = {}

    def fit(self, messages: list[str], labels: list[str]):
        counts = Counter(labels)
        self.majority_class = counts.most_common(1)[0][0]
        total = len(labels)
        self.class_distribution = {k: v/total for k, v in counts.items()}
        print(f"  MajorityClass baseline: always predicts '{self.majority_class}' "
              f"({self.class_distribution[self.majority_class]:.1%} of data)")

    def predict(self, message: str) -> dict:
        return {
            'intent': self.majority_class,
            'confidence': self.class_distribution.get(self.majority_class, 0.5),
            'reasoning': f'Majority-class baseline: always predicts {self.majority_class}',
        }

    def predict_batch(self, messages: list[str]) -> list[dict]:
        return [self.predict(m) for m in messages]


class TfidfLogRegBaseline:
    """TF-IDF vectorizer + Logistic Regression. Simple but surprisingly strong."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        self.model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
        self.is_fitted = False

    def fit(self, messages: list[str], labels: list[str]):
        X = self.vectorizer.fit_transform(messages)
        self.model.fit(X, labels)
        self.is_fitted = True

        # Cross-val score
        scores = cross_val_score(self.model, X, labels, cv=min(5, len(set(labels))), scoring='accuracy')
        print(f"  TF-IDF+LogReg baseline: cross-val accuracy = {scores.mean():.3f} ± {scores.std():.3f}")

    def predict(self, message: str) -> dict:
        if not self.is_fitted:
            return {'intent': 'other', 'confidence': 0.0, 'reasoning': 'Model not fitted'}
        X = self.vectorizer.transform([message])
        proba = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_label = self.model.classes_[pred_idx]
        return {
            'intent': pred_label,
            'confidence': float(proba[pred_idx]),
            'reasoning': f'TF-IDF+LogReg prediction (prob={proba[pred_idx]:.3f})',
        }

    def predict_batch(self, messages: list[str]) -> list[dict]:
        if not self.is_fitted:
            return [{'intent': 'other', 'confidence': 0.0, 'reasoning': 'Not fitted'}] * len(messages)
        X = self.vectorizer.transform(messages)
        probas = self.model.predict_proba(X)
        results = []
        for i, proba in enumerate(probas):
            pred_idx = np.argmax(proba)
            results.append({
                'intent': self.model.classes_[pred_idx],
                'confidence': float(proba[pred_idx]),
                'reasoning': f'TF-IDF+LogReg (prob={proba[pred_idx]:.3f})',
            })
        return results

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)
        print(f"  Saved TF-IDF+LogReg model to {path}")

    def load(self, path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.model = data['model']
        self.is_fitted = True


def train_baselines(brand_id: str, labeled_data: list[dict] = None) -> tuple:
    """
    Train baseline classifiers.
    labeled_data: list of {message, intent} dicts (from LLM-labeled training set)
    """
    if labeled_data is None:
        # Load LLM-labeled data if available
        label_path = PROCESSED_DATA_DIR / f"{brand_id}_labeled_train.jsonl"
        if label_path.exists():
            labeled_data = []
            with open(label_path, 'r') as f:
                for line in f:
                    labeled_data.append(json.loads(line))
        else:
            print(f"  No labeled training data found at {label_path}")
            return None, None

    messages = [d['message'] for d in labeled_data]
    labels = [d['intent'] for d in labeled_data]

    print(f"\nTraining baselines on {len(messages)} labeled examples...")

    majority = MajorityClassBaseline()
    majority.fit(messages, labels)

    tfidf_lr = TfidfLogRegBaseline()
    tfidf_lr.fit(messages, labels)

    # Save
    tfidf_lr.save(PROCESSED_DATA_DIR / f"{brand_id}_tfidf_logreg.pkl")

    return majority, tfidf_lr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True)
    args = parser.parse_args()
    train_baselines(args.brand)
