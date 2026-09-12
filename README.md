# Hiver Take-Home: Trust-Calibrated Customer Support Agent

This repository implements a customer support AI agent for a Twitter brand, focusing heavily on **trust calibration** and **rigorous evaluation**, as requested in the assignment ("the proof is worth more than the system").

## Core Philosophy
1. **Boring Architecture, Novel Evaluation**: The agent uses standard few-shot classification and RAG. The technical novelty lies in the *escalation decision*, which is framed as a confidence-scoring problem mapped to a risk-coverage curve.
2. **Provenance**: Every generated reply cites the exact historical precedent(s) that informed it.
3. **Honesty**: The evaluation harness explicitly measures LLM-judge vs. Human agreement (Cohen's Kappa) and highlights systematic biases.

## Reproducing the Headline Numbers (in <15 minutes)

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   # Create a .env file with your GEMINI_API_KEY
   ```

2. **Download Data**:
   Download `twcs.csv` from Kaggle and place it in `data/raw/twcs.csv`.

3. **Run the Full Pipeline** (on the predefined golden set):
   ```bash
   # Cleans data, builds index, runs classification, retrieval, generation, 
   # and creates the risk-coverage plot.
   python eval/run_eval.py --brand BRAND_ID --subsample 50
   ```
   *Results, plots, and failure analysis will be saved in `eval/results/`.*

## Directory Structure
- `data/`: Data-driven brand selection and thread reconstruction.
- `intent/`: Few-shot taxonomy discovery and classification (+ baselines).
- `retrieval/`: Vector index over resolved threads (Gemini embeddings).
- `generation/`: Grounded reply drafting with citation/provenance tracking.
- `escalation/`: Confidence scoring (intent + retrieval + self-consistency) and Risk-Coverage analysis.
- `eval/`: The golden set sampler, metrics, LLM-judge, and human agreement analysis.

## Key Artifacts
- `eval/results/risk_coverage_curve.png`: Shows what % of volume can be automated at various quality thresholds.
- `report/report.md`: The max 6-page writeup.
- `DECISIONS.md`: Log of non-obvious engineering decisions.
