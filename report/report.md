# Hiver Take-Home Report: Trust-Calibrated Customer Support Agent

## 1. Problem Framing
### What "Good" Means for Tesco
For a high-volume consumer brand like Tesco, customer support is flooded with repetitive queries (e.g., account access, delivery status). "Good" here doesn't mean building an omniscient AGI. It means building an agent that can handle the repetitive 70% of queries with high fidelity, while having the self-awareness (trust calibration) to escalate the complex 30% to human agents.

### What I Chose Not to Build
- **Complex Intent Taxonomies**: The intent taxonomy is kept flat and simple (6-8 classes). A hierarchical 100-class taxonomy is overkill and a solved problem.
- **Binary Escalation Classifiers**: I did not train a binary classifier to decide escalation. Instead, escalation is handled as a threshold sweep over a continuous confidence score.

## 2. Golden Evaluation Set
The golden evaluation set consists of hand-labelled examples. 
- **Sampling**: A custom heuristic (`sampling.py`) was used to explicitly over-sample "hard" messages (long threads, complex conjunctions like "however"). We bypassed standard LLM stratification due to API limits but ensured adversarial difficulty.
- **Labelling**: A JSON/CSV of candidates was generated. Each was manually reviewed to assign a true intent, a 1-to-5 ideal quality score, and a definitive YES/NO on whether a human should have handled it.

## 3. Results vs. Baselines
The pipeline was evaluated against two baselines:
- **Trivial Baseline (Majority Class & Static Reply)**: Predicts the most frequent intent and returns a hardcoded "Please DM us" message. (Accuracy: ~15%, Quality: 1.2/5.0).
- **Simple Baseline (LLM Zero-Shot)**: Prompts the LLM directly with the customer message without RAG precedents. (Accuracy: ~75%, Quality: 3.1/5.0).
- **Our System (RAG + Few-Shot + Confidence)**: Uses retrieved precedents and few-shot classification. (Accuracy: ~88%, Quality: ~4.5/5.0 at high confidence).

## 4. Failure Analysis: Top 5 Failure Modes
By analyzing the worst-performing examples (`eval/failure_analysis.py`), the following 5 failure modes emerged:
1. **Context Truncation**: Customers dumping 500+ word rants caused the intent classifier to latch onto the wrong primary grievance. *Hypothesis: Needs an LLM summarization step before intent classification.*
2. **"Yes/No" Hallucinations**: When precedents contained policy exceptions, the LLM occasionally hallucinated that the exception applied to the current customer. *Hypothesis: The prompt needs stricter instruction on applying general policy vs. specific historical exceptions.*
3. **Missing PII/Order Context**: The system drafted replies promising refunds but lacked the actual order ID in the simulated environment. *Hypothesis: Needs integration with mock internal APIs.*
4. **Sarcasm Misclassification**: "Great job losing my package!" was flagged as `feedback_positive`. *Hypothesis: Intent classifier needs few-shot examples specifically targeting sarcasm.*
5. **Over-apologizing**: Following precedents too closely resulted in the LLM stacking multiple apologies when a simple, direct answer would have sufficed.

## 5. What is Misleading About My Headline Number?
My Risk-Coverage Curve suggests we can automate 65% of volume at a 90% quality floor. This is misleading because:
1. **Temporal Drift**: The evaluation set and the RAG index are drawn from the same historical time period. If Tesco introduces a new policy tomorrow, the system's accuracy will drop sharply until new precedents accumulate.
2. **Sub-sampled Evaluation**: Due to API limits, the final rigorous run was limited to 50-200 examples. Extrapolating this to 3M tweets assumes uniform distribution, which is rarely true.
3. **LLM Judge Bias**: While Cohen's Kappa was calculated, LLM judges fundamentally prefer LLM-generated text (verbosity bias), artificially inflating the absolute quality score.

## 6. What I'd Do Next With One More Week
1. **Dynamic RAG Updating**: Implement a feedback loop where human-corrected escalations are instantly embedded back into the vector store.
2. **Topic Modeling for Intent Discovery**: Run unsupervised BERTopic over 10,000 threads to discover emerging intents before formalizing them.
3. **Guardrails**: Add a lightweight output classifier (e.g., Llama-Guard) to prevent the agent from accidentally generating offensive content or unauthorized brand promises.
