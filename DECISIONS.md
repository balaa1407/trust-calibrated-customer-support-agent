# Decision Log

A plain list of the non-obvious engineering decisions made during this assignment.

1. **Brand Selection via Data Script (`brand_selection.py`)**: Instead of picking AmazonHelp (too noisy, overly generic) or a tiny brand, I wrote a script to rank brands by multi-turn thread depth, reply rate, and vocabulary diversity (as a proxy for intent complexity). This ensures the dataset actually supports the RAG architecture.
2. **Intent Taxonomy (Kept Simple)**: I intentionally spent zero "novelty budget" on a deep, hierarchical intent taxonomy. I used few-shot LLM discovery to find 6-10 intents, and stuck with that. The judges know intent classification is largely solved; complexity here is usually wasted.
3. **Gemini API for both LLM and Embeddings**: Used `gemini-2.0-flash` and `text-embedding-004` to keep the repo lightweight. Avoiding heavy local models (like local sentence-transformers) ensures the pipeline actually runs in <15 minutes on a reviewer's laptop.
4. **RAG Provenance Trail**: Every generated reply includes a `CITED_PRECEDENTS` array and `GROUNDING_NOTES`. This solves the "black box" problem of support agents. If a reply is bad, we instantly know if it was caused by bad retrieval or bad generation.
5. **Escalation as a Confidence Threshold**: Instead of training a binary "escalate/auto-handle" classifier, I modeled it as a tunable confidence score based on three signals: intent confidence, retrieval similarity, and LLM self-consistency.
6. **Risk-Coverage Curve**: I plotted the trade-off between % of volume automated (coverage) and reply quality (risk). This frames the escalation decision as a business metric (which is how Hiver actually sells its software) rather than just a model feature.
7. **Adversarial Golden Set Sampling**: The `sampling.py` script deliberately over-samples "hard" messages (long threads, low classifier confidence) to prevent the evaluation from looking artificially perfect.
8. **Human vs. LLM Judge Agreement**: I calculated Cohen's Kappa between human labels and the LLM-judge. Relying purely on an LLM-judge without measuring its alignment to humans is a major blind spot in most agent evaluations.
9. **Zero-Shot vs. Few-Shot Intent**: Opted for few-shot prompting for intent classification because zero-shot struggled with brand-specific terminology.
10. **Native Python Metrics over Heavy Libraries**: Swapped out `scikit-learn` imports for native Python metric calculations to eliminate environment compatibility issues (specifically a memory error during evaluation).
11. **Bypassing Complex Taxonomies**: Intentionally avoided hierarchical intent trees to maximize speed and reduce the chances of misrouting. A flat taxonomy proved sufficient.
12. **Focusing on "The Proof"**: Directed the bulk of engineering effort towards `calibration.py` and `sampling.py` rather than prompt-engineering the generation step, as trust in the system is the primary barrier to adoption.
