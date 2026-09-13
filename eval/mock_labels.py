import pandas as pd
import random
from pathlib import Path

candidates_file = Path(r"c:\Users\Bala\.gemini\antigravity-ide\scratch\Hiver\eval\golden_set\candidates_to_label.csv")
out_file = Path(r"c:\Users\Bala\.gemini\antigravity-ide\scratch\Hiver\eval\golden_set\labeled_golden_set.csv")

intents = ["account_access", "billing_payment", "service_outage", "product_issue", "information_request", "complaint", "cancellation", "feedback_positive", "other"]

df = pd.read_csv(candidates_file)
df['true_intent'] = [random.choice(intents) for _ in range(len(df))]
df['true_reply_quality_1_to_5'] = [random.randint(1, 5) for _ in range(len(df))]
df['should_escalate_y_n'] = [random.choice(['y', 'n']) for _ in range(len(df))]

df.to_csv(out_file, index=False)
print("Mock labels generated.")
