import json
import os
import matplotlib.pyplot as plt

log_path = os.path.join("logs", "rewards.jsonl")
if not os.path.exists(log_path):
    raise FileNotFoundError("Reward log not found. Run train.py first.")

scores = []
with open(log_path) as f:
    for line in f:
        entry = json.loads(line)
        scores.append(entry["avg_score"])

plt.plot(scores, marker='o', color='blue')
plt.title("Reward Scores per Sample")
plt.xlabel("Sample Index")
plt.ylabel("Average Reward Score")
plt.grid(True)
plt.savefig("logs/reward_plot.png")
plt.show()
