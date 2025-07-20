from transformers import pipeline
from typing import List
import os
import json

class StepwiseRewardModel:
    def __init__(self, model_name="gpt2"):
        self.pipe = pipeline("text-generation", model=model_name)
        self.log_path = os.path.join("logs", "rewards.jsonl")
        os.makedirs("logs", exist_ok=True)

    def score(self, prompt: str, steps: List[str]) -> float:
        step_scores = []
        for step in steps:
            input_text = f"Question: {prompt}\nStep: {step}\nRate this step from 1 to 10:"
            result = self.pipe(input_text, max_new_tokens=10, do_sample=False)[0]["generated_text"]
            try:
                score = float([s for s in result.split() if s.replace('.', '', 1).isdigit()][-1])
                score = max(1.0, min(score, 10.0))
            except:
                score = 5.0
            step_scores.append(score)

        avg_score = sum(step_scores) / len(step_scores)

        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "prompt": prompt,
                "steps": steps,
                "step_scores": step_scores,
                "avg_score": avg_score
            }) + "\n")

        return avg_score

