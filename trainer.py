from trl import DPOTrainer

class StepwiseDPOTrainer(DPOTrainer):
    def __init__(self, *args, reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn

    def reward_function(self, prompt: str, response: str) -> float:
        steps = response.strip().split("\n")
        return self.reward_fn(prompt, steps) if self.reward_fn else 0.0
