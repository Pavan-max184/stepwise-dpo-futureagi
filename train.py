from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from llm_reward_model import StepwiseRewardModel

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
try:
    dataset = load_dataset("Intel/prm800k", split="train[:100]")
except Exception:
    print("\u26a0\ufe0f Failed to load Intel/prm800k. Falling back to gsm8k.")
    dataset = load_dataset("gsm8k", "main", split="train[:100]")

def format_example(example):
    prompt = example.get("question") or example.get("instruction") or "No prompt"
    chosen = example.get("answer") or example.get("output") or "No answer"
    rejected = "I don't know."
    return {
        "prompt": prompt.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected
    }

dataset = dataset.map(format_example)

training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    logging_dir="./logs",
    report_to="none"
)

reward_model = StepwiseRewardModel(model_name="gpt2")

# Custom Trainer using TRL's DPOTrainer
class StepwiseDPOTrainer(DPOTrainer):
    def __init__(self, *args, reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn

    def compute_rewards(self, prompt, response):
        steps = response.strip().split("\n")
        return self.reward_fn(prompt, steps)

trainer = StepwiseDPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_fn=reward_model.score
)

trainer.train()
