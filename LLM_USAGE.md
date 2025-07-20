## LLM Usage in Stepwise-DPO

This project leverages an LLM (GPT-2) in two major roles:

### 1. Policy Model (Main GPT-2)
- Used as the base model in TRL’s `DPOTrainer`
- Fine-tuned on preference pairs (prompt, chosen, rejected)
- Learns to prefer high-quality step-by-step answers

### 2. Reward Model (GPT-2 via HuggingFace Pipeline)
- Used inside `llm_reward_model.py`
- Evaluates each reasoning step independently
- Outputs a numeric score (1–10) per step
- Logs results to `logs/rewards.jsonl` for later analysis

### Why GPT-2?
- Lightweight for experimentation
- Fast inference for reward scoring
- Easy to swap with more powerful LLMs (e.g. `gpt2-medium`, `tiiuae/falcon`) by updating model name in `llm_reward_model.py`

## Customization Ideas

- Replace reward model with your own scoring function (e.g. classifier)
- Swap in larger models or prompt-tuned versions
- Add step explanation logging for interpretability research
