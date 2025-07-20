## Stepwise-DPO with Custom Reward Logging

This project implements Stepwise DPO (Direct Preference Optimization) training with a custom reward model that logs stepwise feedback on LLM outputs. It is designed for research on alignment, interpretability, and stepwise reasoning in LLMs.

### Setup

1. Clone the repo and navigate to the project folder:
```bash
cd stepwise-dpo-futureagi
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Files Overview

- `train.py` — trains a GPT-2 model using TRL’s DPOTrainer and logs reward scores
- `llm_reward_model.py` — computes stepwise rewards using a small LLM (GPT-2)
- `plot_rewards.py` — plots average reward scores over training examples
- `logs/rewards.jsonl` — JSONL file containing step-level feedback scores
- `logs/reward_plot.png` — visualization of score trends

### Run

1. Train the model and collect reward logs:
```bash
python train.py
```

2. Plot reward score trends:
```bash
python plot_rewards.py
```
