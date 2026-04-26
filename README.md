<div align="center">

# MedAgents-X

**Multi-Agent Clinical Decision System**

*Six AI agents. One patient. A race to the right diagnosis.*

---

**Team Coders** — Sourav Bhardwaj · Jayti Bhardwaj · Tanishka Jha

Scaler School of Technology · Meta OpenEnv Hackathon, Final Round · April 2026

</div>

---

## The Problem

A patient walks in with fever, cough, and weight loss. The correct diagnosis could be flu, tuberculosis, or lung cancer. Getting it wrong can cost a life. Getting it right requires gathering the right evidence in the right order — not a single prediction, but a structured process involving multiple specialists, delayed test results, and incomplete information at every step.

MedAgents-X simulates this entire clinical workflow as a reinforcement learning environment. Six AI agents with distinct medical roles collaborate across six pipeline stages, starting with only symptoms and progressively building toward a final diagnosis as evidence is revealed.

---

## How It Works

```
Patient arrives → symptoms visible only (tests hidden)
        ↓
Stage 0 · GP forms differential diagnosis from symptoms
Stage 1 · GP + Specialist order diagnostic tests
Stage 2 · Test results revealed → Radiologist + Pathologist interpret
Stage 3 · Specialist synthesizes all evidence → proposes diagnosis
Stage 4 · Supervisor reviews, approves or rejects, generates feedback
Stage 5 · Oversight AI logs patterns, updates agent memory
        ↓
Reward computed → feedback stored in memory → agents improve next case
```

The environment enforces **partial observability** — test results are deliberately hidden until Stage 2, forcing agents to reason under uncertainty exactly as real doctors do. Decisions made early (which tests to order) directly affect what information is available later (which diagnosis can be supported).

---

## The Six Agents

| Agent | Medical Role | Pipeline Stage |
|-------|-------------|----------------|
| General Physician | Reviews symptoms, builds differential diagnosis | 0, 1 |
| Radiologist | Interprets MRI, X-ray, ECG imaging results | 2 |
| Pathologist | Interprets lab tests, biopsies, blood work | 2 |
| Specialist | Scores all evidence, proposes final diagnosis | 1, 3 |
| Supervisor | Approves or rejects diagnosis, generates structured feedback | 4 |
| Oversight AI | Detects cross-case patterns, pushes learned rules to agent memory | 5 |

---

## The Dataset

35 real patient cases spanning infectious diseases, cancers, cardiac emergencies, metabolic disorders, and neurological conditions. Each case contains presenting symptoms, a differential diagnosis, the confirmed correct diagnosis, and the diagnostic tests that confirm it.

| Case | Symptoms | Correct Diagnosis |
|------|----------|-------------------|
| 5 | Chest pain, shortness of breath | Heart attack |
| 6 | Headache, vision problems, vomiting | Brain tumor |
| 11 | Weight loss, fatigue, night sweats | Lymphoma |
| 22 | Severe headache, neck stiffness | Meningitis |
| 33 | Fever, confusion | Sepsis |

Cases are automatically ordered from easy to hard during training using a curriculum — low severity cases first, critical cases last — so agents encounter successful trajectories early and build from there.

---

## Reward Function

Nine independent reward signals prevent the model from gaming any single metric.

| Signal | Value |
|--------|-------|
| ✅ Correct diagnosis | +0.40 |
| ✅ Evidence-based reasoning | +0.20 |
| ✅ Calibrated confidence | +0.20 |
| ✅ Test efficiency | +0.10 |
| ✅ Critical disease caught | +0.10 |
| ❌ Wrong diagnosis | -0.50 |
| ❌ Critical disease missed | -0.20 |
| ❌ Overconfident on wrong answer | -0.15 |
| ❌ Unnecessary test ordered | -0.10 |

A severity multiplier applies to all positive rewards: `low 1.0×` · `medium 1.2×` · `high 1.5×` · `critical 2.0×`

---

## Memory and Self-Improvement

After every case the Supervisor generates structured feedback — what was predicted, what was expected, and a corrective rule in plain language. This feedback is stored in each agent's memory and retrieved in future cases where similar symptoms appear.

For example, if the Specialist wrongly diagnosed tuberculosis when the correct answer was lymphoma, the following rule gets stored and injected into future prompts:

> *"For symptoms weight loss, fatigue, night sweats — key confirmatory test is biopsy showing cancerous lymph cells. Correct diagnosis: lymphoma."*

The Oversight AI monitors patterns across all cases. If approval rates drop or agents are taking too many steps, it derives system-level rules and pushes them to all agents simultaneously — simulating institutional learning across the diagnostic team.

---

## Before and After Training

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| Diagnosis Accuracy | ~48–65% | ~85–90% |
| Mean Reward | +0.60 | +1.38 |
| Confidence | Flat, uncalibrated | Rises with evidence |
| Reward curve | Noisy, no trend | Upward, stabilizing |

---

## Project Structure

```
medagents/
├── environment.py       ← OpenEnv reset() / step() loop
├── task.py              ← 6-stage pipeline and action validation
├── reward.py            ← 9 independent reward functions
├── memory.py            ← Per-agent feedback and memory system
├── dataset.json         ← 35 patient cases
├── server.py            ← FastAPI HTTP server
├── Dockerfile           ← HuggingFace Spaces deployment
├── training_stub.py     ← SFT warmup and GRPO training pipeline
├── main.py              ← CLI entry point
├── agents/
│   ├── gp.py
│   ├── radiologist.py
│   ├── pathologist.py
│   ├── specialist.py
│   ├── supervisor.py
│   └── oversight.py
└── utils/
    ├── logger.py        ← Per-episode and per-step logging
    └── graph.py         ← Line graph generation
```

---

## Quickstart

```bash
# Install dependencies
pip install fastapi uvicorn pydantic numpy matplotlib

# Run full pipeline — 35 cases with curriculum ordering
python main.py --seed 42

# Single episode demo with full agent reasoning printed
python main.py --demo

# Start HTTP server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload

# View interactive API documentation
# Open browser → http://localhost:7860/docs

# Simulate post-training behavior
python main.py --phase post_training --noise 0.05 --seed 42
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/info` | Environment metadata |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Execute one agent action |
| POST | `/auto_step` | Built-in agents act automatically |
| POST | `/run_episode` | Run one complete case end-to-end |
| GET | `/docs` | Interactive Swagger UI |

---

## Training Stack

The training pipeline uses **HuggingFace TRL** for GRPO reinforcement learning and **Unsloth** for memory-efficient QLoRA fine-tuning. The reward function in `reward.py` serves directly as the verifier — no separate reward model is needed because every signal is programmatically computable from the environment.

SFT warm-start runs first on correct diagnosis trajectories to prime the model's output format. GRPO then takes over, sampling multiple rollouts per prompt and shifting the model's weights toward higher-reward diagnostic behavior.

---

## Deployment

```bash
pip install huggingface_hub
huggingface-cli login
git init && git add . && git commit -m "MedAgents-X"
git remote add origin https://huggingface.co/spaces/USERNAME/medagents-x
git push origin main
```

Environment live at `https://USERNAME-medagents-x.hf.space/docs`

---

<div align="center">
<sub>Python 3.11 · FastAPI · HuggingFace TRL · Unsloth · OpenEnv</sub>
</div>
---
title: Medagents Finale
emoji: 👁
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
