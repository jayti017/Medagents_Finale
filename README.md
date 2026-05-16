---
title: MedAgents-X
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
<div align="center">

# MedAgents-X

**Multi-Agent Clinical Decision System**

*Six AI agents. One patient. A race to the right diagnosis.*

**Team Coders** — Sourav Bhardwaj · Tanishka · Jayti Bhardwaj

Scaler School of Technology · Meta OpenEnv Hackathon, Final Round · April 2026

</div>

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Space (Live Environment) | [spaces/Sourav2207/Medagents_Finale](https://huggingface.co/spaces/Sourav2207/Medagents) |
| 📓 Training Notebook (Colab) | [Open in Colab](https://colab.research.google.com/drive/1TKEMuIhnUBtcjSR_H-K9lIOJACj0YJU4?usp=sharing) |
| 🎥 Demo Video | [YouTube](https://youtu.be/6YHdNd5158I?si=XW44UsBBoSK9cUGs) |
| 📊 Results & Graphs | Accuracy Vs Episode |https://drive.google.com/file/d/1bTvPchhu1mVKAUrvBVrWT4n4dXEy6OBy/view?usp=drivesdk
                        |Reward vs Episode |https://drive.google.com/file/d/1709JuXOdzdrzlETghugvXdq5dnn-mmTZ/view?usp=drivesdk
                        |before vs after training|https://drive.google.com/file/d/1EnH0pGDTsOTAhFHZcngHSkvIS9bl6fkU/view?usp=drivesdk
                        |RL Training|https://drive.google.com/file/d/1HJzC2t9l4L4Q5M2NM2m3cRYpAldrDlSF/view?usp=drivesdk
| 🌐 Live API Docs | [Swagger UI]( https://huggingface.co/spaces/Sourav2207/Medagents) |

---

## The Problem

A patient walks in with fever, cough, and weight loss. The correct diagnosis could be flu, tuberculosis, or lung cancer. Getting it wrong can cost a life. Real diagnosis is not a single prediction — it is a structured process involving multiple specialists, delayed test results, and incomplete information at every step.

MedAgents-X simulates this entire clinical workflow as a reinforcement learning environment. Six AI agents with distinct medical roles collaborate across six pipeline stages, starting with only symptoms and progressively building toward a final diagnosis as evidence is revealed.

---

## How the Environment Works

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

The environment enforces **partial observability** — test results are hidden until Stage 2, forcing agents to reason under uncertainty exactly as real doctors do.

---

## OpenEnv Interface

Built on top of OpenEnv (latest release). Full reset/step loop:

```python
from openenv import OpenEnvClient

client = OpenEnvClient("https://Sourav2207-Medagents_Finale.hf.space")

# Start episode
state = client.reset()

# Execute action
next_state, reward, done, info = client.step({
    "agent_role": "gp",
    "action_type": "form_differential",
    "content": {
        "differential": ["flu", "tuberculosis"],
        "confidence": 0.75,
        "reasoning": "Fever and cough suggest respiratory infection"
    }
})
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/info` | Environment metadata |
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute one action |
| POST | `/auto_step` | Agents act automatically |
| POST | `/run_episode` | Run complete case end-to-end |
| GET | `/docs` | Interactive Swagger UI |

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

Severity multiplier: `low 1.0×` · `medium 1.2×` · `high 1.5×` · `critical 2.0×`

---

## Training

### Framework
- **SFT Warm-Start:** HuggingFace TRL `SFTTrainer` on correct diagnosis traces
- **RL Training:** Manual reward-weighted policy gradient loop
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` with LoRA (r=8)
- **Full training notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1TKEMuIhnUBtcjSR_H-K9lIOJACj0YJU4?usp=sharing)

### Training Pipeline

```
Pre-training rollouts (noise=0.50)
        ↓
SFT warm-start on correct traces (1 epoch, loss: 1.70 → 1.50)
        ↓
Reward-weighted RL training (2 epochs, 60 samples)
        ↓
Post-training evaluation (noise=0.02)
```

---

## Results

### Before vs After Training

| Metric | Before Training | After Training | Improvement |
|--------|----------------|----------------|-------------|
| Diagnosis Accuracy | ~45% | ~88% | **+43%** |
| Mean Reward | +0.55 | +1.42 | **+0.87** |
| SFT Loss | 1.70 | 1.50 | **↓ 11.7%** |

### Training Curves

![Reward vs Episodes](https://drive.google.com/file/d/1709JuXOdzdrzlETghugvXdq5dnn-mmTZ/view?usp=drivesdk)

**Before vs After Training**

![Before vs After](https://drive.google.com/file/d/1EnH0pGDTsOTAhFHZcngHSkvIS9bl6fkU/view?usp=drivesdk)

**RL Training Curves**

![RL Training](https://drive.google.com/file/d/1HJzC2t9l4L4Q5M2NM2m3cRYpAldrDlSF/view?usp=drivesdk)

---

## Dataset

35 real patient cases spanning infectious diseases, cancers, cardiac emergencies, metabolic disorders and neurological conditions.

| Case | Symptoms | Correct Diagnosis |
|------|----------|-------------------|
| 5 | Chest pain, shortness of breath | Heart attack |
| 6 | Headache, vision problems, vomiting | Brain tumor |
| 11 | Weight loss, fatigue, night sweats | Lymphoma |
| 22 | Severe headache, neck stiffness | Meningitis |
| 33 | Fever, confusion | Sepsis |

---

## Memory and Self-Improvement

After every case the Supervisor generates structured feedback stored in each agent's memory. Future cases retrieve relevant past feedback by symptom overlap and inject it into agent prompts — simulating self-improvement between episodes without retraining.

Example learned rule stored after a wrong diagnosis:

> *"For symptoms weight loss, fatigue, night sweats — key confirmatory test is biopsy showing cancerous lymph cells. Correct diagnosis: lymphoma."*

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
├── server/app.py        ← OpenEnv entry point
├── openenv.yaml         ← OpenEnv configuration
├── pyproject.toml       ← Package configuration
├── Dockerfile           ← HuggingFace Spaces deployment
├── training_stub.py     ← SFT + RL training pipeline
├── main.py              ← CLI entry point
├── agents/
│   ├── gp.py · radiologist.py · pathologist.py
│   ├── specialist.py · supervisor.py · oversight.py
└── utils/
    ├── logger.py
    └── graph.py
```

---

## Quickstart

```bash
# Install
pip install fastapi uvicorn pydantic numpy matplotlib openenv

# Run pipeline
python main.py --seed 42

# Start server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload

# View API docs
# Open browser → http://localhost:7860/docs
```

---

## Run on HuggingFace Space

```python
import requests

BASE = "https://huggingface.co/spaces/Sourav2207/Medagents"

# Health check
print(requests.get(f"{BASE}/health").json())

# Start episode
state = requests.post(f"{BASE}/reset", json={}).json()
print("Case:", state["state"]["case_id"])
print("Symptoms:", state["state"]["observation"]["visible_info"]["symptoms"])

# Run full episode
result = requests.post(f"{BASE}/run_episode", json={}).json()
print("Diagnosis:", result["final_diagnosis"])
print("Reward:", result["total_reward"])
```

---

<div align="center">
<sub>Python 3.11 · FastAPI · HuggingFace TRL · OpenEnv · Qwen2.5</sub>
</div>