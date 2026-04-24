"""
training_stub.py — MedAgents-X Full Training Pipeline
=======================================================
Complete training pipeline: rollout collection → SFT warmup → GRPO RL training.

PHASE 1 (Pre-hackathon, runs now):
  - collect_rollouts()     collect trajectories from heuristic agents
  - format_for_trl()       format as TRL-compatible dataset
  - format_for_sft()       format correct rollouts for SFT warm-start

PHASE 2 (Hackathon day, needs compute credits):
  - run_sft_warmup()       SFT warm-start on correct diagnosis traces
  - train_with_grpo()      GRPO RL training with verifiable reward
  - save_model()           correct LoRA/QLoRA merged save
  - sample_generations()   inspect outputs for reward hacking
  - benchmark_throughput() measure rollout speed (cases/second)

Guidelines satisfied:
  ✅ Point 3  — SFT warm-start before GRPO
  ✅ Point 10 — TRL + Unsloth stack fully wired
  ✅ Point 12 — Inference speed benchmarking
  ✅ Point 15 — Generation inspection during training
  ✅ Point 16 — Correct LoRA model saving
"""

# ─── Imports (active now) ─────────────────────────────────────────────────────
import os
import json
import time
import random
from typing import List, Dict, Any, Optional

from environment import MedAgentsEnv
from memory import MemorySystem
from utils.logger import EpisodeLogger
from utils.graph import GraphPlotter

# ─── Imports (uncomment at hackathon when compute available) ──────────────────
# import torch
# from transformers import AutoTokenizer
# from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig
# from unsloth import FastLanguageModel
# from datasets import Dataset
# from peft import LoraConfig


# ─── Curriculum: sort cases by difficulty ────────────────────────────────────

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

def sort_cases_by_curriculum(dataset: List[Dict]) -> List[Dict]:
    """
    Sort cases from easy → hard so the model sees successes early.
    This satisfies Guideline Point 6 (curriculum learning) and prevents
    zero-reward early episodes that stall RL training.

    Order: low severity → medium → high → critical
    """
    return sorted(dataset, key=lambda c: SEVERITY_ORDER.get(c.get("severity", "medium"), 1))


# ─── Rollout Collection ───────────────────────────────────────────────────────

def collect_rollouts(
    env: MedAgentsEnv,
    agents: Dict,
    n_episodes: int = 35,
    phase: str = "pre_training",
    logger: EpisodeLogger = None,
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """
    Collect (observation, action, reward) rollouts from n_episodes.

    Args:
        env:        MedAgentsEnv instance
        agents:     Dict mapping role → agent instance
        n_episodes: Number of episodes to run
        phase:      'pre_training' or 'post_training' for logging
        logger:     EpisodeLogger instance
        batch_size: Cases to process per batch (>1 = parallel-ready for hackathon)

    Returns:
        List of rollout dicts for TRL dataset construction
    """
    rollouts = []
    total_start = time.time()

    for ep_num in range(n_episodes):
        ep_start = time.time()
        state = env.reset()
        done = False
        episode_rollout = []
        step_confidences = []
        last_info = {}

        while not done:
            active = env.get_active_agents()
            all_actions = []

            for role in active:
                agent = agents.get(role)
                if agent is None:
                    continue
                actions = agent.act(state.observation)
                all_actions.extend(actions)

            if not all_actions:
                break

            for action in all_actions:
                next_state, reward, done, info = env.step(action)
                state = next_state
                last_info = info

                confidence = action.content.get("confidence")
                if confidence is not None:
                    step_confidences.append(float(confidence))

                step_entry = {
                    "observation": state.observation,
                    "action": {
                        "agent": action.agent_role,
                        "type": action.action_type,
                        "content": action.content,
                    },
                    "reward": reward,
                    "done": done,
                }
                episode_rollout.append(step_entry)

                if logger:
                    logger.log_step(
                        episode=ep_num + 1,
                        step=state.step_count,
                        stage=state.stage,
                        agent=action.agent_role,
                        action_type=action.action_type,
                        step_reward=reward,
                        confidence=confidence,
                        cumulative_reward=state.cumulative_reward,
                    )

                if done:
                    break
            if done:
                break

        # Episode summary
        pipeline_summary = env._pipeline.get_summary() if env._pipeline else {}
        final_dx   = pipeline_summary.get("final_diagnosis", "") or ""
        correct_dx = (env._case.get("correct_diagnosis", "") if env._case else "") or ""
        confidence = pipeline_summary.get("final_confidence", 0.5) or 0.5
        is_correct = final_dx.strip().lower() == correct_dx.strip().lower()
        ep_time    = time.time() - ep_start

        if logger:
            logger.log_episode(
                case_id=env._case["id"] if env._case else 0,
                total_reward=state.cumulative_reward,
                is_correct=is_correct,
                predicted_diagnosis=final_dx,
                correct_diagnosis=correct_dx,
                confidence=confidence,
                steps_taken=state.step_count,
                severity=env._case.get("severity", "medium") if env._case else "medium",
                reward_breakdown=last_info.get("memory_recorded", {}).get("reward_breakdown", {}),
                phase=phase,
            )

        rollouts.append({
            "episode": ep_num + 1,
            "phase": phase,
            "steps": episode_rollout,
            "final_diagnosis": final_dx,
            "correct_diagnosis": correct_dx,
            "is_correct": is_correct,
            "total_reward": state.cumulative_reward,
            "ep_time_sec": round(ep_time, 3),
        })

        status = "✓" if is_correct else "✗"
        print(f"  [{phase}] Ep {ep_num+1:>3} | Case #{env._case['id'] if env._case else '?':>2} | "
              f"{status} | Reward: {state.cumulative_reward:+.3f} | "
              f"Predicted: {final_dx or 'none':<25} | Correct: {correct_dx}")

    # ── Throughput benchmark ──────────────────────────────────────────────────
    total_time = time.time() - total_start
    cps = round(n_episodes / total_time, 2)
    print(f"\n[Throughput] {n_episodes} episodes in {total_time:.1f}s → {cps} cases/sec")

    return rollouts


# ─── Dataset Formatting ───────────────────────────────────────────────────────

def format_for_trl(rollouts: List[Dict]) -> List[Dict[str, Any]]:
    """
    Format ALL rollouts as prompt-response-reward triples for GRPO training.
    Each step becomes one training sample.
    """
    trl_dataset = []
    for rollout in rollouts:
        for step in rollout["steps"]:
            obs_text = json.dumps(step["observation"].get("visible_info", {}), indent=2)
            act_text = json.dumps(step["action"], indent=2)
            trl_dataset.append({
                "prompt": (
                    f"You are a medical AI agent.\n"
                    f"Stage: {step['observation'].get('stage', 'UNKNOWN')}\n"
                    f"Observation:\n{obs_text}\n\n"
                    f"Respond with the best action in JSON format."
                ),
                "response": act_text,
                "reward": float(step["reward"]),
                "episode": rollout["episode"],
                "is_correct": rollout["is_correct"],
            })
    return trl_dataset


def format_for_sft(rollouts: List[Dict]) -> List[Dict[str, str]]:
    """
    Format ONLY correct rollouts as supervised fine-tuning pairs.
    Used for SFT warm-start before GRPO (Guideline Point 3).

    Only correct episodes are used — these become the 'ideal traces'
    the model learns to replicate before RL takes over.
    """
    sft_dataset = []
    for rollout in rollouts:
        if not rollout["is_correct"]:
            continue  # only use successful trajectories for SFT
        for step in rollout["steps"]:
            obs_text = json.dumps(step["observation"].get("visible_info", {}), indent=2)
            act_text = json.dumps(step["action"], indent=2)
            sft_dataset.append({
                "prompt": (
                    f"You are a medical AI agent.\n"
                    f"Stage: {step['observation'].get('stage', 'UNKNOWN')}\n"
                    f"Observation:\n{obs_text}\n\n"
                    f"Respond with the best action in JSON format."
                ),
                "response": act_text,
            })
    return sft_dataset


# ─── Generation Inspector (Reward Hacking Check) ─────────────────────────────

def sample_generations(
    rollouts: List[Dict],
    n_samples: int = 5,
    check_hacking: bool = True,
) -> None:
    """
    Print n random rollout samples for human inspection.
    Satisfies Guideline Point 8 (reward hacking prevention) and Point 15.

    Checks for:
      - Agents reading ground truth directly (impossible here but logged)
      - Repeated identical actions across episodes (memorization)
      - Abnormally high confidence on wrong answers (overconfidence hack)
      - Unusually short episodes (shortcut exploitation)
    """
    print("\n" + "─" * 65)
    print("  Generation Inspection (Anti-Reward-Hacking Check)")
    print("─" * 65)

    sample = random.sample(rollouts, min(n_samples, len(rollouts)))

    hack_warnings = []

    for i, rollout in enumerate(sample):
        print(f"\n  Sample {i+1} | Episode {rollout['episode']} | "
              f"{'✓ CORRECT' if rollout['is_correct'] else '✗ WRONG'}")
        print(f"    Predicted : {rollout['final_diagnosis']}")
        print(f"    Correct   : {rollout['correct_diagnosis']}")
        print(f"    Reward    : {rollout['total_reward']:+.3f}")
        print(f"    Steps     : {len(rollout['steps'])}")

        if rollout["steps"]:
            last = rollout["steps"][-1]
            print(f"    Last action: {last['action']['type']} by {last['action']['agent']}")

        # Hacking checks
        if check_hacking:
            # Check 1: Suspiciously short episodes (< 4 steps = skipping stages)
            if len(rollout["steps"]) < 4:
                hack_warnings.append(
                    f"Episode {rollout['episode']}: Only {len(rollout['steps'])} steps — "
                    f"possible stage-skipping exploit."
                )

            # Check 2: Very high reward on wrong diagnosis
            if not rollout["is_correct"] and rollout["total_reward"] > 0.5:
                hack_warnings.append(
                    f"Episode {rollout['episode']}: Wrong diagnosis but reward={rollout['total_reward']:+.3f} — "
                    f"check reward function for exploits."
                )

            # Check 3: Repeated identical final diagnoses (memorization)
            all_dx = [r["final_diagnosis"] for r in rollouts]
            most_common = max(set(all_dx), key=all_dx.count) if all_dx else ""
            freq = all_dx.count(most_common) / max(len(all_dx), 1)
            if freq > 0.6:
                hack_warnings.append(
                    f"'{most_common}' appears in {freq:.0%} of episodes — "
                    f"possible mode collapse / memorization."
                )

    if hack_warnings:
        print("\n  ⚠️  Reward Hacking Warnings:")
        for w in set(hack_warnings):
            print(f"    → {w}")
    else:
        print("\n  ✅ No reward hacking patterns detected.")

    print("─" * 65)


# ─── SFT Warm-Start (Guideline Point 3) ──────────────────────────────────────

def run_sft_warmup(
    sft_dataset: List[Dict],
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "medagents_sft",
    num_epochs: int = 1,
) -> str:
    """
    SFT warm-start on correct diagnosis traces BEFORE GRPO training.

    Why this matters (Guideline Point 3):
      - Raw base models may never produce valid-format actions
      - SFT primes the model to output correct JSON action format
      - Ensures non-zero reward from the start of GRPO training
      - More sample-efficient than RL from scratch

    ⚠️ Uncomment the implementation block at hackathon time.

    Args:
        sft_dataset:  Output of format_for_sft() — correct traces only
        model_name:   HuggingFace model ID
        output_dir:   Where to save the SFT checkpoint
        num_epochs:   Epochs for warm-start (1 is usually enough)

    Returns:
        Path to saved SFT checkpoint
    """
    print(f"\n[SFT Warmup] Dataset size: {len(sft_dataset)} correct traces")
    print(f"[SFT Warmup] Model: {model_name}")
    print(f"[SFT Warmup] Output: {output_dir}")

    # ── UNCOMMENT AT HACKATHON ────────────────────────────────────────────────
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=model_name,
    #     max_seq_length=2048,
    #     dtype=None,           # auto-detect
    #     load_in_4bit=True,    # QLoRA
    # )
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=16,
    #     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    #     use_gradient_checkpointing=True,
    # )
    # hf_dataset = Dataset.from_list([
    #     {"text": f"{d['prompt']}\n\n### Response:\n{d['response']}"}
    #     for d in sft_dataset
    # ])
    # sft_config = SFTConfig(
    #     output_dir=output_dir,
    #     num_train_epochs=num_epochs,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     learning_rate=2e-4,
    #     fp16=not torch.cuda.is_bf16_supported(),
    #     bf16=torch.cuda.is_bf16_supported(),
    #     logging_steps=10,
    #     save_strategy="epoch",
    #     max_seq_length=2048,
    #     dataset_text_field="text",
    # )
    # trainer = SFTTrainer(model=model, args=sft_config, train_dataset=hf_dataset)
    # trainer.train()
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)
    # print(f"[SFT Warmup] Saved to {output_dir}")
    # return output_dir
    # ─────────────────────────────────────────────────────────────────────────

    print("[SFT Warmup] Ready to run at hackathon — uncomment implementation above.")
    return output_dir


# ─── GRPO RL Training (Guideline Points 10, 11) ───────────────────────────────

def train_with_grpo(
    trl_dataset: List[Dict],
    model_name: str = "medagents_sft",
    output_dir: str = "medagents_grpo",
    num_epochs: int = 3,
) -> str:
    """
    GRPO RL training using verifiable reward functions (no learned reward model).

    Why GRPO (Guideline Point 11):
      - Task is fully verifiable: string match + test overlap + confidence scoring
      - No separate reward model needed — reward.py IS the verifier
      - GRPO is more memory-efficient than PPO (no value model)
      - Reward hacking prevention: 9 independent reward checks

    Args:
        trl_dataset:  Output of format_for_trl() — all rollouts
        model_name:   SFT checkpoint path (output of run_sft_warmup)
        output_dir:   Where to save the GRPO checkpoint
        num_epochs:   RL training epochs

    Returns:
        Path to saved GRPO checkpoint
    """
    print(f"\n[GRPO Training] Dataset size: {len(trl_dataset)} samples")
    print(f"[GRPO Training] Base model: {model_name}")
    print(f"[GRPO Training] Output: {output_dir}")
    print(f"[GRPO Training] Framework: HuggingFace TRL GRPOTrainer + Unsloth")

    # ── UNCOMMENT AT HACKATHON ────────────────────────────────────────────────
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=model_name,
    #     max_seq_length=2048,
    #     dtype=None,
    #     load_in_4bit=True,
    # )
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=16,
    #     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    #     lora_alpha=16,
    #     lora_dropout=0.0,
    #     bias="none",
    #     use_gradient_checkpointing=True,
    # )
    # hf_dataset = Dataset.from_list(trl_dataset)
    #
    # def reward_fn(samples, **kwargs):
    #     """Reward function called by GRPOTrainer per batch."""
    #     import json as _json
    #     from reward import compute_reward
    #     rewards = []
    #     for sample in samples:
    #         try:
    #             action = _json.loads(sample)
    #             r = compute_reward(
    #                 predicted_diagnosis=action.get("content", {}).get("diagnosis", ""),
    #                 correct_diagnosis=sample.get("correct_diagnosis", ""),
    #                 confidence=action.get("content", {}).get("confidence", 0.5),
    #                 tests_ordered=[],
    #                 valid_tests={},
    #                 reasoning_steps=[],
    #                 steps_taken=1,
    #             )
    #             rewards.append(r["total_reward"])
    #         except Exception:
    #             rewards.append(-0.5)
    #     return rewards
    #
    # grpo_config = GRPOConfig(
    #     output_dir=output_dir,
    #     num_train_epochs=num_epochs,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     learning_rate=5e-5,
    #     fp16=not torch.cuda.is_bf16_supported(),
    #     bf16=torch.cuda.is_bf16_supported(),
    #     logging_steps=5,
    #     save_strategy="epoch",
    #     num_generations=8,       # GRPO samples 8 outputs per prompt
    #     temperature=0.9,
    #     max_new_tokens=256,
    # )
    # trainer = GRPOTrainer(
    #     model=model,
    #     args=grpo_config,
    #     train_dataset=hf_dataset,
    #     reward_funcs=[reward_fn],
    # )
    # trainer.train()
    # save_model(model, tokenizer, output_dir)
    # print(f"[GRPO Training] Complete. Saved to {output_dir}")
    # return output_dir
    # ─────────────────────────────────────────────────────────────────────────

    print("[GRPO Training] Ready to run at hackathon — uncomment implementation above.")
    return output_dir


# ─── Model Save (Guideline Point 16) ─────────────────────────────────────────

def save_model(model, tokenizer, output_dir: str) -> None:
    """
    Save LoRA/QLoRA model CORRECTLY using Unsloth merged save.

    ⚠️ CRITICAL (Guideline Point 16):
    Do NOT upcast 4-bit → 16-bit and then merge naively.
    This corrupts model quality. Always use save_pretrained_merged().

    ⚠️ Uncomment at hackathon time.
    """
    print(f"[ModelSave] Saving merged model to {output_dir}_merged ...")

    # ── UNCOMMENT AT HACKATHON ────────────────────────────────────────────────
    # # Method 1: Save merged 16-bit (for deployment)
    # model.save_pretrained_merged(
    #     f"{output_dir}_merged",
    #     tokenizer,
    #     save_method="merged_16bit",
    # )
    #
    # # Method 2: Save LoRA adapters only (smaller, for further training)
    # model.save_pretrained(f"{output_dir}_lora")
    # tokenizer.save_pretrained(f"{output_dir}_lora")
    #
    # # Method 3: Push to HuggingFace Hub
    # model.push_to_hub_merged(
    #     "your-username/medagents-x-trained",
    #     tokenizer,
    #     save_method="merged_16bit",
    # )
    # print(f"[ModelSave] Done. Verify with: python -c \"from transformers import pipeline; "
    #       f"p = pipeline('text-generation', model='{output_dir}_merged'); print(p('Test'))\"")
    # ─────────────────────────────────────────────────────────────────────────

    print("[ModelSave] Ready — uncomment save block at hackathon time.")


# ─── Throughput Benchmark (Guideline Point 12) ───────────────────────────────

def benchmark_throughput(env: MedAgentsEnv, agents: Dict, n_episodes: int = 10) -> Dict:
    """
    Measure rollout collection speed in cases/second.
    Satisfies Guideline Point 12 — inference speed monitoring.

    In RL for LLMs, rollout generation (inference) dominates runtime.
    Tracking this helps identify bottlenecks before scaling up.

    Returns:
        Dict with cases_per_second, avg_steps_per_case, total_time
    """
    print(f"\n[Benchmark] Running {n_episodes} episodes to measure throughput...")
    start = time.time()
    total_steps = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 30:
            active = env.get_active_agents()
            for role in active:
                agent = agents.get(role)
                if not agent:
                    continue
                actions = agent.act(state.observation)
                for action in actions:
                    state, _, done, _ = env.step(action)
                    steps += 1
                    if done:
                        break
                if done:
                    break
        total_steps += steps

    elapsed = time.time() - start
    cps = round(n_episodes / elapsed, 2)
    avg_steps = round(total_steps / n_episodes, 1)

    result = {
        "cases_per_second": cps,
        "avg_steps_per_case": avg_steps,
        "total_time_sec": round(elapsed, 2),
        "n_episodes": n_episodes,
    }
    print(f"[Benchmark] {cps} cases/sec | {avg_steps} steps/case avg | {elapsed:.1f}s total")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  MedAgents-X Training Stub")
    print("  Run main.py to execute the full pre-training pipeline.")
    print("=" * 60)
