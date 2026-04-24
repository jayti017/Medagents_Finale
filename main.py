"""
main.py — MedAgents-X Entry Point
===================================
Runs the full pre-training pipeline:
  1. Initialize environment, agents, memory, logger
  2. Apply curriculum ordering (easy → hard cases)
  3. Run 35 episodes with rollout collection
  4. Inspect generations for reward hacking
  5. Benchmark throughput
  6. Log all metrics (individual reward columns)
  7. Generate line graphs
  8. Prepare SFT + TRL datasets

Usage:
    python main.py
    python main.py --episodes 35 --seed 42
    python main.py --demo
    python main.py --benchmark
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import MedAgentsEnv
from memory import MemorySystem
from utils.logger import EpisodeLogger
from utils.graph import GraphPlotter
from training_stub import (
    collect_rollouts, format_for_trl, format_for_sft,
    sort_cases_by_curriculum, sample_generations, benchmark_throughput
)

from agents.gp import GPAgent
from agents.radiologist import RadiologistAgent
from agents.pathologist import PathologistAgent
from agents.specialist import SpecialistAgent
from agents.supervisor import SupervisorAgent
from agents.oversight import OversightAgent


def build_agents(memory: MemorySystem, noise_level: float = 0.35) -> dict:
    """Instantiate all agents with memory injection."""
    return {
        "gp":          GPAgent(memory=memory.get_agent_memory("gp"), noise_level=noise_level),
        "radiologist": RadiologistAgent(memory=memory.get_agent_memory("radiologist"), noise_level=noise_level),
        "pathologist": PathologistAgent(memory=memory.get_agent_memory("pathologist"), noise_level=noise_level),
        "specialist":  SpecialistAgent(memory=memory.get_agent_memory("specialist"), noise_level=noise_level),
        "supervisor":  SupervisorAgent(memory=memory.get_agent_memory("supervisor"), noise_level=noise_level),
        "oversight":   OversightAgent(memory=memory),
    }


def run_pipeline(
    n_episodes: int = 35,
    seed: int = 0,
    run_name: str = "medagents_run",
    phase: str = "pre_training",
    noise_level: float = 0.35,
    plot_graphs: bool = True,
    use_curriculum: bool = True,
    inspect_generations: bool = True,
) -> dict:
    """
    Full MedAgents-X pipeline run.

    Args:
        n_episodes:           Number of episodes to run
        seed:                 Random seed for reproducibility
        run_name:             Label for log files
        phase:                'pre_training' or 'post_training'
        noise_level:          Agent noise (0.35 = poor pre-training baseline)
        plot_graphs:          Whether to generate PNG graphs
        use_curriculum:       Sort cases easy→hard (Guideline Point 6)
        inspect_generations:  Sample rollouts for hacking check (Guideline Points 8, 15)
    """

    print("\n" + "═" * 65)
    print("  MedAgents-X: Multi-Agent Clinical Decision System")
    print(f"  Phase: {phase.upper()} | Episodes: {n_episodes} | Seed: {seed}")
    print("═" * 65 + "\n")

    # ── Setup ─────────────────────────────────────────────────────────────────
    memory = MemorySystem()
    logger = EpisodeLogger(run_name=f"{run_name}_{phase}")
    env    = MedAgentsEnv(shuffle=False, max_steps_per_episode=25, memory_system=memory, seed=seed)
    agents = build_agents(memory, noise_level=noise_level)

    print(f"[Setup] Environment  : {env.n_cases} cases loaded from dataset")
    print(f"[Setup] Agents       : {list(agents.keys())}")
    print(f"[Setup] Memory       : {memory.filepath}")
    print(f"[Setup] Logger       : {logger.episode_log_path}")
    print(f"[Setup] Curriculum   : {'ON (easy→hard)' if use_curriculum else 'OFF (original order)'}")

    # ── Curriculum: sort dataset easy → hard (Guideline Point 6) ─────────────
    if use_curriculum:
        env.dataset = sort_cases_by_curriculum(env.dataset)
        env._setup_queue()
        print(f"[Curriculum] Cases reordered: "
              f"{[c['severity'] for c in env.dataset[:5]]} ... (first 5 severities)")

    # ── Run episodes ──────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Running Episodes")
    print("─" * 65)

    rollouts = collect_rollouts(
        env=env,
        agents=agents,
        n_episodes=min(n_episodes, env.n_cases),
        phase=phase,
        logger=logger,
        batch_size=1,
    )

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = logger.save_summary()

    print("\n" + "─" * 65)
    print("  Summary")
    print("─" * 65)
    print(f"  Total Episodes  : {summary.get('total_episodes', 0)}")
    print(f"  Accuracy        : {summary.get('accuracy', 0):.1%}")
    print(f"  Mean Reward     : {summary.get('mean_reward', 0):+.4f}")
    print(f"  Mean Confidence : {summary.get('mean_confidence', 0):.4f}")

    # ── Reward component breakdown (Guideline Point 15) ───────────────────────
    rc = summary.get("reward_components", {})
    if rc:
        print("\n  Reward Component Averages:")
        for k, v in rc.items():
            print(f"    {k:<35} : {v:+.4f}")

    # ── Generation inspection (Guideline Points 8, 15) ────────────────────────
    if inspect_generations and rollouts:
        sample_generations(rollouts, n_samples=5, check_hacking=True)

    # ── Generate graphs ───────────────────────────────────────────────────────
    if plot_graphs:
        print("\n[Graphs] Generating line graphs...")
        plotter = GraphPlotter()
        saved = plotter.plot_all(summary)
        if saved:
            print(f"[Graphs] Saved {len(saved)} graph(s).")

    # ── TRL + SFT dataset preparation ─────────────────────────────────────────
    trl_data = format_for_trl(rollouts)
    sft_data = format_for_sft(rollouts)
    print(f"\n[TRL] {len(trl_data)} GRPO training samples ready.")
    print(f"[SFT] {len(sft_data)} SFT warm-start samples (correct episodes only).")

    print("\n" + "═" * 65)
    print(f"  {phase.replace('_',' ').title()} pipeline complete.")
    print("  Logs  →", logger.episode_log_path)
    print("  Graphs→", os.path.join(os.path.dirname(logger.episode_log_path), "graphs"))
    print("═" * 65 + "\n")

    return summary


def demo_single_episode(seed: int = 7) -> None:
    """Run one episode verbosely — shows full agent reasoning pipeline."""
    print("\n" + "═" * 65)
    print("  MedAgents-X: Single Episode Demo")
    print("═" * 65)

    memory = MemorySystem()
    env    = MedAgentsEnv(shuffle=False, seed=seed)
    agents = build_agents(memory, noise_level=0.35)

    state = env.reset()
    env.render()

    done     = False
    step_num = 0

    while not done:
        active = env.get_active_agents()
        print(f"\n  ── Step {step_num + 1} | Active: {active} ──")

        for role in active:
            agent = agents.get(role)
            if not agent:
                continue
            actions = agent.act(state.observation)
            for action in actions:
                next_state, reward, done, info = env.step(action)
                state = next_state
                print(f"    [{role.upper()}] {action.action_type}")
                if action.content.get("reasoning"):
                    print(f"      reasoning : {action.content['reasoning'][:120]}...")
                if action.content.get("diagnosis"):
                    print(f"      diagnosis : {action.content['diagnosis']} "
                          f"(conf={action.content.get('confidence', '?')})")
                print(f"      step_reward: {reward:+.4f}")
                if done:
                    break
            if done:
                break

        step_num += 1
        if step_num > 30:
            print("  [Demo] Max steps reached.")
            break

    print(f"\n  Final Cumulative Reward: {state.cumulative_reward:+.4f}")
    env.render()


def run_benchmark(seed: int = 0) -> None:
    """Benchmark rollout throughput — cases per second."""
    memory = MemorySystem()
    env    = MedAgentsEnv(shuffle=False, seed=seed)
    agents = build_agents(memory, noise_level=0.35)
    benchmark_throughput(env, agents, n_episodes=10)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MedAgents-X Pipeline Runner")
    p.add_argument("--episodes",        type=int,   default=35,             help="Number of episodes")
    p.add_argument("--seed",            type=int,   default=0,              help="Random seed")
    p.add_argument("--run-name",        type=str,   default="medagents",    help="Log file label")
    p.add_argument("--phase",           type=str,   default="pre_training", help="pre_training or post_training")
    p.add_argument("--noise",           type=float, default=0.35,           help="Agent noise 0.0–1.0")
    p.add_argument("--no-graphs",       action="store_true",                help="Skip graphs")
    p.add_argument("--no-curriculum",   action="store_true",                help="Skip curriculum ordering")
    p.add_argument("--no-inspect",      action="store_true",                help="Skip generation inspection")
    p.add_argument("--demo",            action="store_true",                help="Single episode demo")
    p.add_argument("--benchmark",       action="store_true",                help="Throughput benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo:
        demo_single_episode()
    elif args.benchmark:
        run_benchmark()
    else:
        run_pipeline(
            n_episodes=args.episodes,
            seed=args.seed,
            run_name=args.run_name,
            phase=args.phase,
            noise_level=args.noise,
            plot_graphs=not args.no_graphs,
            use_curriculum=not args.no_curriculum,
            inspect_generations=not args.no_inspect,
        )
