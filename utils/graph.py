"""
utils/graph.py — MedAgents-X Graph Visualization
==================================================
Generates LINE GRAPHS ONLY (as required by hackathon criteria).

Graphs produced:
  1. reward_vs_episodes.png         — Total reward over training episodes
  2. accuracy_vs_episodes.png       — Diagnosis accuracy over training episodes
  3. confidence_vs_steps.png        — Confidence trace within a case
  4. before_vs_after_reward.png     — Pre-training vs post-training reward comparison
  5. reward_smoothed.png            — Smoothed reward curve (moving average)

Usage:
    from utils.graph import GraphPlotter
    plotter = GraphPlotter(output_dir="logs/graphs")
    plotter.plot_all(summary_data)
"""

import os
import json
from typing import List, Dict, Any, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


GRAPH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "graphs")

# ─── Color scheme ─────────────────────────────────────────────────────────────
COLORS = {
    "reward":          "#2196F3",
    "reward_smooth":   "#1565C0",
    "accuracy":        "#4CAF50",
    "confidence":      "#FF9800",
    "pre_training":    "#EF5350",
    "post_training":   "#66BB6A",
    "grid":            "#E0E0E0",
    "background":      "#FAFAFA",
}


class GraphPlotter:
    """
    Generates all required line graphs from logged episode data.
    Falls back to ASCII summary if matplotlib is not installed.
    """

    def __init__(self, output_dir: str = GRAPH_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _setup_axes(self, ax, title: str, xlabel: str, ylabel: str) -> None:
        """Apply consistent styling to all plots."""
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_facecolor(COLORS["background"])
        ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def plot_reward_vs_episodes(self, rewards: List[float], smoothed: Optional[List[float]] = None) -> str:
        """Line graph: reward over episodes."""
        if not MATPLOTLIB_AVAILABLE:
            return self._ascii_fallback("reward_vs_episodes", rewards)

        fig, ax = plt.subplots(figsize=(10, 5))
        episodes = list(range(1, len(rewards) + 1))

        ax.plot(episodes, rewards, color=COLORS["reward"], alpha=0.4, linewidth=1.0, label="Raw Reward")
        if smoothed:
            ax.plot(episodes, smoothed, color=COLORS["reward_smooth"], linewidth=2.2, label="Smoothed (MA-5)")

        ax.axhline(0, color="#9E9E9E", linestyle=":", linewidth=0.8)
        self._setup_axes(ax, "Reward vs Episodes", "Episode", "Total Reward")
        ax.legend(framealpha=0.9)

        path = os.path.join(self.output_dir, "reward_vs_episodes.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Graph] Saved: {path}")
        return path

    def plot_accuracy_vs_episodes(self, accuracy_curve: List[float]) -> str:
        """Line graph: diagnosis accuracy over time."""
        if not MATPLOTLIB_AVAILABLE:
            return self._ascii_fallback("accuracy_vs_episodes", accuracy_curve)

        fig, ax = plt.subplots(figsize=(10, 5))
        episodes = list(range(1, len(accuracy_curve) + 1))

        ax.plot(episodes, accuracy_curve, color=COLORS["accuracy"], linewidth=2.2, label="Diagnosis Accuracy")
        ax.axhline(0.5, color="#9E9E9E", linestyle=":", linewidth=0.8, label="50% Baseline")
        ax.set_ylim(0, 1.05)

        self._setup_axes(ax, "Diagnosis Accuracy vs Episodes", "Episode", "Accuracy")
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.legend(framealpha=0.9)

        path = os.path.join(self.output_dir, "accuracy_vs_episodes.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Graph] Saved: {path}")
        return path

    def plot_confidence_vs_steps(
        self, step_confidences: List[float], episode_id: int = 1
    ) -> str:
        """Line graph: agent confidence across steps within a single episode."""
        if not MATPLOTLIB_AVAILABLE:
            return self._ascii_fallback(f"confidence_ep{episode_id}", step_confidences)

        fig, ax = plt.subplots(figsize=(10, 4))
        steps = list(range(1, len(step_confidences) + 1))

        ax.plot(steps, step_confidences, color=COLORS["confidence"], linewidth=2.0,
                marker="o", markersize=5, label=f"Episode #{episode_id} Confidence")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#9E9E9E", linestyle=":", linewidth=0.8, label="Threshold (0.5)")

        self._setup_axes(ax, f"Agent Confidence vs Steps (Episode #{episode_id})", "Step", "Confidence")
        ax.legend(framealpha=0.9)

        path = os.path.join(self.output_dir, f"confidence_vs_steps_ep{episode_id}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Graph] Saved: {path}")
        return path

    def plot_before_vs_after(
        self,
        pre_rewards: List[float],
        post_rewards: List[float],
    ) -> str:
        """
        Line graph: pre-training vs post-training reward curves.
        This is the key 'improvement' graph for hackathon judging.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[Graph] matplotlib not available — skipping before/after plot.")
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))

        # Both curves on same x-axis (episode index)
        pre_eps  = list(range(1, len(pre_rewards) + 1))
        post_eps = list(range(1, len(post_rewards) + 1))

        ax.plot(pre_eps, pre_rewards, color=COLORS["pre_training"], linewidth=2.0,
                linestyle="--", alpha=0.85, label="Before Training (Baseline)")
        ax.plot(post_eps, post_rewards, color=COLORS["post_training"], linewidth=2.2,
                label="After Training (Improved)")

        ax.axhline(0, color="#BDBDBD", linestyle=":", linewidth=0.7)
        self._setup_axes(ax, "Before vs After Training — Reward Improvement",
                         "Episode", "Total Reward")
        ax.legend(framealpha=0.9, fontsize=11)

        path = os.path.join(self.output_dir, "before_vs_after_reward.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Graph] Saved: {path}")
        return path

    def plot_all(self, summary: Dict[str, Any]) -> List[str]:
        """
        Generate all graphs from a summary dict (output of EpisodeLogger.save_summary()).
        Returns list of saved file paths.
        """
        paths = []

        rewards   = summary.get("reward_curve", [])
        smoothed  = summary.get("reward_curve_smoothed", [])
        acc_curve = summary.get("accuracy_curve", [])

        if rewards:
            paths.append(self.plot_reward_vs_episodes(rewards, smoothed))
        if acc_curve:
            paths.append(self.plot_accuracy_vs_episodes(acc_curve))

        # Before/after: if both phases present in summary
        by_phase = summary.get("by_phase", {})
        if "pre_training" in by_phase and "post_training" in by_phase:
            pre_r  = [by_phase["pre_training"]["mean_reward"]] * 35
            post_r = [by_phase["post_training"]["mean_reward"]] * 35
            paths.append(self.plot_before_vs_after(pre_r, post_r))

        return [p for p in paths if p]

    def _ascii_fallback(self, name: str, values: List[float]) -> str:
        """Print ASCII sparkline when matplotlib is not available."""
        if not values:
            return ""
        mn, mx = min(values), max(values)
        scale = mx - mn if mx != mn else 1
        bars = "▁▂▃▄▅▆▇█"
        spark = "".join(bars[int((v - mn) / scale * 7)] for v in values[:60])
        print(f"[Graph ASCII] {name}: {spark}")
        return f"ascii:{name}"

    @staticmethod
    def load_summary(path: str) -> Dict:
        with open(path) as f:
            return json.load(f)
