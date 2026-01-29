from src.training.train_population import train_population
from src.evaluation.evaluate import evaluate_detailed
from src.utils.plotting import (
    plot_learning_curve,
    plot_episode_lengths,
    plot_confusion_matrix,
)
from src.utils.metrics import confusion_matrix_rl


def main():
    # =============================
    # Train population agent
    # =============================
    agent = train_population()

    # =============================
    # Detailed evaluation
    # =============================
    metrics = evaluate_detailed(agent)

    print("\n===== FINAL METRICS =====")
    print(f"Average Reward      : {metrics['avg_reward']:.2f}")
    print(f"Reward Std Dev      : {metrics['std_reward']:.2f}")
    print(f"Average Ep Length   : {metrics['avg_length']:.1f}")
    print(f"Success Rate        : {metrics['success_rate']*100:.1f}%")

    # =============================
    # Confusion Matrix (RL-style)
    # =============================
    cm = confusion_matrix_rl(metrics["successes"])

    print("\nConfusion Matrix:")
    print(cm)

    # =============================
    # Visualization
    # =============================
    plot_learning_curve(metrics["rewards"])
    plot_episode_lengths(metrics["lengths"])
    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()
