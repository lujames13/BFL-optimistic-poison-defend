"""Utility functions for blockchain federated learning.

This module provides utilities for visualization, data handling, and metrics
for federated learning with blockchain integration.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fl.utils")


def save_metrics(
    metrics: Dict[str, List[Tuple[int, float]]], 
    filename: str, 
    output_dir: str = "results"
) -> str:
    """Save metrics to a file.
    
    Args:
        metrics: Dictionary of metrics to save.
        filename: Name of the file to save to.
        output_dir: Directory to save to.
        
    Returns:
        Path to the saved file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to pickle file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(metrics, f)
    
    logger.info(f"Saved metrics to {output_path}")
    
    return output_path


def load_metrics(filename: str, input_dir: str = "results") -> Dict:
    """Load metrics from a file.
    
    Args:
        filename: Name of the file to load from.
        input_dir: Directory to load from.
        
    Returns:
        Dictionary of loaded metrics.
    """
    input_path = os.path.join(input_dir, filename)
    
    with open(input_path, "rb") as f:
        metrics = pickle.load(f)
    
    logger.info(f"Loaded metrics from {input_path}")
    
    return metrics


def plot_metrics(
    metrics: Dict[str, List[Tuple[int, float]]],
    metric_name: str = "accuracy",
    title: Optional[str] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot metrics.
    
    Args:
        metrics: Dictionary of metrics to plot.
        metric_name: Name of the metric to plot.
        title: Title of the plot.
        output_path: Path to save the plot to.
        
    Returns:
        Matplotlib figure.
    """
    plt.figure(figsize=(10, 6))
    
    # Extract metric data
    if metric_name in metrics:
        rounds = [m[0] for m in metrics[metric_name]]
        values = [m[1] for m in metrics[metric_name]]
        plt.plot(rounds, values, marker="o", linestyle="-", label=metric_name)
        
        # Add trend line
        z = np.polyfit(rounds, values, 1)
        p = np.poly1d(z)
        plt.plot(rounds, p(rounds), linestyle="--", alpha=0.5, color="gray")
    else:
        logger.warning(f"Metric {metric_name} not found in metrics dictionary.")
    
    # Set title and labels
    plt.title(title or f"{metric_name.capitalize()} over rounds")
    plt.xlabel("Round")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True, alpha=0.3)
    
    # Add legend if there are multiple metrics
    if len(metrics) > 1:
        plt.legend()
    
    # Save plot if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    
    return plt.gcf()


def plot_comparison(
    metrics_list: List[Dict[str, List[Tuple[int, float]]]],
    labels: List[str],
    metric_name: str = "accuracy",
    title: Optional[str] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot multiple metrics for comparison.
    
    Args:
        metrics_list: List of metrics dictionaries to compare.
        labels: List of labels for each metrics dictionary.
        metric_name: Name of the metric to plot.
        title: Title of the plot.
        output_path: Path to save the plot to.
        
    Returns:
        Matplotlib figure.
    """
    plt.figure(figsize=(12, 7))
    
    # Plot each set of metrics
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        if metric_name in metrics:
            rounds = [m[0] for m in metrics[metric_name]]
            values = [m[1] for m in metrics[metric_name]]
            plt.plot(rounds, values, marker="o", linestyle="-", label=label)
    
    # Set title and labels
    plt.title(title or f"Comparison of {metric_name}")
    plt.xlabel("Round")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    
    return plt.gcf()


def plot_attack_impact(
    honest_metrics: Dict[str, List[Tuple[int, float]]],
    attack_metrics: Dict[str, List[Tuple[int, float]]],
    defense_metrics: Dict[str, List[Tuple[int, float]]],
    metric_name: str = "accuracy",
    title: Optional[str] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot the impact of attacks and defenses.
    
    Args:
        honest_metrics: Metrics for honest scenario.
        attack_metrics: Metrics for scenario with attacks.
        defense_metrics: Metrics for scenario with attacks and defense.
        metric_name: Name of the metric to plot.
        title: Title of the plot.
        output_path: Path to save the plot to.
        
    Returns:
        Matplotlib figure.
    """
    return plot_comparison(
        metrics_list=[honest_metrics, attack_metrics, defense_metrics],
        labels=["Honest", "Under Attack", "With Defense"],
        metric_name=metric_name,
        title=title or f"Impact of Attacks and Defenses on {metric_name}",
        output_path=output_path
    )


def calculate_convergence_round(
    metrics: Dict[str, List[Tuple[int, float]]],
    metric_name: str = "accuracy",
    target_value: float = 0.9
) -> Optional[int]:
    """Calculate the round at which the metric reaches the target value.
    
    Args:
        metrics: Dictionary of metrics.
        metric_name: Name of the metric to use.
        target_value: Target value to reach.
        
    Returns:
        Round at which the target was reached, or None if not reached.
    """
    if metric_name not in metrics:
        logger.warning(f"Metric {metric_name} not found in metrics dictionary.")
        return None
    
    # Extract metric data
    rounds_values = metrics[metric_name]
    
    # Find the first round where the metric exceeds the target
    for round_num, value in rounds_values:
        if value >= target_value:
            return round_num
    
    # Target not reached
    return None


def save_model(
    model: tf.keras.Model, 
    filename: str, 
    output_dir: str = "models"
) -> str:
    """Save a TensorFlow model.
    
    Args:
        model: TensorFlow model to save.
        filename: Name of the file to save to.
        output_dir: Directory to save to.
        
    Returns:
        Path to the saved model.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    output_path = os.path.join(output_dir, filename)
    model.save(output_path)
    
    logger.info(f"Saved model to {output_path}")
    
    return output_path


def load_model(filename: str, input_dir: str = "models") -> tf.keras.Model:
    """Load a TensorFlow model.
    
    Args:
        filename: Name of the file to load from.
        input_dir: Directory to load from.
        
    Returns:
        Loaded TensorFlow model.
    """
    input_path = os.path.join(input_dir, filename)
    
    model = tf.keras.models.load_model(input_path)
    
    logger.info(f"Loaded model from {input_path}")
    
    return model


def combine_metrics(metrics_dict: Dict[str, Dict[str, List[Tuple[int, float]]]]) -> Dict:
    """Combine metrics from multiple runs.
    
    Args:
        metrics_dict: Dictionary of metrics dictionaries, keyed by run name.
        
    Returns:
        Dictionary of combined metrics.
    """
    combined = {}
    
    for run_name, metrics in metrics_dict.items():
        combined[run_name] = metrics
    
    return combined


def evaluate_defense_effectiveness(
    honest_metrics: Dict[str, List[Tuple[int, float]]],
    attack_metrics: Dict[str, List[Tuple[int, float]]],
    defense_metrics: Dict[str, List[Tuple[int, float]]],
    metric_name: str = "accuracy"
) -> Dict[str, float]:
    """Evaluate the effectiveness of the defense mechanism.
    
    Args:
        honest_metrics: Metrics for honest scenario.
        attack_metrics: Metrics for scenario with attacks.
        defense_metrics: Metrics for scenario with attacks and defense.
        metric_name: Name of the metric to use.
        
    Returns:
        Dictionary of effectiveness metrics.
    """
    # Get final metric values
    honest_final = honest_metrics[metric_name][-1][1]
    attack_final = attack_metrics[metric_name][-1][1]
    defense_final = defense_metrics[metric_name][-1][1]
    
    # Calculate absolute degradation due to attack
    attack_impact = honest_final - attack_final
    
    # Calculate recovery with defense
    defense_recovery = defense_final - attack_final
    
    # Calculate relative recovery (percentage of damage mitigated)
    if attack_impact > 0:
        recovery_percentage = (defense_recovery / attack_impact) * 100
    else:
        recovery_percentage = 0.0
    
    # Calculate convergence rounds
    honest_convergence = calculate_convergence_round(honest_metrics, metric_name)
    attack_convergence = calculate_convergence_round(attack_metrics, metric_name)
    defense_convergence = calculate_convergence_round(defense_metrics, metric_name)
    
    # Calculate convergence delay
    convergence_delay = None
    if honest_convergence is not None and defense_convergence is not None:
        convergence_delay = defense_convergence - honest_convergence
    
    return {
        "honest_final": honest_final,
        "attack_final": attack_final,
        "defense_final": defense_final,
        "attack_impact": attack_impact,
        "defense_recovery": defense_recovery,
        "recovery_percentage": recovery_percentage,
        "honest_convergence": honest_convergence,
        "attack_convergence": attack_convergence,
        "defense_convergence": defense_convergence,
        "convergence_delay": convergence_delay
    }


def print_effectiveness_report(effectiveness: Dict[str, float]) -> None:
    """Print a report of defense effectiveness.
    
    Args:
        effectiveness: Dictionary of effectiveness metrics.
    """
    print("=" * 50)
    print("DEFENSE EFFECTIVENESS REPORT")
    print("=" * 50)
    print(f"Final accuracy (honest): {effectiveness['honest_final']:.4f}")
    print(f"Final accuracy (under attack): {effectiveness['attack_final']:.4f}")
    print(f"Final accuracy (with defense): {effectiveness['defense_final']:.4f}")
    print("-" * 50)
    print(f"Attack impact: {effectiveness['attack_impact']:.4f} (absolute)")
    print(f"Defense recovery: {effectiveness['defense_recovery']:.4f} (absolute)")
    print(f"Recovery percentage: {effectiveness['recovery_percentage']:.1f}%")
    print("-" * 50)
    
    if effectiveness['honest_convergence'] is not None:
        print(f"Convergence round (honest): {effectiveness['honest_convergence']}")
    else:
        print("Convergence round (honest): Not reached")
    
    if effectiveness['attack_convergence'] is not None:
        print(f"Convergence round (under attack): {effectiveness['attack_convergence']}")
    else:
        print("Convergence round (under attack): Not reached")
    
    if effectiveness['defense_convergence'] is not None:
        print(f"Convergence round (with defense): {effectiveness['defense_convergence']}")
    else:
        print("Convergence round (with defense): Not reached")
    
    if effectiveness['convergence_delay'] is not None:
        print(f"Convergence delay: {effectiveness['convergence_delay']} rounds")
    else:
        print("Convergence delay: Cannot calculate")
    
    print("=" * 50)


def generate_thesis_plots(
    metrics_dict: Dict[str, Dict[str, List[Tuple[int, float]]]],
    output_dir: str = "thesis_figures"
) -> Dict[str, str]:
    """Generate plots for thesis or research paper.
    
    Args:
        metrics_dict: Dictionary of metrics dictionaries, keyed by experiment name.
        output_dir: Directory to save plots to.
        
    Returns:
        Dictionary mapping plot names to file paths.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = {}
    
    # Generate individual accuracy plots for each experiment
    for experiment, metrics in metrics_dict.items():
        output_path = os.path.join(output_dir, f"{experiment}_accuracy.png")
        plot_metrics(
            metrics, 
            metric_name="accuracy", 
            title=f"{experiment} - Accuracy over Rounds",
            output_path=output_path
        )
        plot_paths[f"{experiment}_accuracy"] = output_path
    
    # Generate comparison plots for different attack scenarios
    if all(key in metrics_dict for key in ["honest", "attack", "defense"]):
        # Accuracy comparison
        output_path = os.path.join(output_dir, "attack_defense_comparison.png")
        plot_attack_impact(
            metrics_dict["honest"],
            metrics_dict["attack"],
            metrics_dict["defense"],
            metric_name="accuracy",
            title="Impact of Attacks and Defenses on Accuracy",
            output_path=output_path
        )
        plot_paths["attack_defense_comparison"] = output_path
        
        # Loss comparison
        if "loss" in metrics_dict["honest"]:
            output_path = os.path.join(output_dir, "attack_defense_loss.png")
            plot_attack_impact(
                metrics_dict["honest"],
                metrics_dict["attack"],
                metrics_dict["defense"],
                metric_name="loss",
                title="Impact of Attacks and Defenses on Loss",
                output_path=output_path
            )
            plot_paths["attack_defense_loss"] = output_path
    
    # Generate comparison of different defense mechanisms
    defense_metrics = {k: v for k, v in metrics_dict.items() if "defense" in k}
    if len(defense_metrics) > 1:
        output_path = os.path.join(output_dir, "defense_comparison.png")
        plot_comparison(
            list(defense_metrics.values()),
            list(defense_metrics.keys()),
            metric_name="accuracy",
            title="Comparison of Different Defense Mechanisms",
            output_path=output_path
        )
        plot_paths["defense_comparison"] = output_path
    
    logger.info(f"Generated {len(plot_paths)} plots for thesis in {output_dir}")
    
    return plot_paths


def calculate_blockchain_metrics(
    blockchain_logs: List[Dict],
    num_rounds: int
) -> Dict[str, Union[float, int, List[float]]]:
    """Calculate blockchain-related metrics.
    
    Args:
        blockchain_logs: List of blockchain transaction logs.
        num_rounds: Number of federated learning rounds.
        
    Returns:
        Dictionary of blockchain metrics.
    """
    # Extract gas usage
    gas_used = [log.get("gasUsed", 0) for log in blockchain_logs]
    total_gas = sum(gas_used)
    avg_gas_per_round = total_gas / num_rounds if num_rounds > 0 else 0
    
    # Extract transaction times
    times = [log.get("timeMs", 0) for log in blockchain_logs]
    total_time = sum(times)
    avg_time_per_tx = total_time / len(times) if times else 0
    avg_time_per_round = total_time / num_rounds if num_rounds > 0 else 0
    
    # Count transactions by type
    tx_types = {}
    for log in blockchain_logs:
        tx_type = log.get("method", "unknown")
        tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
    
    return {
        "total_gas": total_gas,
        "avg_gas_per_round": avg_gas_per_round,
        "gas_by_round": [sum(gas_used[i:i+4]) for i in range(0, len(gas_used), 4)],  # Approx 4 txs per round
        "total_tx_time_ms": total_time,
        "avg_time_per_tx_ms": avg_time_per_tx,
        "avg_time_per_round_ms": avg_time_per_round,
        "tx_count": len(blockchain_logs),
        "tx_by_type": tx_types
    }


def print_blockchain_report(blockchain_metrics: Dict[str, Union[float, int, List[float]]]) -> None:
    """Print a report of blockchain metrics.
    
    Args:
        blockchain_metrics: Dictionary of blockchain metrics.
    """
    print("=" * 50)
    print("BLOCKCHAIN METRICS REPORT")
    print("=" * 50)
    print(f"Total gas used: {blockchain_metrics['total_gas']:,}")
    print(f"Average gas per round: {blockchain_metrics['avg_gas_per_round']:,.1f}")
    print(f"Total transaction time: {blockchain_metrics['total_tx_time_ms']:,.1f} ms ({blockchain_metrics['total_tx_time_ms']/1000:.1f} s)")
    print(f"Average time per transaction: {blockchain_metrics['avg_time_per_tx_ms']:,.1f} ms ({blockchain_metrics['avg_time_per_tx_ms']/1000:.2f} s)")
    print(f"Average time per round: {blockchain_metrics['avg_time_per_round_ms']:,.1f} ms ({blockchain_metrics['avg_time_per_round_ms']/1000:.2f} s)")
    print(f"Total transactions: {blockchain_metrics['tx_count']}")
    print("-" * 50)
    print("Transactions by type:")
    for tx_type, count in blockchain_metrics['tx_by_type'].items():
        print(f"  {tx_type}: {count}")
    print("=" * 50)


def plot_blockchain_metrics(
    blockchain_metrics: Dict[str, Union[float, int, List[float]]],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot blockchain metrics.
    
    Args:
        blockchain_metrics: Dictionary of blockchain metrics.
        output_path: Path to save the plot to.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot gas usage by round
    rounds = list(range(1, len(blockchain_metrics["gas_by_round"]) + 1))
    axs[0].bar(rounds, blockchain_metrics["gas_by_round"])
    axs[0].set_title("Gas Usage by Round")
    axs[0].set_xlabel("Round")
    axs[0].set_ylabel("Gas Used")
    axs[0].grid(True, alpha=0.3)
    
    # Plot transaction counts by type
    tx_types = list(blockchain_metrics["tx_by_type"].keys())
    tx_counts = list(blockchain_metrics["tx_by_type"].values())
    axs[1].bar(tx_types, tx_counts)
    axs[1].set_title("Transaction Count by Type")
    axs[1].set_xlabel("Transaction Type")
    axs[1].set_ylabel("Count")
    axs[1].grid(True, alpha=0.3)
    # Rotate x-axis labels if there are many types
    if len(tx_types) > 4:
        plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved blockchain metrics plot to {output_path}")
    
    return fig