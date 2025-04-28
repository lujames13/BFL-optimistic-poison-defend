"""
Defense effectiveness evaluation for Byzantine-robust federated learning.

This module implements tools to evaluate the effectiveness of the Krum defense
mechanism against various attack models in a blockchain-based federated learning system.
"""

import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from ..blockchain_connector import BlockchainConnector
from ..ipfs_connector import ModelIPFSConnector


class DefenseEvaluator:
    """Evaluator for defense mechanisms in federated learning."""
    
    def __init__(
        self,
        blockchain_connector: BlockchainConnector,
        ipfs_connector: ModelIPFSConnector,
        config: Dict[str, Any]
    ):
        """Initialize the defense evaluator.
        
        Args:
            blockchain_connector: Connector to the blockchain network
            ipfs_connector: Connector to IPFS
            config: Evaluation configuration
        """
        self.blockchain_connector = blockchain_connector
        self.ipfs_connector = ipfs_connector
        self.config = config
        
        # Extract configuration
        self.results_dir = config.get("results_dir", "results")
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Initialized defense evaluator with results directory: {self.results_dir}")
    
    def get_task_history(self, task_id: int) -> Dict[str, Any]:
        """Get the history of a federated learning task.
        
        Args:
            task_id: Task ID to evaluate
            
        Returns:
            Dictionary with task history
        """
        try:
            print(f"Getting history for task {task_id}")
            
            # Get task information
            task_info = self.blockchain_connector.getTaskInfo(task_id)
            
            # Get rounds information
            rounds = []
            for round_id in range(1, task_info.get("completedRounds", 0) + 1):
                round_info = self.blockchain_connector.get_round_info(round_id)
                rounds.append(round_info)
            
            # Return combined history
            return {
                "task": task_info,
                "rounds": rounds
            }
        except Exception as e:
            print(f"Error getting task history: {str(e)}")
            return {"error": str(e)}
    
    def download_model_for_evaluation(self, model_hash: str, model_class: torch.nn.Module) -> torch.nn.Module:
        """Download a model from IPFS and prepare it for evaluation.
        
        Args:
            model_hash: IPFS hash of the model
            model_class: PyTorch model class
            
        Returns:
            Loaded PyTorch model
        """
        try:
            # Download model parameters
            parameters = self.ipfs_connector.download_model(model_hash)
            
            # Create model instance
            model = model_class()
            
            # Load parameters
            params_dict = zip(model.parameters(), parameters)
            with torch.no_grad():
                for model_param, param in params_dict:
                    model_param.copy_(torch.tensor(param))
            
            return model
        except Exception as e:
            print(f"Error downloading model for evaluation: {str(e)}")
            return None
    
    def compare_models(self, models: List[Tuple[str, torch.nn.Module]], test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Compare multiple models on test data.
        
        Args:
            models: List of (name, model) tuples
            test_data: Test data as (inputs, targets) tuple
            
        Returns:
            Dictionary with comparison results
        """
        x_test, y_test = test_data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        results = {}
        
        for name, model in models:
            model.to(device)
            model.eval()
            
            # Evaluate model
            with torch.no_grad():
                inputs = x_test.to(device)
                targets = y_test.to(device)
                
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                total = targets.size(0)
                
                accuracy = correct / total
                
                results[name] = {
                    "loss": float(loss.item()),
                    "accuracy": float(accuracy)
                }
        
        return results
    
    def evaluate_krum_defense(self, task_id: int, round_id: int, model_class: torch.nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Evaluate Krum defense effectiveness for a specific round.
        
        Args:
            task_id: Task ID to evaluate
            round_id: Round ID to evaluate
            model_class: PyTorch model class
            test_data: Test data as (inputs, targets) tuple
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get round information
            round_info = self.blockchain_connector.get_round_info(round_id)
            
            # Get participating clients
            clients = []
            for client_id in range(1, self.blockchain_connector.get_system_status()["totalClients"] + 1):
                if self.blockchain_connector.didClientParticipate(client_id, round_id):
                    clients.append(client_id)
            
            # Download client models
            client_models = []
            for client_id in clients:
                model_hash = self.blockchain_connector.getModelUpdateHash(client_id, round_id)
                model = self.download_model_for_evaluation(model_hash, model_class)
                client_models.append((f"Client {client_id}", model))
            
            # Download selected model by Krum
            global_model_hash = round_info.get("globalModelHash")
            global_model = self.download_model_for_evaluation(global_model_hash, model_class)
            
            client_models.append(("Global (Krum selected)", global_model))
            
            # Compare models
            comparison_results = self.compare_models(client_models, test_data)
            
            # Get the selected client ID from blockchain
            selected_client_id = None
            for client_id in clients:
                update_info = self.blockchain_connector.get_round_info(round_id).get("updates", {}).get(str(client_id), {})
                if update_info.get("accepted", False):
                    selected_client_id = client_id
                    break
            
            # Prepare results
            result = {
                "task_id": task_id,
                "round_id": round_id,
                "global_model_hash": global_model_hash,
                "selected_client_id": selected_client_id,
                "client_models": comparison_results,
                "timestamp": time.time()
            }
            
            return result
        except Exception as e:
            print(f"Error evaluating Krum defense: {str(e)}")
            return {"error": str(e)}
    
    def generate_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """Generate a report from evaluation results.
        
        Args:
            results: Evaluation results
            filename: Output filename (optional)
            
        Returns:
            Path to the generated report
        """
        if filename is None:
            filename = f"defense_report_{int(time.time())}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Save results as JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Report saved to {filepath}")
        return filepath
    
    def plot_accuracy_comparison(self, results: Dict[str, Any], filename: str = None) -> str:
        """Plot accuracy comparison from evaluation results.
        
        Args:
            results: Evaluation results
            filename: Output filename (optional)
            
        Returns:
            Path to the generated plot
        """
        if filename is None:
            filename = f"accuracy_comparison_{int(time.time())}.png"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Extract client model accuracies
        client_models = results.get("client_models", {})
        
        names = []
        accuracies = []
        
        for name, metrics in client_models.items():
            names.append(name)
            accuracies.append(metrics.get("accuracy", 0.0))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, accuracies)
        
        # Highlight the selected model
        selected_client_id = results.get("selected_client_id")
        if selected_client_id is not None:
            selected_idx = None
            for i, name in enumerate(names):
                if name == f"Client {selected_client_id}" or name == "Global (Krum selected)":
                    selected_idx = i
                    break
            
            if selected_idx is not None:
                bars[selected_idx].set_color('green')
        
        plt.title(f"Model Accuracy Comparison - Round {results.get('round_id')}")
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(filepath)
        print(f"Accuracy comparison plot saved to {filepath}")
        return filepath
    
    def compare_with_without_defense(self, task_with_defense: int, task_without_defense: int, round_id: int, model_class: torch.nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Compare performance with and without defense.
        
        Args:
            task_with_defense: Task ID with defense
            task_without_defense: Task ID without defense
            round_id: Round ID to compare
            model_class: PyTorch model class
            test_data: Test data as (inputs, targets) tuple
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Get models with defense
            with_defense_info = self.blockchain_connector.get_round_info(round_id)
            with_defense_hash = with_defense_info.get("globalModelHash")
            with_defense_model = self.download_model_for_evaluation(with_defense_hash, model_class)
            
            # Get models without defense (assuming same round ID in both tasks)
            without_defense_info = self.blockchain_connector.get_round_info(round_id)
            without_defense_hash = without_defense_info.get("globalModelHash")
            without_defense_model = self.download_model_for_evaluation(without_defense_hash, model_class)
            
            # Compare models
            models = [
                ("With Defense (Krum)", with_defense_model),
                ("Without Defense", without_defense_model)
            ]
            
            comparison_results = self.compare_models(models, test_data)
            
            # Calculate improvement
            with_acc = comparison_results["With Defense (Krum)"]["accuracy"]
            without_acc = comparison_results["Without Defense"]["accuracy"]
            
            improvement = (with_acc - without_acc) / without_acc * 100 if without_acc > 0 else float('inf')
            
            # Prepare results
            result = {
                "task_with_defense": task_with_defense,
                "task_without_defense": task_without_defense,
                "round_id": round_id,
                "with_defense_hash": with_defense_hash,
                "without_defense_hash": without_defense_hash,
                "comparison": comparison_results,
                "improvement_percentage": float(improvement),
                "timestamp": time.time()
            }
            
            return result
        except Exception as e:
            print(f"Error comparing with/without defense: {str(e)}")
            return {"error": str(e)}
    
    def evaluate_attack_impact(self, task_id: int, attacker_ids: List[int], honest_ids: List[int], model_class: torch.nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Evaluate impact of attackers on federated learning.
        
        Args:
            task_id: Task ID to evaluate
            attacker_ids: List of attacker client IDs
            honest_ids: List of honest client IDs
            model_class: PyTorch model class
            test_data: Test data as (inputs, targets) tuple
            
        Returns:
            Dictionary with evaluation results
        """
        task_info = self.blockchain_connector.getTaskInfo(task_id)
        total_rounds = task_info.get("completedRounds", 0)
        
        # Track metrics across rounds
        global_accuracies = []
        attacker_accuracies = []
        honest_accuracies = []
        selected_clients = []
        
        # Evaluate each round
        for round_id in range(1, total_rounds + 1):
            # Get round information
            round_info = self.blockchain_connector.get_round_info(round_id)
            global_model_hash = round_info.get("globalModelHash")
            
            # Check which clients were selected
            selected_client_id = None
            for client_id in attacker_ids + honest_ids:
                update_info = self.blockchain_connector.get_round_info(round_id).get("updates", {}).get(str(client_id), {})
                if update_info.get("accepted", False):
                    selected_client_id = client_id
                    break
            
            selected_clients.append(selected_client_id)
            
            # Load models
            models = []
            
            # Global model
            global_model = self.download_model_for_evaluation(global_model_hash, model_class)
            models.append(("Global", global_model))
            
            # Attacker models
            attacker_models = []
            for client_id in attacker_ids:
                if self.blockchain_connector.didClientParticipate(client_id, round_id):
                    model_hash = self.blockchain_connector.getModelUpdateHash(client_id, round_id)
                    model = self.download_model_for_evaluation(model_hash, model_class)
                    models.append((f"Attacker {client_id}", model))
                    attacker_models.append(model)
            
            # Honest models
            honest_models = []
            for client_id in honest_ids:
                if self.blockchain_connector.didClientParticipate(client_id, round_id):
                    model_hash = self.blockchain_connector.getModelUpdateHash(client_id, round_id)
                    model = self.download_model_for_evaluation(model_hash, model_class)
                    models.append((f"Honest {client_id}", model))
                    honest_models.append(model)
            
            # Compare all models
            comparison = self.compare_models(models, test_data)
            
            # Track accuracies
            global_accuracies.append(comparison["Global"]["accuracy"])
            
            # Average attacker and honest accuracies
            if attacker_models:
                avg_attacker_acc = sum(comparison[f"Attacker {client_id}"]["accuracy"] for client_id in attacker_ids if f"Attacker {client_id}" in comparison) / len(attacker_models)
                attacker_accuracies.append(avg_attacker_acc)
            else:
                attacker_accuracies.append(None)
            
            if honest_models:
                avg_honest_acc = sum(comparison[f"Honest {client_id}"]["accuracy"] for client_id in honest_ids if f"Honest {client_id}" in comparison) / len(honest_models)
                honest_accuracies.append(avg_honest_acc)
            else:
                honest_accuracies.append(None)
        
        # Calculate defense effectiveness
        attack_success_rate = sum(1 for client_id in selected_clients if client_id in attacker_ids) / total_rounds if total_rounds > 0 else 0
        defense_effectiveness = 1 - attack_success_rate
        
        result = {
            "task_id": task_id,
            "total_rounds": total_rounds,
            "attacker_ids": attacker_ids,
            "honest_ids": honest_ids,
            "selected_clients": selected_clients,
            "global_accuracies": global_accuracies,
            "attacker_accuracies": attacker_accuracies,
            "honest_accuracies": honest_accuracies,
            "attack_success_rate": float(attack_success_rate),
            "defense_effectiveness": float(defense_effectiveness),
            "timestamp": time.time()
        }
        
        return result
    
    def plot_accuracies_over_rounds(self, results: Dict[str, Any], filename: str = None) -> str:
        """Plot accuracies over rounds from evaluation results.
        
        Args:
            results: Evaluation results
            filename: Output filename (optional)
            
        Returns:
            Path to the generated plot
        """
        if filename is None:
            filename = f"accuracies_over_rounds_{int(time.time())}.png"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Extract data
        rounds = range(1, results.get("total_rounds", 0) + 1)
        global_accuracies = results.get("global_accuracies", [])
        attacker_accuracies = results.get("attacker_accuracies", [])
        honest_accuracies = results.get("honest_accuracies", [])
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(rounds, global_accuracies, 'b-', marker='o', label='Global Model')
        
        # Filter out None values for attacker and honest accuracies
        attacker_rounds = [r for r, acc in zip(rounds, attacker_accuracies) if acc is not None]
        attacker_accs = [acc for acc in attacker_accuracies if acc is not None]
        if attacker_accs:
            plt.plot(attacker_rounds, attacker_accs, 'r--', marker='x', label='Attackers (Avg)')
        
        honest_rounds = [r for r, acc in zip(rounds, honest_accuracies) if acc is not None]
        honest_accs = [acc for acc in honest_accuracies if acc is not None]
        if honest_accs:
            plt.plot(honest_rounds, honest_accs, 'g-.', marker='+', label='Honest Clients (Avg)')
        
        # Mark which clients were selected
        selected_clients = results.get("selected_clients", [])
        attacker_ids = results.get("attacker_ids", [])
        honest_ids = results.get("honest_ids", [])
        
        for i, client_id in enumerate(selected_clients):
            if client_id in attacker_ids:
                plt.plot(i + 1, global_accuracies[i], 'ro', markersize=10, alpha=0.5)
            elif client_id in honest_ids:
                plt.plot(i + 1, global_accuracies[i], 'go', markersize=10, alpha=0.5)
        
        plt.title(f"Model Accuracy Over Rounds (Task {results.get('task_id')})")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add defense effectiveness information
        defense_effectiveness = results.get("defense_effectiveness", 0) * 100
        plt.figtext(0.5, 0.01, f"Defense Effectiveness: {defense_effectiveness:.2f}%", 
                   ha='center', fontsize=12, bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})
        
        plt.tight_layout()
        plt.savefig(filepath)
        print(f"Accuracies over rounds plot saved to {filepath}")
        return filepath


def main():
    """Main function to run the defense evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Defense Evaluator")
    parser.add_argument("--contract_address", type=str, required=True, help="Blockchain contract address")
    parser.add_argument("--node_url", type=str, default="http://127.0.0.1:8545", help="Blockchain node URL")
    parser.add_argument("--ipfs_url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--task_id", type=int, required=True, help="Task ID to evaluate")
    parser.add_argument("--mode", type=str, choices=["krum", "comparison", "attack"], default="krum",
                      help="Evaluation mode: krum, comparison, or attack")
    parser.add_argument("--round_id", type=int, help="Round ID to evaluate (for krum and comparison modes)")
    parser.add_argument("--task_without_defense", type=int, help="Task ID without defense (for comparison mode)")
    parser.add_argument("--attacker_ids", type=str, help="Comma-separated list of attacker client IDs (for attack mode)")
    parser.add_argument("--honest_ids", type=str, help="Comma-separated list of honest client IDs (for attack mode)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "results_dir": args.results_dir
    }
    
    # Initialize blockchain connector
    blockchain_connector = BlockchainConnector(
        contract_address=args.contract_address,
        node_url=args.node_url
    )
    
    # Initialize IPFS connector
    ipfs_connector = ModelIPFSConnector(api_url=args.ipfs_url)
    
    # Create evaluator
    evaluator = DefenseEvaluator(
        blockchain_connector=blockchain_connector,
        ipfs_connector=ipfs_connector,
        config=config
    )
    
    # Create a simple model class
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(20, 10)
            self.softmax = torch.nn.Softmax(dim=1)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x
    
    # Load test data
    print(f"Loading test data from {args.data_path}")
    try:
        # Try to load actual data
        data = torch.load(args.data_path)
        x_test, y_test = data["test"]
    except:
        print("Could not load data, using random data for demonstration")
        # Create random data for demonstration
        x_test = torch.randn(20, 10)
        y_test = torch.randint(0, 10, (20,))
    
    # Perform evaluation based on mode
    if args.mode == "krum":
        if args.round_id is None:
            print("Round ID is required for Krum evaluation")
            return
        
        print(f"Evaluating Krum defense for task {args.task_id}, round {args.round_id}")
        results = evaluator.evaluate_krum_defense(
            args.task_id, args.round_id, SimpleModel, (x_test, y_test)
        )
        
        # Generate report
        report_path = evaluator.generate_report(results, f"krum_evaluation_task{args.task_id}_round{args.round_id}.json")
        
        # Generate plot
        plot_path = evaluator.plot_accuracy_comparison(results, f"krum_accuracy_task{args.task_id}_round{args.round_id}.png")
        
    elif args.mode == "comparison":
        if args.round_id is None or args.task_without_defense is None:
            print("Round ID and task_without_defense are required for comparison mode")
            return
        
        print(f"Comparing with/without defense for tasks {args.task_id}/{args.task_without_defense}, round {args.round_id}")
        results = evaluator.compare_with_without_defense(
            args.task_id, args.task_without_defense, args.round_id, SimpleModel, (x_test, y_test)
        )
        
        # Generate report
        report_path = evaluator.generate_report(
            results, f"comparison_task{args.task_id}_vs_task{args.task_without_defense}_round{args.round_id}.json"
        )
        
    elif args.mode == "attack":
        if args.attacker_ids is None or args.honest_ids is None:
            print("Attacker IDs and honest IDs are required for attack mode")
            return
        
        # Parse client IDs
        attacker_ids = [int(id.strip()) for id in args.attacker_ids.split(',')]
        honest_ids = [int(id.strip()) for id in args.honest_ids.split(',')]
        
        print(f"Evaluating attack impact for task {args.task_id}")
        print(f"Attackers: {attacker_ids}")
        print(f"Honest clients: {honest_ids}")
        
        results = evaluator.evaluate_attack_impact(
            args.task_id, attacker_ids, honest_ids, SimpleModel, (x_test, y_test)
        )
        
        # Generate report
        report_path = evaluator.generate_report(results, f"attack_impact_task{args.task_id}.json")
        
        # Generate plot
        plot_path = evaluator.plot_accuracies_over_rounds(results, f"attack_impact_task{args.task_id}.png")
    
    print("Evaluation completed successfully")


if __name__ == "__main__":
    main()
                