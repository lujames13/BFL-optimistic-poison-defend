"""
Orchestrator module for blockchain-based federated learning.

This module integrates all components of the BFL-Optimistic-Poison-Defend system,
providing a high-level interface for running federated learning tasks with
blockchain verification and Krum defense.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import argparse
import torch
import numpy as np
import concurrent.futures
import subprocess

from .blockchain_connector import BlockchainConnector
from .ipfs_connector import ModelIPFSConnector
from .server.server import FlowerServer
from .evaluation.defense_effectiveness import DefenseEvaluator


class FLOrchestrator:
    """Orchestrator for blockchain-based federated learning."""
    
    def __init__(
        self,
        config_path: str,
        blockchain_config: Dict[str, Any],
        ipfs_config: Dict[str, Any]
    ):
        """Initialize the orchestrator.
        
        Args:
            config_path: Path to main configuration file
            blockchain_config: Blockchain configuration
            ipfs_config: IPFS configuration
        """
        self.config_path = config_path
        self.blockchain_config = blockchain_config
        self.ipfs_config = ipfs_config
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize connectors
        self.blockchain_connector = BlockchainConnector(
            contract_address=blockchain_config["contract_address"],
            node_url=blockchain_config["node_url"]
        )
        
        self.ipfs_connector = ModelIPFSConnector(
            api_url=ipfs_config["api_url"]
        )
        
        # Create output directories
        self.output_dir = self.config.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(self.output_dir, "orchestrator.log")
        self.log(f"Initialized FL Orchestrator with config from {config_path}")
    
    def log(self, message: str):
        """Log a message to the log file.
        
        Args:
            message: Message to log
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def create_model(self) -> torch.nn.Module:
        """Create model based on configuration.
        
        Returns:
            PyTorch model
        """
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "simple")
        
        if model_type == "simple":
            # Create a simple model
            input_dim = model_config.get("input_dim", 10)
            hidden_dim = model_config.get("hidden_dim", 20)
            output_dim = model_config.get("output_dim", 10)
            
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim),
                torch.nn.Softmax(dim=1)
            )
        elif model_type == "cnn":
            # Create a CNN model
            channels = model_config.get("channels", 1)
            output_dim = model_config.get("output_dim", 10)
            
            model = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 7 * 7, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, output_dim)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.log(f"Created {model_type} model")
        return model
    
    def start_server(self) -> Dict[str, Any]:
        """Start the federated learning server.
        
        Returns:
            Dictionary with server launch information
        """
        server_config = self.config.get("server", {})
        
        # Create output directories
        server_dir = os.path.join(self.output_dir, "server")
        os.makedirs(server_dir, exist_ok=True)
        
        # Create server config file
        server_config_path = os.path.join(server_dir, "server_config.json")
        with open(server_config_path, 'w') as f:
            json.dump(server_config, f, indent=2)
        
        # Create model
        model = self.create_model()
        model_path = os.path.join(server_dir, "initial_model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Prepare server command
        script_path = os.path.join(os.path.dirname(__file__), "server", "server.py")
        cmd = [
            "python", script_path,
            "--task_params", server_config_path,
            "--contract_address", self.blockchain_config["contract_address"],
            "--node_url", self.blockchain_config["node_url"],
            "--ipfs_url", self.ipfs_config["api_url"],
            "--model_path", model_path
        ]
        
        # Start server process
        self.log(f"Starting server with command: {' '.join(cmd)}")
        server_log_path = os.path.join(server_dir, "server.log")
        
        with open(server_log_path, 'w') as log_file:
            server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        self.log(f"Server started with PID {server_process.pid}")
        
        return {
            "process": server_process,
            "pid": server_process.pid,
            "log_path": server_log_path,
            "config_path": server_config_path
        }
    
    def start_clients(self, num_clients: int, num_attackers: int = 0, attacker_type: str = "label_flipping") -> Dict[str, Any]:
        """Start federated learning clients.
        
        Args:
            num_clients: Number of honest clients to start
            num_attackers: Number of malicious clients to start
            attacker_type: Type of attack for malicious clients
            
        Returns:
            Dictionary with client launch information
        """
        client_config = self.config.get("client", {})
        
        # Create output directories
        clients_dir = os.path.join(self.output_dir, "clients")
        os.makedirs(clients_dir, exist_ok=True)
        
        # Prepare for launching clients
        client_processes = []
        client_info = []
        
        # Launch honest clients
        for i in range(1, num_clients + 1):
            client_id = i
            client_dir = os.path.join(clients_dir, f"client_{client_id}")
            os.makedirs(client_dir, exist_ok=True)
            
            # Create client config file
            client_config_path = os.path.join(client_dir, "client_config.json")
            client_specific_config = client_config.copy()
            client_specific_config["client_id"] = client_id
            
            with open(client_config_path, 'w') as f:
                json.dump(client_specific_config, f, indent=2)
            
            # Create test data (in real implementation this would be actual data)
            data_path = os.path.join(client_dir, "data.pt")
            x_train = torch.randn(100, 10)
            y_train = torch.randint(0, 10, (100,))
            x_test = torch.randn(20, 10)
            y_test = torch.randint(0, 10, (20,))
            torch.save({"train": (x_train, y_train), "test": (x_test, y_test)}, data_path)
            
            # Prepare client command
            script_path = os.path.join(os.path.dirname(__file__), "client", "client.py")
            cmd = [
                "python", script_path,
                "--client_id", str(client_id),
                "--contract_address", self.blockchain_config["contract_address"],
                "--node_url", self.blockchain_config["node_url"],
                "--ipfs_url", self.ipfs_config["api_url"],
                "--data_path", data_path,
                "--config_path", client_config_path
            ]
            
            # Start client process
            self.log(f"Starting honest client {client_id} with command: {' '.join(cmd)}")
            client_log_path = os.path.join(client_dir, "client.log")
            
            with open(client_log_path, 'w') as log_file:
                client_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            client_processes.append(client_process)
            client_info.append({
                "id": client_id,
                "pid": client_process.pid,
                "log_path": client_log_path,
                "config_path": client_config_path,
                "type": "honest"
            })
            
            self.log(f"Honest client {client_id} started with PID {client_process.pid}")
        
        # Launch attacker clients
        for i in range(1, num_attackers + 1):
            client_id = num_clients + i
            client_dir = os.path.join(clients_dir, f"attacker_{client_id}")
            os.makedirs(client_dir, exist_ok=True)
            
            # Create attacker data
            data_path = os.path.join(client_dir, "data.pt")
            x_train = torch.randn(100, 10)
            y_train = torch.randint(0, 10, (100,))
            x_test = torch.randn(20, 10)
            y_test = torch.randint(0, 10, (20,))
            torch.save({"train": (x_train, y_train), "test": (x_test, y_test)}, data_path)
            
            # Prepare attacker command
            script_path = os.path.join(os.path.dirname(__file__), "attack", "attack_simulator.py")
            cmd = [
                "python", script_path,
                "--client_id", str(client_id),
                "--contract_address", self.blockchain_config["contract_address"],
                "--node_url", self.blockchain_config["node_url"],
                "--ipfs_url", self.ipfs_config["api_url"],
                "--data_path", data_path,
                "--attack_type", attacker_type,
                "--intensity", "0.8"  # High intensity for demonstration
            ]
            
            # Start attacker process
            self.log(f"Starting {attacker_type} attacker {client_id} with command: {' '.join(cmd)}")
            client_log_path = os.path.join(client_dir, "attacker.log")
            
            with open(client_log_path, 'w') as log_file:
                client_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            client_processes.append(client_process)
            client_info.append({
                "id": client_id,
                "pid": client_process.pid,
                "log_path": client_log_path,
                "type": "attacker",
                "attack_type": attacker_type
            })
            
            self.log(f"Attacker client {client_id} started with PID {client_process.pid}")
        
        return {
            "processes": client_processes,
            "clients": client_info,
            "num_honest": num_clients,
            "num_attackers": num_attackers
        }
    
    def monitor_training(self, task_id: int, timeout: int = 3600) -> Dict[str, Any]:
        """Monitor federated learning training progress.
        
        Args:
            task_id: Task ID to monitor
            timeout: Maximum time to wait for completion in seconds
            
        Returns:
            Dictionary with monitoring results
        """
        start_time = time.time()
        completed = False
        
        self.log(f"Starting to monitor task {task_id}")
        
        while not completed and (time.time() - start_time) < timeout:
            try:
                # Get task status
                task_info = self.blockchain_connector.getTaskInfo(task_id)
                status = task_info.get("status", 0)
                
                if status == 2:  # TaskStatus.Completed
                    self.log(f"Task {task_id} completed successfully")
                    completed = True
                    break
                elif status == 3:  # TaskStatus.Terminated
                    self.log(f"Task {task_id} was terminated")
                    break
                
                # Get system status
                system_status = self.blockchain_connector.get_system_status()
                current_round = system_status.get("currentRound", 0)
                round_status = system_status.get("currentRoundStatus", 0)
                
                self.log(f"Task {task_id} - Current round: {current_round}, Status: {round_status}")
                
                # Wait before checking again
                time.sleep(10)
                
            except Exception as e:
                self.log(f"Error monitoring task: {str(e)}")
                time.sleep(30)  # Wait longer after an error
        
        elapsed_time = time.time() - start_time
        
        if not completed and elapsed_time >= timeout:
            self.log(f"Monitoring timed out after {timeout} seconds")
        
        return {
            "task_id": task_id,
            "completed": completed,
            "elapsed_time": elapsed_time
        }
    
    def evaluate_results(self, task_id: int, honest_ids: List[int], attacker_ids: List[int]) -> Dict[str, Any]:
        """Evaluate training results and defense effectiveness.
        
        Args:
            task_id: Task ID to evaluate
            honest_ids: List of honest client IDs
            attacker_ids: List of attacker client IDs
            
        Returns:
            Dictionary with evaluation results
        """
        # Create output directory
        eval_dir = os.path.join(self.output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Create evaluator
        evaluator = DefenseEvaluator(
            blockchain_connector=self.blockchain_connector,
            ipfs_connector=self.ipfs_connector,
            config={"results_dir": eval_dir}
        )
        
        self.log(f"Evaluating task {task_id} with {len(honest_ids)} honest clients and {len(attacker_ids)} attackers")
        
        # Create a simple model for evaluation
        model_class = self.create_model().__class__
        
        # Create test data for evaluation
        x_test = torch.randn(100, 10)
        y_test = torch.randint(0, 10, (100,))
        test_data = (x_test, y_test)
        
        # Evaluate attack impact
        attack_results = evaluator.evaluate_attack_impact(
            task_id, attacker_ids, honest_ids, model_class, test_data
        )
        
        # Generate report and plot
        report_path = evaluator.generate_report(
            attack_results, f"attack_impact_task{task_id}.json"
        )
        
        plot_path = evaluator.plot_accuracies_over_rounds(
            attack_results, f"attack_impact_task{task_id}.png"
        )
        
        self.log(f"Evaluation complete. Report saved to {report_path}")
        self.log(f"Plot saved to {plot_path}")
        
        # Extract key metrics
        defense_effectiveness = attack_results.get("defense_effectiveness", 0) * 100
        attack_success_rate = attack_results.get("attack_success_rate", 0) * 100
        
        self.log(f"Defense effectiveness: {defense_effectiveness:.2f}%")
        self.log(f"Attack success rate: {attack_success_rate:.2f}%")
        
        return {
            "task_id": task_id,
            "report_path": report_path,
            "plot_path": plot_path,
            "defense_effectiveness": defense_effectiveness,
            "attack_success_rate": attack_success_rate,
            "results": attack_results
        }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run a complete federated learning experiment.
        
        Returns:
            Dictionary with experiment results
        """
        experiment_config = self.config.get("experiment", {})
        
        # Extract experiment parameters
        num_honest_clients = experiment_config.get("num_honest_clients", 5)
        num_attackers = experiment_config.get("num_attackers", 2)
        attacker_type = experiment_config.get("attacker_type", "label_flipping")
        timeout = experiment_config.get("timeout", 3600)
        
        self.log(f"Starting experiment with {num_honest_clients} honest clients and {num_attackers} {attacker_type} attackers")
        
        try:
            # Start server
            server_info = self.start_server()
            
            # Wait for server to initialize
            time.sleep(10)
            
            # Get system status to check if server started a task
            system_status = self.blockchain_connector.get_system_status()
            task_id = self.blockchain_connector.currentTaskId
            
            if not task_id:
                self.log("Error: Server did not start a task")
                return {"success": False, "error": "Server did not start a task"}
            
            self.log(f"Server started task with ID: {task_id}")
            
            # Start clients
            client_info = self.start_clients(num_honest_clients, num_attackers, attacker_type)
            
            # Monitor training
            monitoring_results = self.monitor_training(task_id, timeout)
            
            # Get client IDs
            honest_ids = [client["id"] for client in client_info["clients"] if client["type"] == "honest"]
            attacker_ids = [client["id"] for client in client_info["clients"] if client["type"] == "attacker"]
            
            # Evaluate results
            evaluation_results = self.evaluate_results(task_id, honest_ids, attacker_ids)
            
            # Clean up processes
            self.log("Cleaning up processes")
            for process in client_info["processes"]:
                if process.poll() is None:  # If process is still running
                    process.terminate()
            
            if server_info["process"].poll() is None:
                server_info["process"].terminate()
            
            # Compile results
            experiment_results = {
                "success": True,
                "task_id": task_id,
                "server_info": server_info,
                "client_info": client_info,
                "monitoring_results": monitoring_results,
                "evaluation_results": evaluation_results,
                "timestamp": time.time()
            }
            
            # Save experiment results
            results_path = os.path.join(self.output_dir, "experiment_results.json")
            with open(results_path, 'w') as f:
                # Convert non-serializable items
                serializable_results = experiment_results.copy()
                serializable_results["server_info"] = {k: v for k, v in server_info.items() if k != "process"}
                serializable_results["client_info"]["clients"] = client_info["clients"]
                serializable_results["client_info"].pop("processes", None)
                
                json.dump(serializable_results, f, indent=2)
            
            self.log(f"Experiment completed successfully. Results saved to {results_path}")
            
            return experiment_results
            
        except Exception as e:
            self.log(f"Error during experiment: {str(e)}")
            return {"success": False, "error": str(e)}


def main():
    """Main function to run the orchestrator."""
    parser = argparse.ArgumentParser(description="Federated Learning Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--contract_address", type=str, required=True, help="Blockchain contract address")
    parser.add_argument("--node_url", type=str, default="http://127.0.0.1:8545", help="Blockchain node URL")
    parser.add_argument("--ipfs_url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    
    args = parser.parse_args()
    
    # Setup blockchain and IPFS configuration
    blockchain_config = {
        "contract_address": args.contract_address,
        "node_url": args.node_url
    }
    
    ipfs_config = {
        "api_url": args.ipfs_url
    }
    
    # Create orchestrator
    orchestrator = FLOrchestrator(
        config_path=args.config,
        blockchain_config=blockchain_config,
        ipfs_config=ipfs_config
    )
    
    # Run experiment
    results = orchestrator.run_experiment()
    
    if results["success"]:
        print("Experiment completed successfully!")
    else:
        print(f"Experiment failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()