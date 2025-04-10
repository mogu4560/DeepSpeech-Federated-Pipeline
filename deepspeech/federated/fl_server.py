import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FederatedSpeechServer:
    """
    Central Federated Averaging (FedAvg) and FedProx server for aggregating
    decentralized speech client weights without inspecting client audio files.
    """
    def __init__(self, min_clients: int = 2, num_rounds: int = 5):
        self.min_clients = min_clients
        self.num_rounds = num_rounds
        self.global_weights: List[np.ndarray] = []
        self.round_history: List[Dict[str, Any]] = []

    def aggregate_fedavg(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Computes weighted average of client parameters: W_global = sum(n_i * W_i) / sum(n_i).
        """
        if not results:
            return []

        total_samples = sum(num_samples for _, num_samples in results)
        first_weights = results[0][0]
        aggregated_weights = [np.zeros_like(w) for w in first_weights]

        for client_weights, num_samples in results:
            weight_factor = num_samples / total_samples
            for idx, w in enumerate(client_weights):
                aggregated_weights[idx] += w * weight_factor

        return aggregated_weights

    def run_federated_round(self, round_num: int, client_updates: List[Tuple[List[np.ndarray], int, float]]) -> Dict[str, Any]:
        """Runs a single server aggregation round."""
        logger.info(f"--- Server Aggregation Round {round_num}/{self.num_rounds} ---")
        
        results_for_agg = [(weights, num_samples) for weights, num_samples, _ in client_updates]
        self.global_weights = self.aggregate_fedavg(results_for_agg)

        avg_client_loss = float(np.mean([loss for _, _, loss in client_updates]))
        
        round_metric = {
            "round": round_num,
            "participating_clients": len(client_updates),
            "avg_training_loss": round(avg_client_loss, 4),
            "weights_aggregated": len(self.global_weights)
        }
        self.round_history.append(round_metric)
        return round_metric
