import numpy as np
from typing import List, Tuple, Dict, Any

class DifferentialPrivacyAggregator:
    """
    Applies Gaussian DP mechanisms and clipping to server-side model aggregation.
    """
    def __init__(self, clip_norm: float = 1.0, noise_scale: float = 0.01):
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale

    def clip_weight_updates(self, updates: List[np.ndarray]) -> List[np.ndarray]:
        clipped = []
        for u in updates:
            norm = np.linalg.norm(u)
            if norm > self.clip_norm:
                u = u * (self.clip_norm / norm)
            clipped.append(u)
        return clipped

    def add_gaussian_noise(self, weight: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.noise_scale, size=weight.shape)
        return weight + noise

    def aggregate_with_privacy(self, client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        if not client_weights:
            return []
            
        num_clients = len(client_weights)
        first_layer_count = len(client_weights[0])
        aggregated = []

        for layer_idx in range(first_layer_count):
            layer_updates = [cw[layer_idx] for cw in client_weights]
            clipped_updates = self.clip_weight_updates(layer_updates)
            avg_layer = np.mean(clipped_updates, axis=0)
            noisy_layer = self.add_gaussian_noise(avg_layer)
            aggregated.append(noisy_layer)

        return aggregated
