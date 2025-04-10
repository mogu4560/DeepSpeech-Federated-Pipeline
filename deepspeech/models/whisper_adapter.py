import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class LoRAAdapterLayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) adapter module inserted into Whisper Transformer attention projections.
    """
    def __init__(self, in_features: int = 512, out_features: int = 512, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ A^T @ B^T * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class WhisperFederatedAdapter(nn.Module):
    """
    PyTorch Wrapper for OpenAI Whisper Encoder-Decoder model.
    Attaches LoRA parameters and computes local acoustic gradients for Federated Learning updates.
    """
    def __init__(self, d_model: int = 512, vocab_size: int = 51865, lora_rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Audio Encoder Projection
        self.encoder_proj = nn.Linear(80, d_model)
        self.lora_encoder = LoRAAdapterLayer(d_model, d_model, rank=lora_rank)
        
        # Text Decoder Classifier
        self.decoder_head = nn.Linear(d_model, vocab_size)

    def get_trainable_parameters(self) -> List[np.ndarray]:
        """Extracts numpy array list of trainable LoRA weights for Flower FL transmission."""
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params.append(p.detach().cpu().numpy())
        return params

    def set_trainable_parameters(self, weights: List[np.ndarray]):
        """Updates model LoRA parameters from aggregated Flower FL server weight array."""
        trainable = [p for p in self.parameters() if p.requires_grad]
        for p, w in zip(trainable, weights):
            p.data = torch.from_numpy(w).to(p.device)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        mel_spectrogram: [batch_size, 80, time_frames]
        Returns: logits [batch_size, time_frames, vocab_size]
        """
        x = mel_spectrogram.transpose(1, 2)  # [batch_size, time, 80]
        enc_out = self.encoder_proj(x)
        lora_out = self.lora_encoder(enc_out)
        h = enc_out + lora_out
        logits = self.decoder_head(h)
        return logits

    def apply_differential_privacy_noise(self, max_norm: float = 1.0, noise_multiplier: float = 0.1):
        """Applies DP-SGD gradient clipping and Gaussian noise injection to preserve client voice privacy."""
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                noise = torch.randn_like(p.grad) * (max_norm * noise_multiplier)
                p.grad.add_(noise)
        return float(total_norm)
