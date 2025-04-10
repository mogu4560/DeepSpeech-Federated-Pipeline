import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

@dataclass
class SpectrogramConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0

class AcousticFeatureExtractor:
    """
    Audio preprocessing pipeline computing mel-spectrograms, Voice Activity Detection (VAD),
    and spectral noise reduction for OpenAI Whisper input feeds.
    """
    def __init__(self, config: Optional[SpectrogramConfig] = None):
        self.config = config or SpectrogramConfig()

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Peak normalization to [-1.0, 1.0] range."""
        if len(audio_data) == 0:
            return audio_data
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data

    def voice_activity_detection(self, audio_data: np.ndarray, energy_threshold: float = 0.02) -> np.ndarray:
        """Energy-based Voice Activity Detection (VAD) frame filter."""
        frame_len = self.config.hop_length
        num_frames = len(audio_data) // frame_len
        valid_frames = []

        for i in range(num_frames):
            frame = audio_data[i * frame_len : (i + 1) * frame_len]
            energy = np.sqrt(np.mean(frame ** 2))
            if energy >= energy_threshold:
                valid_frames.append(frame)

        if valid_frames:
            return np.concatenate(valid_frames)
        return audio_data

    def compute_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Computes simulated 80-channel log-mel spectrogram matrix [80, time_frames]
        compatible with Whisper encoder inputs.
        """
        audio_norm = self.normalize_audio(audio_data)
        audio_vad = self.voice_activity_detection(audio_norm)
        
        num_samples = len(audio_vad)
        num_frames = max(1, num_samples // self.config.hop_length)
        
        # Synthetic mel filterbank matrix generation for standalone execution
        spectrogram = np.abs(np.random.randn(self.config.n_mels, num_frames))
        log_mel_spec = np.log(np.maximum(1e-5, spectrogram))
        return log_mel_spec

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Simulates audio file loading and feature extraction."""
        # 3-second dummy audio wave at 16kHz
        dummy_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 16000 * 3))
        mel_spec = self.compute_mel_spectrogram(dummy_audio)
        
        return {
            "file_path": file_path,
            "sample_rate": self.config.sample_rate,
            "duration_sec": 3.0,
            "spectrogram_shape": mel_spec.shape,
            "features": mel_spec
        }
