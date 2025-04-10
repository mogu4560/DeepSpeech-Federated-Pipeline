import pytest
import numpy as np
import torch
from deepspeech.audio.feature_extractor import AcousticFeatureExtractor
from deepspeech.models.whisper_adapter import WhisperFederatedAdapter
from deepspeech.federated.fl_client import FederatedSpeechClient
from deepspeech.federated.fl_server import FederatedSpeechServer

def test_feature_extractor():
    extractor = AcousticFeatureExtractor()
    res = extractor.process_file("dummy_audio.wav")
    assert res["sample_rate"] == 16000
    assert res["spectrogram_shape"][0] == 80

def test_whisper_adapter_forward():
    model = WhisperFederatedAdapter()
    mel_input = torch.randn(2, 80, 50)
    logits = model(mel_input)
    assert logits.shape == (2, 50, model.vocab_size)

def test_federated_client_server_loop():
    client1 = FederatedSpeechClient(client_id=1, data_path="./data1")
    client2 = FederatedSpeechClient(client_id=2, data_path="./data2")
    server = FederatedSpeechServer(min_clients=2, num_rounds=1)

    w1, n1, meta1 = client1.fit(client1.get_parameters(), {})
    w2, n2, meta2 = client2.fit(client2.get_parameters(), {})

    client_updates = [(w1, n1, meta1["loss"]), (w2, n2, meta2["loss"])]
    round_res = server.run_federated_round(1, client_updates)

    assert round_res["participating_clients"] == 2
    assert round_res["weights_aggregated"] > 0
