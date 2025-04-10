# DeepSpeech-Federated: Secure Acoustic Transcription Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Flower](https://img.shields.io/badge/Flower-Federated_Learning-brightgreen.svg)](https://flower.dev)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-black.svg)](https://github.com/openai/whisper)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://docker.com)

DeepSpeech-Federated is a privacy-preserving federated speech-to-text learning framework powered by Flower and OpenAI Whisper. It enables edge nodes to train acoustic models locally on private voice recordings without ever centralizing sensitive raw audio logs, securely aggregating model weights across nodes using Federated Averaging (FedAvg).

---

## Key Features

- **Privacy-Preserving ASR:** Raw audio and transcriptions never leave the local client environment.
- **Whisper Fine-Tuning:** Fine-tunes OpenAI Whisper encoder-decoder layers locally using LoRA / adapter weights.
- **Secure Aggregation:** Server aggregates parameter updates across decentralized clients via Flower's `FedAvg` protocol.
- **Dockerized Deployments:** Containerized server and client nodes for scalable Kubernetes / Docker Compose deployment.

---

## Quick Start

### Installation & Docker Build

```bash
git clone https://github.com//DeepSpeech-Federated.git
cd DeepSpeech-Federated
pip install -r requirements.txt

# Build Docker container for client/server nodes
docker build -t deepspeech-federated:latest .
```

### Running Federated Training

**Step 1: Start the Central Aggregation Server**
```bash
python -m federated.server --min_clients 2 --num_rounds 5
```

**Step 2: Start Client Nodes (in separate terminals or containers)**
```bash
python -m federated.client --client_id 1 --data_dir ./data/client1
python -m federated.client --client_id 2 --data_dir ./data/client2
```

---

## Architecture Diagram

```text
 Client Node 1 (Local Audio)  ──► Whisper Model ──┐
                                                 ├──► Flower FedAvg Server ──► Global Whisper Weights
 Client Node 2 (Local Audio)  ──► Whisper Model ──┘
```

---
