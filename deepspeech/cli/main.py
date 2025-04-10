import argparse
import sys
import logging
from deepspeech.federated.fl_client import FederatedSpeechClient
from deepspeech.federated.fl_server import FederatedSpeechServer
from deepspeech.audio.feature_extractor import AcousticFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_simulation(num_clients: int, rounds: int):
    print(f"🎙️ Starting DeepSpeech-Federated Simulation ({num_clients} client nodes, {rounds} rounds)...")
    server = FederatedSpeechServer(min_clients=num_clients, num_rounds=rounds)
    clients = [FederatedSpeechClient(client_id=i+1, data_path=f"./data/client_{i+1}") for i in range(num_clients)]

    current_weights = clients[0].get_parameters()

    for r in range(1, rounds + 1):
        client_updates = []
        for c in clients:
            updated_weights, num_samples, meta = c.fit(current_weights, {})
            client_updates.append((updated_weights, num_samples, meta["loss"]))
        
        round_res = server.run_federated_round(r, client_updates)
        current_weights = server.global_weights
        print(f"Round {r} Complete -> Avg Training Loss: {round_res['avg_training_loss']}")

    print("\n✅ Federated Speech-to-Text Model Training Complete! Global weights saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSpeech-Federated CLI Launcher")
    parser.add_argument("--clients", type=int, default=3, help="Number of decentralized client nodes")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL aggregation rounds")
    args = parser.parse_args()

    run_simulation(args.clients, args.rounds)
