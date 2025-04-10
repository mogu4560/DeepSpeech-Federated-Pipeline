import argparse

class FederatedServer:
    def __init__(self, min_clients: int = 2, num_rounds: int = 5):
        self.min_clients = min_clients
        self.num_rounds = num_rounds

    def start(self):
        print(f"🚀 Starting Federated Averaging (FedAvg) Server...")
        print(f"Waiting for minimum {self.min_clients} client nodes on port 8080...")
        print(f"Configured for {self.num_rounds} aggregation rounds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    args = parser.parse_args()
    
    server = FederatedServer(args.min_clients, args.num_rounds)
    server.start()
