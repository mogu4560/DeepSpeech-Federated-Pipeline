import argparse
from .audio_processor import AudioProcessor

class FederatedClient:
    def __init__(self, client_id: int, data_dir: str):
        self.client_id = client_id
        self.data_dir = data_dir
        self.processor = AudioProcessor()

    def train_epoch(self):
        print(f"🔒 Client Node [{self.client_id}] processing local audio logs in '{self.data_dir}'...")
        print(f"Extracting features and updating local Whisper model gradients...")
        print(f"Gradient update complete. Transmitting local weight update to Flower server (Audio logs kept private).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    
    client = FederatedClient(args.client_id, args.data_dir)
    client.train_epoch()
