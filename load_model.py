from example_model import Model, Config
import torch

if __name__ == "__main__":
    config = Config.from_pretrained("./checkpoints/TorchTrainer_2023-12-22_11-56-19/TorchTrainer_27337_00000_0_2023-12-22_11-56-19/checkpoint_000000/checkpoint")
    model = Model(config)
    model(torch.randn(1, 3, 64, 64))
