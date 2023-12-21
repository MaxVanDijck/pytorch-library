import ray

from deepspeed.accelerator import DeepSpeedAccelerator, get_accelerator
import deepspeed
import numpy as np
import datasets

from torchvision.models import resnet18
import torch.nn as nn
import torch
import io 
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, normalize
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, DataConfig
from torchvision import transforms
import logging

log = logging.getLogger("ray")
log.setLevel(logging.INFO)


def collate_fn(batch):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    x = []
    for item in batch['image']:
        image = pil_to_tensor(Image.open(io.BytesIO(item['bytes'])))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        x.append(image)
    x = torch.stack(x).to(torch.float32)
    y = torch.tensor(batch['label'], dtype=torch.uint8)
    x = normalize(x)
    return x, y

def get_model():
    model = resnet18(num_classes=200)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model

def get_ray_dataset():
    hf_dataset = datasets.load_dataset("zh-plus/tiny-imagenet")
    ray_ds = ray.data.from_huggingface(hf_dataset["train"])
    return ray_ds

def train_func(config):
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize DeepSpeed Engine
    deepspeed_config = config["deepspeed_config"]
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config,
    )
    device = get_accelerator().device_name(model.local_rank)
    
    ray_ds = get_ray_dataset()
    criterion = nn.CrossEntropyLoss().to(device)


    for epoch in range(config["num_epochs"]):
        for images, labels in ray_ds.iter_torch_batches(prefetch_batches=8, batch_size=config["train_batch_size"], collate_fn=collate_fn, local_shuffle_buffer_size=256):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log.info(loss)

if __name__ == "__main__":
    deepspeed_config = {
        "optimizer": {
            "type": "Adagrad",
            "params": {
                "lr": 2e-5,
            },
        },
        "scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 10}},
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},  # Turn this on if using AMPERE GPUs.
        "zero_optimization": {
            "stage": 0,
           "offload_optimizer": {
                "device": "cpu",
            },
            "offload_param": {
                "device": "cpu",
            },
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": True,
        "steps_per_print": 10,
        "train_micro_batch_size_per_gpu": 16,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
    }
    training_config = {
        "seed": 42,
        "num_epochs": 3,
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "deepspeed_config": deepspeed_config,
    }
    ray_datasets = {
        "train": get_ray_dataset(),
        "validation": get_ray_dataset()
    }
    trainer = TorchTrainer(
        train_func,
        train_loop_config=training_config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        datasets=ray_datasets,
        dataset_config=DataConfig(datasets_to_split=["train", "validation"]),
    )

    result = trainer.fit()

