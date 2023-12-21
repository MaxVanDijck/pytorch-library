from pytorch_lightning.strategies import strategy
import ray
import os

from ray import train
from train import get_ray_dataset
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.train.huggingface.transformers import (
    prepare_trainer,
    RayTrainReportCallback
)

from transformers import TrainingArguments, Trainer
from example_model import Model, Config

from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor, normalize
import torch
from PIL import Image
import io

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
    return {"tensor":x, "labels":y}


def train_func(config: dict):
    os.environ["OMP_NUM_THREADS"] = str(
        train.get_context().get_trial_resources().bundles[-1].get("CPU", 1)
    )
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 2)
    learning_rate = config.get("learning_rate", 0.00002)

    training_args = TrainingArguments(
        "output",
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        max_steps=100 * epochs,
        learning_rate=learning_rate,
        weight_decay=0,
        warmup_steps=0,
        label_names=["class"],
        push_to_hub=False,
        report_to=None,
        disable_tqdm=True,  # declutter the output a little
        fp16=False,
        gradient_checkpointing=False,
        # deepspeed=config["deepspeed_config"], # TODO(max): this *should* be the only line you need to comment out to run distributed
    )

    model_config = Config(block_type="basic")
    model = Model(model_config)


    train_ds = train.get_dataset_shard("train")
    train_ds_iterable = train_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_iterable,
        eval_dataset=train_ds_iterable,
    )

    # Add callback to report checkpoints to Ray Train
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()





if __name__ == "__main__":
    runtime_envvars = dict(os.environ)
    ray.init(
        runtime_env={
            "working_dir": ".",
            "env_vars": runtime_envvars
        }
    )

    deep_speed_config = {
        "optimizer": {
            "type": "Adagrad",
        },
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},  # Turn this on if using AMPERE GPUs.
        "zero_optimization": {
            "stage": 1,
           "offload_optimizer": {
                "device": "cpu",
            },
            "offload_param": {
                "device": "cpu",
            },
        },
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }
    train_loop_config = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "deepspeed_config": deep_speed_config
    }
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=False,
    )
    datasets = {"train": get_ray_dataset()}
    run_config=RunConfig(storage_path="~/max/pytorch-library/checkpoints")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets=datasets,
        run_config=run_config,
    )

    trainer.fit()


