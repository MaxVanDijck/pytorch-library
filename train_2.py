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

from transformers import TrainerCallback, TrainingArguments, Trainer
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
        logging_steps=10,
        eval_steps=10,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        save_strategy="steps",
        save_steps=1000,
        max_steps=config["steps_per_epoch"] * epochs,
        learning_rate=learning_rate,
        weight_decay=0,
        warmup_steps=0,
        label_names=["logits"],
        push_to_hub=False,
        report_to=None,
        disable_tqdm=True,  # declutter the output a little
        half_precision_backend="auto",
        gradient_checkpointing=False,
        deepspeed=config["deepspeed_config"],
    )

    model_config = Config(block_type="basic")
    model = Model(model_config)


    train_ds = train.get_dataset_shard("train")
    train_ds_iterable = train_ds.iter_torch_batches(prefetch_batches=4, batch_size=batch_size, collate_fn=collate_fn, local_shuffle_buffer_size=512)

    eval_ds = train.get_dataset_shard("eval")
    eval_ds_iterable = eval_ds.iter_torch_batches(prefetch_batches=4, batch_size=batch_size, collate_fn=collate_fn, local_shuffle_buffer_size=512)

    def compute_metrics(eval_pred):
        import numpy as np
        import evaluate
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
        callbacks=[LogCallback()]
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
            "env_vars": runtime_envvars,
            "excludes": ["checkpoints"]
        }
    )

    train_loop_config = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "deepspeed_config": None #os.path.abspath("deepspeed/ds_config_zero0.json"), # deep_speed_config # TODO(max) have the multiple deepspeed strategys as easy drop-ins
    }
    # https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=False,
        resources_per_worker={"CPU": 4, "GPU": 0}
    )
    datasets = {"train": get_ray_dataset(), "eval": get_ray_dataset()}
    train_ds_size = datasets["train"].count()
    train_loop_config["steps_per_epoch"] = train_ds_size // (train_loop_config["batch_size"] * 1)

    run_config=RunConfig(storage_path=os.path.abspath("checkpoints"))

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets=datasets,
        run_config=run_config,
    )

    trainer.fit()


