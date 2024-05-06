from enum import StrEnum, auto

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging
from evaluate import load
import torch.nn.functional as F
import torch.nn as nn

from example_model import Config, Model
import math
import os
import tempfile
import time
from dataclasses import dataclass
from ranger import Ranger

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
import torch
import tqdm


import ray
from ray import train
import ray.util.scheduling_strategies
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint
from typing import Callable
from hooks import Hook


log = logging.getLogger(__name__)

@dataclass
class DatasetCollateFns:
    train: Callable | None
    eval: Callable | None
    test: Callable | None

    @classmethod
    def from_ray_collate_fn_dict(cls, collate_fns):
        return cls(
           collate_fns["train"] if "train" in collate_fns else None,
           collate_fns["valid"] if "valid" in collate_fns else None,
           collate_fns["test"] if "test" in collate_fns else None,
        )

@dataclass
class Metadata:
    length: int

    @classmethod
    def from_ray_dataset(cls, dataset):
        return cls(
            length=dataset.count()
        )

@dataclass
class DatasetMetadata:
    train: Metadata | None
    eval: Metadata | None
    test: Metadata | None

    @classmethod
    def from_ray_dataset_dict(cls, datasets):
        return cls(
           Metadata.from_ray_dataset(datasets["train"]) if "train" in datasets else None,
           Metadata.from_ray_dataset(datasets["valid"]) if "valid" in datasets else None,
           Metadata.from_ray_dataset(datasets["test"]) if "test" in datasets else None,
        )


@dataclass
class GlobalConfig:
    hydra: DictConfig
    deepspeed: DeepSpeedPlugin | None
    scaling_config: train.ScalingConfig
    run_config: train.RunConfig
    data_config: train.DataConfig
    dataset_metadata: DatasetMetadata
    dataset_collate_fns: DatasetCollateFns

class Stage(StrEnum):
    BEFORE_RUN = auto()
    AFTER_RUN = auto()
    BEFORE_TRAIN = auto()
    AFTER_TRAIN = auto()
    BEFORE_EVAL = auto()
    AFTER_EVAL = auto()

class Trainer:
    def __init__(
        self, 
        config: GlobalConfig, 
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._hooks: list[Hook] = []

    @property
    def max_epochs(self) -> int:
        return self.config.hydra.num_epochs

    @property
    def train_dataset(self) -> ray.data.Dataset:
        return train.get_dataset_shard("train")

    @property
    def valid_dataset(self) -> ray.data.Dataset:
        return train.get_dataset_shard("valid")

    @property
    def test_dataset(self) -> ray.data.Dataset:
        return train.get_dataset_shard("test")

    def register_hook(self, hook: Hook) -> None:
        self._hooks.append(hook)

    def call_hooks(self, stage: Stage) -> None:
        for hook in self._hooks:
            getattr(hook, stage)(self)

    def setup(self) -> None:
        # Ray.train has a bug somewhere that causes ACCELERATE_TORCH_DEVICE to not be set
        # properly on multi-gpu nodes
        cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        local_rank = int(os.environ["LOCAL_RANK"])
        device_id = cuda_visible_device[local_rank]
        # os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu" # TODO: comment out, need this for local training

        # Initialize accelerator
        kwargs = {
            "mixed_precision": self.config.hydra.mixed_precision,
            "gradient_accumulation_steps": self.config.hydra.gradient_accumulation_steps,
        }

        if self.config.hydra.use_deepseed:
            ds_plugin = self.config.deepspeed
            ds_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = self.config.hydra.batch_size
            kwargs.update(deepspeed_plugin=ds_plugin)

        self.accelerator = Accelerator(
            **kwargs
        )

        set_seed(self.config.hydra.seed)
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)

    def train(self) -> None:
        self.model.train()
        loss_total = torch.tensor(0.0).to(self.accelerator.device)

        train_dataloader = self.train_dataset.iter_torch_batches(
            batch_size=self.config.hydra.batch_size,
            collate_fn=self.config.dataset_collate_fns.train,
        )

        for step, batch in tqdm.tqdm(
            enumerate(train_dataloader), total=self.config.dataset_metadata.train.length // self.config.hydra.batch_size + 1
        ):
            with self.accelerator.accumulate(self.model):
                outputs = self.model(batch[0], batch[1])
                loss = outputs["loss"]
                loss_total += loss.item()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()


            if self.accelerator.is_main_process:
                epoch = "dunno"
                self.accelerator.print(
                    f"[epoch {epoch} step {step}] "
                    f"loss: {loss.item()}"
                    f"Learning rate: {self.lr_scheduler.get_lr()[0]}"
                )

            aggregated_loss = torch.mean(self.accelerator.gather(loss[None])).item()


            # as long as this is not the last step report here
            if step != (self.config.dataset_metadata.train.length // self.config.hydra.batch_size - 1):
                train.report(
                    {
                        "iteration": step,
                        "train_loss_batch": aggregated_loss,
                        "avg_train_loss_epoch": None,
                        "eval_loss": None,
                        "perplexity": None,
                        "num_iterations": step + 1,
                        "train_time_per_epoch": None,
                        "eval_time_per_epoch": None,
                        "avg_fwd_time_per_epoch": None,
                        "avg_bwd_time_per_epoch": None,
                        "learning_rate": self.lr_scheduler.get_lr()[0],
                    }
                )

            if self.config.hydra.as_test and step >= 5:
                break

    def eval(self) -> None:
        self.model.eval()
        losses = []
        labels = []
        logits = []

        eval_dataloader = self.valid_dataset.iter_torch_batches(batch_size=self.config.hydra.batch_size, collate_fn=self.config.dataset_collate_fns.eval)
        for step, batch in tqdm.tqdm(
            enumerate(eval_dataloader), total=self.config.dataset_metadata.eval.length // (self.config.hydra.batch_size + 1)
        ):

            batch = batch[0].to(self.accelerator.device), batch[1].to(self.accelerator.device)
            with torch.no_grad():
                outputs = self.model(batch[0], batch[1])

            loss = outputs["loss"]
            logits_out = F.softmax(outputs["logits"], dim=-1).argmax(1).tolist()

            # The tensors are gathered by concatenating them on the first dimension, so we
            # add a new dimension to the scalar loss to get a tensor of shape (K,) for K
            # workers.
            losses.append(self.accelerator.gather(loss[None]))
            labels += batch[1].tolist()
            logits += logits_out

            if self.config.hydra.as_test:
                break

        # We stack losses so that we have a tensor of shape (T, K) where T is the number of
        # steps and K is the number of workers.
        losses = torch.stack(losses)
        try:
            eval_loss = torch.mean(losses).item()
            perplexity = math.exp(eval_loss)
            precision_metric = load("precision")
            precision = precision_metric.compute(references=labels, predictions=logits, average='micro')["precision"]
            accuracy_metric = load("accuracy")
            accuracy = accuracy_metric.compute(references=labels, predictions=logits)["accuracy"]
        except OverflowError:
            perplexity = float("inf")

    def test(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        self.call_hooks(Stage.BEFORE_RUN)
        self.setup()
        for _ in range(self.max_epochs):
            self.call_hooks(Stage.BEFORE_TRAIN)
            self.train()
            self.call_hooks(Stage.AFTER_TRAIN)
            self.call_hooks(Stage.BEFORE_EVAL)
            self.eval()
            save(self.accelerator, self.model)
            self.call_hooks(Stage.AFTER_EVAL)

        self.test()
        self.call_hooks(Stage.AFTER_RUN)







def training_function(config: GlobalConfig):
    # Initialize model
    model_config = Config(block_type="basic")
    model = Model(model_config)

    # Initialize optimizer
    optimizer = Ranger(model.parameters())

    # Initialize lr_scheduler
    num_steps_per_epoch = config.dataset_metadata.train.length / config.hydra.batch_size
    total_training_steps = (
        num_steps_per_epoch * config.hydra.num_epochs // config.hydra.gradient_accumulation_steps
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=total_training_steps,
        gamma=0.5,
    )

    # run trainer
    trainer = Trainer(config, model, optimizer, lr_scheduler)
    trainer.run()



def save(accelerator, model):
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        accelerator.print(f"Saving the model locally at {temp_checkpoint_dir}")
        accelerator.wait_for_everyone()

        checkpoint_save_start = time.perf_counter()

        if accelerator.is_main_process:
            print("Saving tokenizer and config.")

        accelerator.wait_for_everyone()

        # Checkpointing strategy 1: Distributed checkpointing
        # This checkpointing method makes deepspeed checkpoints on each node
        # and then Ray Train will aggregate them to a central s3 bucket.
        # It should be done on all processes (not just the Rank 0)
        # aggregate_on_rank_0 = False
        # checkpoint_model(
        #     checkpoint_folder=tempdir,
        #     ckpt_id=epoch,
        #     model=model,
        #     epoch=epoch,
        #     last_global_step=step
        # )

        # Checkpointing strategy 2: Aggregate model on the rank 0 worker then upload
        aggregate_on_rank_0 = True
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            temp_checkpoint_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            safe_serialization=True,
            state_dict=accelerator.get_state_dict(model),
        )
        accelerator.wait_for_everyone()
        print("Checkpoint save time: ", time.perf_counter() - checkpoint_save_start)

        checkpoint_upload_start = time.perf_counter()

        # Create the checkpoint object to report to Ray Train and upload to storage.
        # If we aggregated the model on rank 0, we only need to report
        # the checkpoint from the rank 0 worker, since all other checkpoint
        # directories are empty (`save_pretrained` was a noop for other workers).
        if aggregate_on_rank_0:
            checkpoint = (
                Checkpoint.from_directory(temp_checkpoint_dir)
                if accelerator.is_main_process
                else None
            )
        else:
            # Distributed checkpointing should upload shards from each worker.
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        # Note: After `train.report`, in the case of remote storage,
        # the checkpoint directory will be uploaded to the remote storage.
        metrics = {}
        train.report(metrics, checkpoint=checkpoint)

        print(
            "Checkpoint upload time: ",
            time.perf_counter() - checkpoint_upload_start,
        )
        print(
            "Total checkpointing time: ",
            time.perf_counter() - checkpoint_save_start,
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(hydra_config: DictConfig):
    # Initialize Ray
    runtime_envvars = dict(os.environ)
    ray.init(
        runtime_env={
            "working_dir": ".",
            "env_vars": runtime_envvars,
            "excludes": ["checkpoints"],
        }
    )

    # Initialize Ray Datasets
    datasets = instantiate(hydra_config.dataset)
    collate_fns = DatasetCollateFns.from_ray_collate_fn_dict(instantiate(hydra_config.collate_fns))
    dataset_metadata = DatasetMetadata.from_ray_dataset_dict(datasets)

    # Setup Config
    deepspeed = DeepSpeedPlugin(hf_ds_config=hydra_config.deepspeed_config) if hydra_config.use_deepseed else None

    ray_scaling_config = instantiate(hydra_config.scaling_config)
    ray_run_config = instantiate(hydra_config.run_config)
    ray_data_config = instantiate(hydra_config.data_config)

    global_config = GlobalConfig(
        hydra_config,
        deepspeed,
        ray_scaling_config,
        ray_run_config,
        ray_data_config,
        dataset_metadata,
        collate_fns,
    )

    # Initialize TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=training_function,
        train_loop_config=global_config,
        scaling_config=global_config.scaling_config,
        run_config=global_config.run_config,
        datasets=datasets,
        dataset_config=global_config.data_config,
    )

    # Run Training
    result = trainer.fit()

    # Log Results
    # checkpoints are sorted in increasing score order.
    best_checkpoint, best_checkpoint_metrics = result.best_checkpoints[-1]
    log.info(f"Results are stored at: {result.path}")
    log.info(f"Best checkpoint is stored at: {best_checkpoint}")
    log.info(f"With {hydra_config.run_config.checkpoint_config.checkpoint_score_attribute}: {best_checkpoint_metrics[hydra_config.run_config.checkpoint_config.checkpoint_score_attribute]}")


if __name__ == "__main__":
    main() # type: ignore
