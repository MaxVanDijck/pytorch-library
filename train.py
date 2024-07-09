from enum import StrEnum, auto

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from hooks.logger import LoggerHook
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
from dataclasses import dataclass, field
from ranger import Ranger

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
import torch


import ray
from ray import train
import ray.util.scheduling_strategies
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint
from typing import Callable
from hooks import Hook
from hooks.wandb import WandbHook


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
    BEFORE_TRAIN_STEP = auto()
    AFTER_TRAIN_STEP = auto()
    BEFORE_EVAL = auto()
    AFTER_EVAL = auto()
    AFTER_EVAL_STEP = auto()


@dataclass
class HookState:
    """
    Container class for all metrics/values hooks should access
    """
    epoch: int | None = None
    learning_rate: float | None = None

    train_loss_step: float | None = None
    train_step: int | None = None
    train_metrics: dict = field(default_factory=dict)

    val_loss_step: float | None = None
    val_step: int | None = None
    val_metrics: dict = field(default_factory=dict)



class Trainer:
    hook_state = HookState()

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
        """Calls registered hooks in main process"""
        if self.accelerator.is_main_process:
            for hook in self._hooks:
                getattr(hook, stage)(self)

    def setup(self) -> None:
        # Ray.train has a bug somewhere that causes ACCELERATE_TORCH_DEVICE to not be set
        # properly on multi-gpu nodes
        cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        local_rank = int(os.environ["LOCAL_RANK"])
        device_id = cuda_visible_device[local_rank]
        os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"
        # os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu" # TODO: find a way to automatically enable cpu, need this for local training

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
            local_shuffle_buffer_size=32
        )

        for step, batch in enumerate(train_dataloader):
            self.call_hooks(Stage.BEFORE_TRAIN_STEP)

            with self.accelerator.accumulate(self.model):
                outputs = self.model(batch[0], batch[1])
                loss = outputs["loss"]
                loss_total += loss.item()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()


            aggregated_loss = torch.mean(self.accelerator.gather(loss[None])).item()
            self.hook_state.train_step = step
            self.hook_state.train_loss_step = aggregated_loss
            self.hook_state.learning_rate = self.lr_scheduler.get_lr()[0]
            self.call_hooks(Stage.AFTER_TRAIN_STEP)

            if self.config.hydra.as_test and step >= 5:
                break

    def eval(self) -> None:
        self.model.eval()
        losses = []
        labels = []
        logits = []
        raw_output = []

        eval_dataloader = self.valid_dataset.iter_torch_batches(batch_size=self.config.hydra.batch_size, collate_fn=self.config.dataset_collate_fns.eval)
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = self.model(batch[0], batch[1])

            loss = outputs["loss"]

            # The tensors are gathered by concatenating them on the first dimension, so we
            # add a new dimension to the scalar loss to get a tensor of shape (K,) for K
            # workers.
            losses.append(self.accelerator.gather(loss[None]))

            labels += batch[1].tolist()
            logits += torch.round(outputs["logits"].squeeze(1)).tolist()
            raw_output += outputs["logits"].squeeze(1).tolist()

            self.hook_state.val_step = step
            self.hook_state.val_loss_step = loss.item()
            self.call_hooks(Stage.AFTER_EVAL_STEP)

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
            roc_auc_score_metric = load("roc_auc")
            roc_auc_score = roc_auc_score_metric.compute(references=labels, prediction_scores=raw_output)["roc_auc"]
        except OverflowError:
            perplexity = float("inf")
            precision = float("inf")
            accuracy = float("inf")
            roc_auc_score = float("inf")

        # TODO: refine saving + recording model checkpoints alongside metrics
        self.hook_state.val_metrics = {
            "perplexity": perplexity,
            "precision": precision,
            "accuracy": accuracy,
            "roc_auc_score": roc_auc_score,
        }

        

    def test(self) -> None:
        pass

    def run(self) -> None:
        self.setup()
        self.call_hooks(Stage.BEFORE_RUN)
        for epoch in range(self.max_epochs):
            self.hook_state.epoch = epoch
            self.call_hooks(Stage.BEFORE_TRAIN)
            self.train()
            self.call_hooks(Stage.AFTER_TRAIN)
            self.call_hooks(Stage.BEFORE_EVAL)
            self.eval()
            save(self.accelerator, self.model, self.hook_state.val_metrics)
            self.call_hooks(Stage.AFTER_EVAL)

        self.test()
        self.call_hooks(Stage.AFTER_RUN)







def training_function(config: GlobalConfig):
    # Initialize model
    model_config = Config(block_type="basic")
    model = Model(model_config)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters())

    # Initialize lr_scheduler
    num_steps_per_epoch = config.dataset_metadata.train.length / config.hydra.batch_size
    total_training_steps = (
        num_steps_per_epoch * config.hydra.num_epochs // config.hydra.gradient_accumulation_steps
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=2000,
        gamma=0.1,
    )

    # run trainer
    trainer = Trainer(config, model, optimizer, lr_scheduler)
    trainer.register_hook(WandbHook())
    trainer.register_hook(LoggerHook())
    trainer.run()



def save(accelerator, model, metrics):
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
        log_to_driver=True,
        runtime_env={
            "working_dir": ".",
            "env_vars": runtime_envvars,
            "excludes": ["checkpoints", "data"],
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
    # checkpoints are sorted in increasing score order, therefore grab last
    best_checkpoint, best_checkpoint_metrics = result.best_checkpoints[-1]
    log.info(f"Results are stored at: {result.path}")
    log.info(f"Best checkpoint is stored at: {best_checkpoint}")
    log.info(f"With {hydra_config.run_config.checkpoint_config.checkpoint_score_attribute}: {best_checkpoint_metrics[hydra_config.run_config.checkpoint_config.checkpoint_score_attribute]}")


if __name__ == "__main__":
    main() # type: ignore
