import argparse
from train import collate_fn
from example_model import Config, Model
from train import get_ray_dataset
from filelock import FileLock
import functools
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import tree
from typing import Tuple

import deepspeed  # noqa: F401

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
import torch
import torch.nn as nn
import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import ray
from ray import train
import ray.util.scheduling_strategies
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint



OPTIM_BETAS = (0.9, 0.999)
OPTIM_EPS = 1e-8
NUM_WARMUP_STEPS = 10
OPTIM_WEIGHT_DECAY = 0.0
ATTENTION_LAYER_NAME = "self_attn"



def evaluate(
    *, model, eval_ds, accelerator, bsize, ds_kwargs, as_test: bool = False
) -> Tuple[float, float]:
    model.eval()
    losses = []

    eval_dataloader = eval_ds.iter_torch_batches(batch_size=bsize, **ds_kwargs)
    eval_ds_len = len(list(eval_ds.iter_batches(batch_size=1)))
    for step, batch in tqdm.tqdm(
        enumerate(eval_dataloader), total=eval_ds_len // (bsize + 1)
    ):
        with torch.no_grad():
            outputs = model(*batch)

        loss = outputs["loss"]
        # The tensors are gathered by concatenating them on the first dimension, so we
        # add a new dimension to the scalar loss to get a tensor of shape (K,) for K
        # workers.
        losses.append(accelerator.gather(loss[None]))

        if as_test:
            break

    # We stack losses so that we have a tensor of shape (T, K) where T is the number of
    # steps and K is the number of workers.
    losses = torch.stack(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss



def training_function(kwargs: dict):
    print("training_function called")

    # Train has a bug somewhere that causes ACCELERATE_TORCH_DEVICE to not be set
    # properly on multi-gpu nodes
    cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = cuda_visible_device[local_rank]
    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"
    os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"

    config = kwargs


    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    gradient_accumulation_steps = int(1)

    # Get deepspeed config to setup the batch size per device
    ds_plugin = config["ds_plugin"]
    ds_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = batch_size

    # Initialize accelerator
    accelerator = Accelerator(
        # deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='no',
    )

    set_seed(seed)

    # train_ds is the local shard for this model
    train_ds = train.get_dataset_shard("train")
    valid_ds = train.get_dataset_shard("valid")

    train_ds_len = len(list(train_ds.iter_batches(batch_size=1)))

    
    model_config = Config(block_type="basic")
    model = Model(model_config)



    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    optimizer = optimizer_cls(
        model.parameters(),
        lr=lr,
        betas=OPTIM_BETAS,
        weight_decay=0.01,
        eps=OPTIM_EPS,
    )

    # Instantiate scheduler
    # Creates Dummy Scheduler if `scheduler` was specified in the config file or
    # else, creates `args.lr_scheduler_type` Scheduler
    # get train and valid dataset lengths

    num_steps_per_epoch = 1000
    total_training_steps = (
        num_steps_per_epoch * num_epochs // gradient_accumulation_steps
    )

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS * 1,
            num_training_steps=total_training_steps * 1,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=NUM_WARMUP_STEPS * args.num_devices,
            total_num_steps=total_training_steps * args.num_devices,
        )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the
    # same order we gave them to the prepare method.
    s = time.time()
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    print(f"Prepare done in {time.time() - s} seconds.")

    # Now we train the model
    if accelerator.is_main_process:
        print("Starting training ...")
        print("Number of batches on main process", train_ds_len // batch_size)

    for epoch in range(num_epochs):
        fwd_time_sum, bwd_time_sum, optim_step_time_sum = 0, 0, 0
        s_epoch = time.time()
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)

        train_dataloader = train_ds.iter_torch_batches(
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        for step, batch in tqdm.tqdm(
            enumerate(train_dataloader), total=train_ds_len // batch_size + 1
        ):

            # We could avoid this line since we set the accelerator with
            # `device_placement=True`.
            with accelerator.accumulate(model):
                s_fwd = time.time()
                outputs = model(*batch)
                loss = outputs["loss"]
                loss_sum += loss.item()
                e_fwd = time.time()
                fwd_time = e_fwd - s_fwd
                fwd_time_sum += fwd_time
                s_bwd = time.time()
                accelerator.backward(loss)
                e_bwd = time.time()
                bwd_time = e_bwd - s_bwd
                bwd_time_sum += bwd_time

                s_opt_step = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                e_opt_step = time.time()
                optim_step_time_sum += e_opt_step - s_opt_step

            if accelerator.is_main_process:
                accelerator.print(
                    f"[epoch {epoch} step {step}] "
                    f"loss: {loss.item()} step-time: {e_opt_step - s_fwd}"
                )

            aggregated_loss = torch.mean(accelerator.gather(loss[None])).item()


            # as long as this is not the last step report here
            if step != (train_ds_len // batch_size - 1):
                train.report(
                    {
                        "epoch": epoch,
                        "iteration": step,
                        "train_loss_batch": aggregated_loss,
                        "avg_train_loss_epoch": None,
                        "eval_loss": None,
                        "perplexity": None,
                        "num_iterations": step + 1,
                        "train_time_per_epoch": None,
                        "eval_time_per_epoch": None,
                        "fwd_time": fwd_time,
                        "bwd_time": bwd_time,
                        "avg_fwd_time_per_epoch": None,
                        "avg_bwd_time_per_epoch": None,
                        "learning_rate": lr_scheduler.get_lr()[0],
                    }
                )

            if config["as_test"]:
                break

        e_epoch = time.time()
        accelerator.print("Train time per epoch: ", e_epoch - s_epoch)

        eval_s_epoch = time.time()
        print("Running evaluation ...")
        perplex, eloss = evaluate(
            model=model,
            eval_ds=valid_ds,
            accelerator=accelerator,
            bsize=32,
            ds_kwargs={"collate_fn": collate_fn},
            as_test=config["as_test"],
        )
        accelerator.print("Eval result loss", eloss)
        accelerator.print("Eval perplex", perplex)

        eval_e_epoch = time.time()
        accelerator.print("Eval time per epoch: ", eval_e_epoch - eval_s_epoch)
        accelerator.print("avg fwd time: ", fwd_time_sum / (step + 1))
        accelerator.print("avg bwd time: ", bwd_time_sum / (step + 1))
        accelerator.print("avg opt step time: ", optim_step_time_sum / (step + 1))

        metrics = {
            "epoch": epoch,
            "iteration": step,
            "train_loss_batch": aggregated_loss,
            "avg_train_loss_epoch": loss_sum.item() / (step + 1),
            "eval_loss": eloss,
            "perplexity": perplex,
            "num_iterations": step + 1,
            "train_time_per_epoch": e_epoch - s_epoch,
            "eval_time_per_epoch": eval_e_epoch - eval_s_epoch,
            "fwd_time": fwd_time,
            "bwd_time": bwd_time,
            "avg_fwd_time_per_epoch": fwd_time_sum / (step + 1),
            "avg_bwd_time_per_epoch": bwd_time_sum / (step + 1),
            "learning_rate": lr_scheduler.get_lr()[0],
        }

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


        if config["as_test"]:
            break



def main():
    config = {
        "as_test": True,
        "lr": 1e-2,
        "num_epochs": 10,
        "seed": 42,
        "batch_size": 32,
        "gradient_accumulation_steps": 10,
    }

    # Add deepspeed plugin to the config
    ds_plugin = DeepSpeedPlugin(hf_ds_config=config.get("deepspeed/ds_config_zero0.json"))
    config.update(ds_plugin=ds_plugin)


    runtime_envvars = dict(os.environ)
    ray.init(
        runtime_env={
            "working_dir": ".",
            "env_vars": runtime_envvars,
            "excludes": ["checkpoints"]
        }
    )

    # Read data
    datasets = {"train": get_ray_dataset(), "valid": get_ray_dataset()}

    trainer = TorchTrainer(
        training_function,
        train_loop_config=config,
        run_config=train.RunConfig(
            storage_path=os.path.abspath('checkpoints'),
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="perplexity",
                checkpoint_score_order="min",
            ),
        ),
        scaling_config=train.ScalingConfig(
            num_workers=1,
            use_gpu=False,
            resources_per_worker={"CPU": 4, "GPU": 0},
        ),
        datasets=datasets,
        dataset_config=ray.train.DataConfig(datasets_to_split=["train", "valid"]),
    )

    result: train.Result = trainer.fit()
    # `best_checkpoints` are sorted in increasing score order.
    # (Ex: in this case, negative perplexity, since we set `checkpoint_score_order=min`)
    best_checkpoint, best_checkpoint_metrics = result.best_checkpoints[-1]

    print("Results are stored at:")
    print(result.path)
    print("Best checkpoint is stored at:")
    print(best_checkpoint)
    print(f"With perplexity: {best_checkpoint_metrics['perplexity']}")


if __name__ == "__main__":
    main()