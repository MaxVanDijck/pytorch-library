import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging
from evaluate import load
import torch.nn.functional as F

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


log = logging.getLogger(__name__)

@dataclass
class Metadata:
    length: int

    @classmethod
    def from_ray_dataset(cls, dataset):
        return cls(dataset.count())


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
class GlobalConfig:
    hydra: DictConfig
    deepspeed: DeepSpeedPlugin | None
    scaling_config: train.ScalingConfig
    run_config: train.RunConfig
    data_config: train.DataConfig
    dataset_metadata: DatasetMetadata
    dataset_collate_fns: DatasetCollateFns

OPTIM_BETAS = (0.9, 0.999)
OPTIM_EPS = 1e-6
NUM_WARMUP_STEPS = 10
OPTIM_WEIGHT_DECAY = 0.0
ATTENTION_LAYER_NAME = "self_attn"




def evaluate(
    *, model, eval_ds, accelerator, bsize, ds_kwargs, as_test: bool = False
) -> tuple[float, float]:
    model.eval()
    losses = []
    labels = []
    logits = []

    eval_dataloader = eval_ds.iter_torch_batches(batch_size=bsize, **ds_kwargs)
    eval_ds_len = len(list(eval_ds.iter_batches(batch_size=1)))
    for step, batch in tqdm.tqdm(
        enumerate(eval_dataloader), total=eval_ds_len // (bsize + 1)
    ):

        batch = batch[0].to(accelerator.device), batch[1].to(accelerator.device)
        with torch.no_grad():
            outputs = model(batch[0], batch[1])

        loss = outputs["loss"]
        logits_out = F.softmax(outputs["logits"], dim=-1).argmax(1).tolist()

        # The tensors are gathered by concatenating them on the first dimension, so we
        # add a new dimension to the scalar loss to get a tensor of shape (K,) for K
        # workers.
        losses.append(accelerator.gather(loss[None]))
        labels += batch[1].tolist()
        logits += logits_out

        if as_test:
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
    return perplexity, eval_loss, precision, accuracy



def training_function(config: GlobalConfig):
    # Train has a bug somewhere that causes ACCELERATE_TORCH_DEVICE to not be set
    # properly on multi-gpu nodes
    cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = cuda_visible_device[local_rank]
    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"
    # os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu" # TODO: comment out, need this for local training

    # Initialize accelerator
    kwargs = {
        "mixed_precision": config.hydra.mixed_precision,
        "gradient_accumulation_steps": config.hydra.gradient_accumulation_steps,
    }

    if config.hydra.use_deepseed:
        ds_plugin = config.deepspeed
        ds_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = config.hydra.batch_size
        kwargs.update(deepspeed_plugin=ds_plugin)

    accelerator = Accelerator(
        **kwargs
    )

    set_seed(config.hydra.seed)
    train_ds = train.get_dataset_shard("train")
    valid_ds = train.get_dataset_shard("valid")
    train_ds_len = config.dataset_metadata.train.length # TODO make into dataclass

    # Initialize model
    model_config = Config(block_type="basic")
    model = Model(model_config)




    optimizer_cls = (
        Ranger
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    optimizer = optimizer_cls(
        model.parameters(),
    )
    

    # Instantiate scheduler
    # Creates Dummy Scheduler if `scheduler` was specified in the config file or
    # else, creates `args.lr_scheduler_type` Scheduler
    # get train and valid dataset lengths

    num_steps_per_epoch = train_ds_len / config.hydra.batch_size
    total_training_steps = (
        num_steps_per_epoch * 1 // config.hydra.gradient_accumulation_steps
    )

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=total_training_steps,
            gamma=0.5,
        )
    else:
        num_devices = 1
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=NUM_WARMUP_STEPS * num_devices,
            total_num_steps=total_training_steps * num_devices,
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
        print("Number of batches on main process", train_ds_len // config.hydra.batch_size)

    for epoch in range(config.hydra.num_epochs):
        fwd_time_sum, bwd_time_sum, optim_step_time_sum, load_batch_time_sum = 0, 0, 0, 0
        s_epoch = time.time()
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)

        train_dataloader = train_ds.iter_torch_batches(
            batch_size=config.hydra.batch_size,
            collate_fn=config.dataset_collate_fns.train,
        )

        for step, batch in tqdm.tqdm(
            enumerate(train_dataloader), total=train_ds_len // config.hydra.batch_size + 1
        ):

            # We could avoid this line since we set the accelerator with
            # `device_placement=True`.
            # batch = batch[0].to(accelerator.device), batch[1].to(accelerator.device)
            s_load_batch = time.time()
            with accelerator.accumulate(model):
                e_load_batch = time.time()
                load_batch_time = e_load_batch - s_load_batch
                load_batch_time_sum += load_batch_time
                s_fwd = time.time()
                outputs = model(batch[0], batch[1])
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
                s_load_batch = time.time()


            if accelerator.is_main_process:
                accelerator.print(
                    f"[epoch {epoch} step {step}] "
                    f"loss: {loss.item()} step-time: {e_opt_step - s_fwd}"
                    f"Learning rate: {lr_scheduler.get_lr()[0]}"
                )

            aggregated_loss = torch.mean(accelerator.gather(loss[None])).item()


            # as long as this is not the last step report here
            if step != (train_ds_len // config.hydra.batch_size - 1):
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

            if config.hydra.as_test and step >= 5:
                break

        e_epoch = time.time()
        accelerator.print("Train time per epoch: ", e_epoch - s_epoch)

        eval_s_epoch = time.time()
        print("Running evaluation ...")
        perplex, eloss, precision, accuracy = evaluate(
            model=model,
            eval_ds=valid_ds,
            accelerator=accelerator,
            bsize=32,
            ds_kwargs={"collate_fn": config.dataset_collate_fns.eval},
            as_test=config.hydra.as_test,
        )
        accelerator.print("Eval result loss", eloss)
        accelerator.print("Eval perplex", perplex)

        eval_e_epoch = time.time()
        accelerator.print("Eval time per epoch: ", eval_e_epoch - eval_s_epoch)
        accelerator.print("avg fwd time: ", fwd_time_sum / (step + 1))
        accelerator.print("avg bwd time: ", bwd_time_sum / (step + 1))
        accelerator.print("avg batch time: ", load_batch_time_sum / (step + 1))
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
            "precision": precision,
            "accuracy": accuracy,
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


        if config.hydra.as_test:
            break



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
