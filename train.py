import argparse
import contextlib
import glob
import math
import os
import random
import socket
import time

import torch
import torch.distributed
import torch.multiprocessing

from data import load_vocabulary
from dataset import create_training_dataset
from model import Transformer
from utils import init_logger, get_logger


num_layers = 6
num_heads = 16
dim_model = 1024
dim_ffn = 4096
dropout = 0.1

# None means the embeddings are automatically shared if the source and target
# vocabularies are the same.
share_embeddings = None

max_source_len = 150
max_target_len = 150

batch_type = "tokens"
batch_size = 16000
effective_batch_size = 400000
label_smoothing = 0.1

# Learning rate schedule: inverse square root
learning_rate = 0.001
warmup_steps = 4000
initial_learning_rate = 1e-7
adam_betas = (0.9, 0.98)

report_every = 20
save_every = 500
max_step = 50000
keep_checkpoints = 10

compile_model = False
mixed_precision = True
seed = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to the source training file")
    parser.add_argument("--tgt", required=True, help="Path to the source training file")
    parser.add_argument(
        "--src_vocab", required=True, help="Path to the source vocabulary"
    )
    parser.add_argument(
        "--tgt_vocab", required=True, help="Path to the target vocabulary"
    )
    parser.add_argument(
        "--save_dir", default="checkpoints/", help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use"
    )
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        train,
        args=(
            args.num_gpus,
            args.src,
            args.tgt,
            args.src_vocab,
            args.tgt_vocab,
            args.save_dir,
        ),
        nprocs=args.num_gpus,
        join=True,
    )


def train(
    rank,
    world_size,
    source_path,
    target_path,
    source_vocabulary_path,
    target_vocabulary_path,
    save_dir,
):
    """Runs the training on a single device."""

    is_master = rank == 0

    if is_master:
        init_logger()

    logger = get_logger()

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    # Initialize distributed training.
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    source_vocabulary, _ = load_vocabulary(source_vocabulary_path)
    target_vocabulary, _ = load_vocabulary(target_vocabulary_path)
    padding_idx = 0

    accum_steps = (
        effective_batch_size // (batch_size * world_size)
        if effective_batch_size is not None
        else 1
    )

    dataset = create_training_dataset(
        source_path,
        target_path,
        source_vocabulary,
        target_vocabulary,
        batch_size=batch_size,
        batch_type=batch_type,
        pad_to_multiple=8 if compile_model else 1,
        maximum_source_length=max_source_len,
        maximum_target_length=max_target_len,
        device=device,
        num_accum_batches=accum_steps,
        num_shards=world_size,
        shard_index=rank,
        seed=seed,
    )

    model = create_model(
        device,
        src_vocab_size=len(source_vocabulary),
        tgt_vocab_size=len(target_vocabulary),
        num_layers=num_layers,
        num_heads=num_heads,
        dim_model=dim_model,
        dim_ffn=dim_ffn,
        dropout=dropout,
        share_embeddings=(
            source_vocabulary_path == target_vocabulary_path
            if share_embeddings is None
            else share_embeddings
        ),
        padding_idx=padding_idx,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=1,  # The learning rate is defined by the scheduler.
        betas=adam_betas,
        fused=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        inv_sqrt_decay(learning_rate, warmup_steps, initial_learning_rate),
    )

    ce_loss = torch.nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        ignore_index=padding_idx,
        reduction="sum",
    )

    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    step = 0

    if is_master and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = get_latest_checkpoint(save_dir)

    if checkpoint_path is not None:
        logger.info("Restoring checkpoint %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        step = int(checkpoint["step"])
        if step >= max_step:
            logger.info("Training already reached max_step = %d", max_step)
            return

        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        scaler.load_state_dict(checkpoint["grad_scaler"])

    last_log_time = time.time()
    num_tokens = 0

    if is_master:
        logger.info("Accumulate gradients of %d batches", accum_steps)
        logger.info(
            "Optimize %d parameters",
            sum(parameter.numel() for parameter in trainable_parameters),
        )

    for batches in dataset:
        # Compute the global batch size for this training step.
        sample_size = sum(target.ne(padding_idx).sum() for _, _, target in batches)
        sample_size = torch.as_tensor(sample_size, dtype=torch.float32, device=device)
        torch.distributed.all_reduce(sample_size, op=torch.distributed.ReduceOp.SUM)

        total_loss = 0

        for i, (source, target_in, target_out) in enumerate(batches):
            last_batch = i + 1 == len(batches)

            with torch.autocast(
                device.type, dtype=torch.float16, enabled=mixed_precision
            ):
                logits = model(source, target_in)
                loss = ce_loss(logits.view(-1, logits.shape[-1]), target_out.view(-1))

                # Multiply by world_size because DDP divides the gradients by world_size.
                loss = loss * world_size / sample_size

            # Only synchronize gradients for the last accumulated batch.
            with (contextlib.nullcontext() if last_batch else model.no_sync()):
                scaler.scale(loss).backward()

            num_tokens += source.ne(padding_idx).sum().item()
            num_tokens += target_in.ne(padding_idx).sum().item()
            total_loss += loss.item() / world_size

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        step += 1

        if step % report_every == 0:
            # Aggregate the metrics from all ranks and send the result in the master process.
            stats = torch.as_tensor(
                [num_tokens, total_loss],
                dtype=torch.float32,
                device=device,
            )
            torch.distributed.reduce(stats, dst=0, op=torch.distributed.ReduceOp.SUM)

            if is_master:
                num_tokens, total_loss = stats.tolist()

                current_time = time.time()
                elapsed_time = current_time - last_log_time
                last_log_time = current_time

                logger.info(
                    "step = %d"
                    " ; tokens/s = %d"
                    " ; learning rate = %f"
                    " ; loss = %f",
                    step,
                    int(num_tokens / elapsed_time),
                    scheduler.get_last_lr()[0],
                    total_loss,
                )

            num_tokens = 0

        if step % save_every == 0 or step == max_step:
            if is_master:
                checkpoint = {
                    "grad_scaler": scaler.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "model": model.module.state_dict(),
                    "model_config": model.module.config,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }

                save_path = os.path.join(save_dir, "checkpoint-%d.pt" % step)
                logger.info("Saving checkpoint %s", save_path)
                torch.save(checkpoint, save_path)
                clean_checkpoint_directory(save_dir, keep_checkpoints)

        if step == max_step:
            break


def inv_sqrt_decay(lr, warmup_steps, initial_lr):
    def _fn(step):
        if step < warmup_steps:
            return initial_lr + (lr - initial_lr) * (step / warmup_steps)
        else:
            return lr * math.sqrt(warmup_steps / step)

    return _fn


def create_model(device, **kwargs):
    model = Transformer(**kwargs)

    if compile_model:
        model = torch.compile(model)

    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        static_graph=True,
    )

    return model


def get_checkpoint_step(path):
    """Returns the training step from a checkpoint."""
    filename = os.path.basename(path)
    filename, _ = os.path.splitext(filename)
    return int(filename.split("-")[-1])


def get_checkpoints(directory):
    """Returns the list of checkpoints in a directory, ordered on the step number."""
    if not os.path.exists(directory):
        return []

    checkpoints = glob.glob(os.path.join(directory, "checkpoint-*.pt"))
    checkpoints = sorted(checkpoints, key=get_checkpoint_step)
    return checkpoints


def clean_checkpoint_directory(directory, max_to_keep):
    """Removes old checkpoints to keep a maximum number of checkpoints in a directory."""
    checkpoints = get_checkpoints(directory)

    while len(checkpoints) > max_to_keep:
        os.remove(checkpoints.pop(0))


def get_latest_checkpoint(directory):
    """Returns the latest checkpoint in a directory."""
    checkpoints = get_checkpoints(directory)
    return checkpoints[-1] if checkpoints else None


def get_port():
    """Returns an available port number."""
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_port())
    main()
