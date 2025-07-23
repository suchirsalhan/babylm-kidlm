import os
import time
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import (
    OPTConfig,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import SaveStrategy

import wandb

# --- Constants --- #
TRAIN_EPOCHS = 10
GLOBAL_BATCH_SIZE = 64  # 64 * 16k = 1M tokens per batch


def get_deepspeed_config(accumulation_steps, num_devices):
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
        },
        "train_batch_size": GLOBAL_BATCH_SIZE // num_devices,
        "gradient_accumulation_steps": accumulation_steps,
        "bf16": {"enabled": True},
    }


class CustomCheckpointingCallback(TrainerCallback):
    """
    BabyLM-style checkpointing:
    - Every 1M tokens until 10M
    - Every 10M tokens until 100M
    - Every 100M tokens until 1B
    """

    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.total_tokens = 1_000_000_000
        self.token_checkpoints = (
            [i * 1_000_000 for i in range(1, 11)] +
            [i * 10_000_000 for i in range(2, 11)] +
            [i * 100_000_000 for i in range(2, 11)]
        )
        self.token_checkpoints = sorted(set(self.token_checkpoints))
        self.next_checkpoint_idx = 0
        self.tokens_seen = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        tokens_this_step = GLOBAL_BATCH_SIZE * self.seq_len
        self.tokens_seen += tokens_this_step

        if self.next_checkpoint_idx < len(self.token_checkpoints):
            next_threshold = self.token_checkpoints[self.next_checkpoint_idx]
            if self.tokens_seen >= next_threshold:
                print(f"[Checkpoint] Saving at {self.tokens_seen:,} tokens seen (threshold: {next_threshold:,})")
                control.should_save = True
                self.next_checkpoint_idx += 1

        return control


def train_model(
    model_type="opt",
    seq_len=128,
    use_deepspeed=False,
    push_to_hub=True,
    dry_run=False,
    num_devices=4,
    accumulation_steps=1,
    dataset=None,
    resume=True,
    resume_from=None,
):
    # --- Dataset Setup --- #
    if dataset is None:
        dataset_name = f"train_100M_{seq_len}_single_shuffle"
        dataset = f"Talking-Babies/{dataset_name}"
    else:
        dataset_name = dataset.split("/")[-1]

    sanitized_dataset_name = dataset_name.replace("/", "-")

    print(f"Loading dataset: {dataset}")
    try:
        dataset = load_dataset(dataset)
    except Exception as e:
        print(f"Dataset '{dataset}' not found.\nError: {e}")
        exit(1)

    dataset = dataset.map(lambda x: {"labels": x["input_ids"]}, num_proc=16)
    train_dataset = dataset["train"]

    if dry_run:
        train_dataset = train_dataset.select(range(100))
        output_dir = f"./dryruns/{model_type}-babylm-{seq_len}"
    else:
        output_dir = f"./checkpoints/{model_type}-babylm-{seq_len}"

    os.makedirs(output_dir, exist_ok=True)
    run_name = f"{model_type}_babylm_{seq_len}"

    # --- Checkpoint Detection Logic --- #
    latest_checkpoint = None
    if resume_from:
        if os.path.isdir(resume_from):
            latest_checkpoint = resume_from
            print(f"🔁 Resuming from specific checkpoint: {latest_checkpoint}")
        else:
            raise ValueError(f"Specified checkpoint path does not exist: {resume_from}")
    elif resume and not dry_run:
        checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
        if checkpoints:
            latest_checkpoint = str(checkpoints[-1])
            print(f"🔁 Resuming from latest checkpoint: {latest_checkpoint}")
        else:
            print("ℹ️ No checkpoint found. Starting from scratch.")
    else:
        print("🚫 Resumption disabled or not applicable. Starting from scratch.")

    per_device_batch_size = GLOBAL_BATCH_SIZE / (accumulation_steps * num_devices)
    if int(per_device_batch_size) != per_device_batch_size:
        raise ValueError(
            f"Batch size {per_device_batch_size} is not an integer. "
            f"Adjust GLOBAL_BATCH_SIZE, num_devices, or accumulation_steps."
        )
    per_device_batch_size = int(per_device_batch_size)
    print(f"Per device batch size: {per_device_batch_size} (effective batch size = {accumulation_steps} * {num_devices})")

    # --- Model Setup --- #
    if latest_checkpoint:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
    else:
        if model_type == "opt":
            config = OPTConfig(
                vocab_size=50257,
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
                ffn_dim=3072,
                max_position_embeddings=seq_len,
            )
            model = OPTForCausalLM(config)

        elif model_type == "mamba":
            from transformers import MambaConfig, MambaForCausalLM
            config = MambaConfig(
                vocab_size=50257,
                hidden_size=768,
                num_hidden_layers=32,
            )
            model = MambaForCausalLM(config)

    # --- WandB Logging --- #
    local_rank = int(os.environ.get("RANK", 0))
    if local_rank == 0:
        wandb.init(
            entity="babylm-interaction",
            project=f"{model_type}-models",
            name=run_name,
            mode="disabled" if dry_run else "online",
            config={"resume_from": latest_checkpoint if latest_checkpoint else "none"}
        )

    total_steps = TRAIN_EPOCHS * len(train_dataset) // GLOBAL_BATCH_SIZE
    warmup_steps = int(total_steps * 0.05)

    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=TRAIN_EPOCHS,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=999_999_999,
        bf16=True,
        report_to="wandb",
        run_name=run_name,
        deepspeed=get_deepspeed_config(accumulation_steps, num_devices) if use_deepspeed else None,
        logging_steps=max(total_steps // 1000, 1),
        disable_tqdm=False,
        push_to_hub=push_to_hub,
        hub_model_id=f"Talking-Babies/{model_type}-{sanitized_dataset_name}",
        hub_strategy="every_save",
        learning_rate=5e-5 * (seq_len / 64),
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[CustomCheckpointingCallback(seq_len)],
    )

    # --- Stats --- #
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    box_width = 70
    print("\n" + "=" * box_width)
    print(f"{'📊 MODEL TRAINING CONFIGURATION 📊':^{box_width}}")
    print("=" * box_width)
    print(f"🤖 {'Model:':<25} {model_type.upper()}")
    print(f"📏 {'Sequence Length:':<25} {seq_len}")
    print(f"🧠 {'Total parameters:':<25} {total_params}")
    print(f"🔄 {'Trainable parameters:':<25} {trainable_params}")
    print("=" * box_width + "\n")

    # --- Train --- #
    start_time = time.time()
    trainer.train(resume_from_checkpoint=latest_checkpoint if latest_checkpoint else None)
    end_time = time.time()

    print(f"✅ Training {model_type.upper()} for seq_len {seq_len} done in {end_time - start_time:.2f}s")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="opt", choices=["opt", "mamba"])
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_devices", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--no_push_to_hub", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to load, e.g., 'Talking-Babies/train_100M_128_single_shuffle'."
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to resume from the latest checkpoint (default: True)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume from"
    )

    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        seq_len=args.seq_len,
        use_deepspeed=args.use_deepspeed,
        push_to_hub=not args.no_push_to_hub,
        dry_run=args.dry_run,
        num_devices=args.num_devices,
        accumulation_steps=args.accumulation_steps,
        dataset=args.dataset,
        resume=args.resume,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
