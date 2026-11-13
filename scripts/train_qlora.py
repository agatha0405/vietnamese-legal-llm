"""Train and resume Qwen QLoRA SFT on the frozen dataset splits."""

import argparse
from importlib.metadata import version
import math
import os
from pathlib import Path
import re
from time import perf_counter
import tomllib
from uuid import uuid4

from evaluate_baseline import ROOT, check_environment, digest, load_qwen, load_test, read_jsonl
from run_checkpoint import archive, publish, read_json, verify, write_json, writer


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
CHECKPOINT_FILES = (*ADAPTER_FILES, "trainer_state.json", "optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth")
FINAL_FILES = (*ADAPTER_FILES, "trainer_state.json", "training_metrics.json")


def training_data(settings, tokenizer):
    from datasets import Dataset
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    datasets = []
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, completion_only_loss=True)
    for split in ("train", "validation"):
        encoded = []
        for row in read_jsonl(ROOT / "data/processed" / f"{split}.jsonl"):
            prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            ids = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
            eos_index = ids.index(tokenizer.eos_token_id, len(prompt))
            mask = [int(len(prompt) <= index <= eos_index) for index in range(len(ids))]
            if (
                len(ids) > settings["max_length"] or ids[:len(prompt)] != prompt
                or ids != row["input_ids"] or mask != row["completion_mask"] or sum(mask) < 2
            ):
                raise ValueError(f"Invalid assistant completion boundary: {row['id']}")
            encoded.append({"input_ids": ids, "completion_mask": mask})
        example = collator([encoded[0], encoded[-1]])
        for labels, record in zip(example["labels"].tolist(), (encoded[0], encoded[-1]), strict=True):
            expected = [token if keep else -100 for token, keep in zip(record["input_ids"], record["completion_mask"], strict=True)]
            if labels != expected + [-100] * (len(labels) - len(expected)):
                raise ValueError("TRL collator did not mask prompt or padding")
        print(f"{split}: {len(encoded)} rows, longest {max(len(row['input_ids']) for row in encoded)} tokens", flush=True)
        datasets.append(Dataset.from_list(encoded))
    return datasets


def scan_checkpoints(run, identity):
    valid, invalid = [], []
    for path in run.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match is None or not path.is_dir():
            continue
        try:
            step = verify(path, identity, CHECKPOINT_FILES)
            if step != int(match[1]) or read_json(path / "trainer_state.json")["global_step"] != step:
                raise ValueError("Checkpoint step mismatch")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"Ignoring {path.name}: {exc}", flush=True)
            invalid.append(path)
        else:
            valid.append((step, path))
    return sorted(valid), invalid


def archived_checkpoint(run, identity):
    candidates = sorted(
        (path for path in (run / "history").glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+-[0-9a-f]+", path.name)),
        key=lambda path: int(path.name.split("-")[1]), reverse=True,
    )
    for path in candidates:
        try:
            step = verify(path, identity, CHECKPOINT_FILES)
            if step == int(path.name.split("-")[1]) == read_json(path / "trainer_state.json")["global_step"]:
                return step, path
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def keep_active_checkpoints(run, valid, limit):
    for _, previous in valid[:-limit]:
        archive(run, previous)


def selected_checkpoint(run, identity):
    valid, invalid = scan_checkpoints(run, identity)
    latest = valid[-1] if valid else archived_checkpoint(run, identity)
    return latest, valid, invalid


def train(run, identity, qwen, datasets, offline, checkpoint, pause_at):
    import torch
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    settings = identity["training"]
    adapter = identity["adapter"]
    base, tokenizer, environment = load_qwen(qwen, offline)
    base.config.use_cache = False
    write_json(run / f"attempt-{uuid4().hex}.json", {
        "environment": environment, "resume": checkpoint.name if checkpoint else None,
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    })

    class PersistentTrainer(SFTTrainer):
        def _get_output_dir(self, trial):
            return str(getattr(self, "checkpoint_staging", None) or super()._get_output_dir(trial))

        def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
            # Publish and verify before applying retention.
            return

        def _save_checkpoint(self, model, trial):
            step = self.state.global_step
            staging = run / f".checkpoint-{step}-{uuid4().hex}"
            name = f"checkpoint-{step}"
            self.checkpoint_staging = staging
            try:
                super()._save_checkpoint(model, trial)
            finally:
                self.checkpoint_staging = None
            publish(staging / name, run / name, identity, step, CHECKPOINT_FILES)
            staging.rmdir()
            active, _ = scan_checkpoints(run, identity)
            keep_active_checkpoints(run, active, settings["save_total_limit"])
            print(f"Checkpoint verified at step {step}: {run / name}", flush=True)

    class PauseAtCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if pause_at == state.global_step:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            control.should_training_stop = False
            return control

    trainer = PersistentTrainer(
        model=base,
        processing_class=tokenizer,
        train_dataset=datasets[0],
        eval_dataset=datasets[1],
        peft_config=LoraConfig(
            task_type="CAUSAL_LM", r=adapter["rank"], lora_alpha=adapter["alpha"],
            lora_dropout=adapter["lora_dropout"], target_modules=adapter["target_modules"], bias="none",
        ),
        args=SFTConfig(
            output_dir=str(run), **settings, fp16=True, bf16=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_kwargs={"skip_prepare_dataset": True}, completion_only_loss=True, packing=False,
            save_only_model=False, report_to="none", disable_tqdm=True, logging_nan_inf_filter=False,
            prediction_loss_only=True,
        ),
        callbacks=[PauseAtCheckpoint()],
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    if not 0 < trainable < total or any(
        parameter.requires_grad and "lora_" not in name for name, parameter in trainer.model.named_parameters()
    ):
        raise RuntimeError("Expected only LoRA parameters to be trainable")
    scaler = trainer.accelerator.scaler
    if scaler is None:
        raise RuntimeError("Expected an FP16 gradient scaler")
    scaler.load_state_dict({**scaler.state_dict(), "scale": 256.0})
    print(f"Trainable {trainable:,}/{total:,}. Resume: {checkpoint or 'fresh'}", flush=True)
    start_step = read_json(checkpoint / "trainer_state.json")["global_step"] if checkpoint else 0
    started = perf_counter()
    outcome = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    elapsed = perf_counter() - started
    step = trainer.state.global_step
    write_json(run / f"training_segment-{uuid4().hex}.json", {
        "start_step": start_step,
        "end_step": step, "trainer_metrics": outcome.metrics, "elapsed_s": elapsed,
        "gpu": environment["gpu"], "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    })
    if step != trainer.state.max_steps:
        if verify(run / f"checkpoint-{step}", identity, CHECKPOINT_FILES) != step:
            raise RuntimeError("Paused run did not publish its checkpoint")
        print(f"Paused at step {step}/{trainer.state.max_steps}. Rerun without --pause-at-step to continue.")
        return

    if verify(run / f"checkpoint-{step}", identity, CHECKPOINT_FILES) != step:
        raise RuntimeError("Final step checkpoint is missing")
    validation = trainer.evaluate()
    if not math.isfinite(validation["eval_loss"]):
        raise ValueError("Non-finite final validation loss")
    history = [row for row in trainer.state.log_history if "loss" in row]
    if not history or any(not math.isfinite(row["loss"]) for row in history):
        raise ValueError("Training loss is missing or non-finite")
    staged = run / f".final-{uuid4().hex}"
    trainer.save_model(str(staged))
    trainer.state.save_to_json(staged / "trainer_state.json")
    write_json(staged / "training_metrics.json", {
        "global_step": step, "validation": validation, "trainable_parameters": trainable,
        "total_parameters": total, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "last_step_loss": history[-1]["loss"],
        "last_training_segment_s": elapsed,
        "duration_scope": "Last invocation of trainer.train, including its scheduled validation and checkpoint saves",
    })
    publish(staged, run / "final_adapter", identity, step, FINAL_FILES)
    print(f"Final adapter verified at step {step}: {run / 'final_adapter'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(8, 16), default=8)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--pause-at-step", type=int)
    parser.add_argument("--fallback", action="store_true", help="Use the pinned 1.5B fallback in a separate run directory")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = tomllib.loads((ROOT / "configs/training.toml").read_text(encoding="utf-8"))
    qwen = tomllib.loads((ROOT / "configs/inference.toml").read_text(encoding="utf-8"))
    if args.fallback:
        fallback = tomllib.loads((ROOT / "configs/environment.toml").read_text(encoding="utf-8"))["models"]["fallback"]
        qwen.update(model_id=fallback["id"], model_revision=fallback["revision"], tokenizer_revision=fallback["revision"])
    identity = {
        "schema": 1, "qwen": qwen, "training": config["training"],
        "adapter": {**config["adapter"], "rank": args.rank, "alpha": 2 * args.rank},
    }
    final = run / "final_adapter"
    manifest = run / "run_config.json"
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError("Run configuration differs. Select the original options or use a separate run directory.")
    if final.exists():
        print(f"Complete: {final} at step {verify(final, identity, FINAL_FILES)}")
        return
    if args.status:
        latest, _, invalid = selected_checkpoint(run, identity)
        print(f"Last valid checkpoint: {latest or 'none'}; invalid: {len(invalid)}; writer lock: {(run / '.writer').exists()}")
        return

    from transformers import AutoTokenizer

    load_test(qwen, ROOT / "data/processed")
    check_environment()
    for name, pinned in (("peft", "0.17.1"), ("trl", "0.22.2"), ("datasets", "4.1.1")):
        if version(name) != pinned:
            raise RuntimeError(f"Expected {name}=={pinned}")
    tokenizer = AutoTokenizer.from_pretrained(qwen["model_id"], revision=qwen["tokenizer_revision"], local_files_only=args.offline)
    if args.fallback:
        # Validate stored boundaries against the original tokenizer before using the fallback.
        primary = tomllib.loads((ROOT / "configs/inference.toml").read_text(encoding="utf-8"))
        original = AutoTokenizer.from_pretrained(primary["model_id"], revision=primary["tokenizer_revision"], local_files_only=args.offline)
        if tokenizer.chat_template != original.chat_template or tokenizer.get_vocab() != original.get_vocab():
            raise ValueError("Fallback tokenizer differs from the dataset tokenizer")
    datasets = training_data(config["training"], tokenizer)
    if args.check_only:
        print(f"PASS: frozen files, official tokenizer and completion masking ({digest(ROOT / 'configs/training.toml')})")
        return

    with writer(run):
        if manifest.exists() and read_json(manifest) != identity:
            raise ValueError("Run configuration changed")
        if final.exists():
            print(f"Complete: {final} at step {verify(final, identity, FINAL_FILES)}")
            return
        if not manifest.exists():
            write_json(manifest, identity)
        latest, valid, invalid = selected_checkpoint(run, identity)
        for path in invalid:
            archive(run, path)
        keep_active_checkpoints(run, valid, config["training"]["save_total_limit"])
        if latest and not valid:
            restored = run / f"checkpoint-{latest[0]}"
            latest[1].rename(restored)
            latest = latest[0], restored
        if args.pause_at_step is not None and args.pause_at_step <= (latest[0] if latest else 0):
            raise ValueError("Pause step must be later than the latest checkpoint")
        train(run, identity, qwen, datasets, args.offline, latest[1] if latest else None, args.pause_at_step)


if __name__ == "__main__":
    main()
