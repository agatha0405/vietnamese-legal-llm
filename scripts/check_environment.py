"""Verify the pinned CUDA stack and one disposable QLoRA update."""

import argparse
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def check_environment(force_fp16):
    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        raise RuntimeError("Use Python 3.12 or 3.13.")

    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    versions = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        installed = version(requirement.name)
        if installed not in requirement.specifier:
            raise RuntimeError(f"{requirement}: found {installed}")
        versions[requirement.name] = installed

    import bitsandbytes as bnb
    import torch

    for module, api in (
        ("datasets", "Dataset"),
        ("gradio", "ChatInterface"),
        ("peft", "prepare_model_for_kbit_training"),
        ("transformers", "BitsAndBytesConfig"),
        ("trl", "SFTTrainer"),
    ):
        getattr(import_module(module), api)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Colab, connect to a GPU runtime.")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) < (6, 0):
        raise RuntimeError("bitsandbytes NF4 requires NVIDIA compute capability 6.0 or later.")
    dtype = torch.float16
    if not force_fp16 and torch.cuda.is_bf16_supported(including_emulation=False):
        dtype = torch.bfloat16
    layer = bnb.nn.Linear4bit(
        64, 16, bias=False, quant_type="nf4", compress_statistics=True, compute_dtype=dtype
    ).to("cuda:0")
    sample = torch.ones((2, 64), dtype=dtype, device="cuda:0", requires_grad=True)
    result = layer(sample)
    result.float().sum().backward()
    if result.shape != (2, 16) or not torch.isfinite(result).all():
        raise RuntimeError("NF4 forward returned invalid output.")
    if sample.grad is None or not torch.isfinite(sample.grad).all():
        raise RuntimeError("NF4 backward returned invalid gradients.")

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "pip_freeze": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"], text=True
        ).splitlines(),
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_mib": round(props.total_memory / 2**20, 1),
        "cuda_build": torch.version.cuda,
        "compute_dtype": str(dtype),
        "library_imports": "passed",
        "nf4_forward_backward": "passed",
    }


def check_qlora(model_spec, settings, environment):
    import bitsandbytes as bnb
    from datasets import Dataset
    import torch
    from peft import LoraConfig, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(settings["seed"])
    use_bf16 = environment["compute_dtype"] == "torch.bfloat16"
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    model_id = model_spec["id"]
    revision = model_spec["revision"]
    print(f"Loading {model_id} @ {revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        use_safetensors=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=settings["quant_type"],
            bnb_4bit_use_double_quant=settings["double_quant"],
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    quantized_layers = sum(isinstance(module, bnb.nn.Linear4bit) for module in base.modules())
    if not getattr(base, "is_loaded_in_4bit", False) or not quantized_layers:
        raise RuntimeError("Qwen did not load with bitsandbytes 4-bit linear layers.")

    # A non-legal prompt checks the tokenizer and generation path.
    prompt = [{"role": "user", "content": "Hãy chào bằng tiếng Việt."}]
    inputs = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda:0")
    with torch.inference_mode():
        generated = base.generate(**inputs, do_sample=False, max_new_tokens=settings["max_new_tokens"])
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    if new_tokens.numel() == 0:
        raise RuntimeError("Base generation produced zero new tokens.")
    print(f"Base generation passed ({new_tokens.numel()} new tokens).", flush=True)

    base.config.use_cache = False
    # A single in-memory fixture exercises conversational completion-only SFT.
    fixture = Dataset.from_list(
        [{"prompt": prompt, "completion": [{"role": "assistant", "content": "Xin chào!"}]}]
    )
    print("Running one SFT step and checking the adapter roundtrip.", flush=True)
    with TemporaryDirectory(prefix="qlora-smoke-") as temp:
        trainer = SFTTrainer(
            model=base,
            processing_class=tokenizer,
            train_dataset=fixture,
            # TRL prepares the k-bit base and attaches the trainable adapter in that order.
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                r=settings["rank"],
                lora_alpha=settings["alpha"],
                target_modules="all-linear",
                bias="none",
            ),
            args=SFTConfig(
                output_dir=temp,
                max_length=settings["max_length"],
                max_steps=1,
                per_device_train_batch_size=1,
                learning_rate=settings["learning_rate"],
                optim="adamw_torch",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                completion_only_loss=True,
                packing=False,
                bf16=use_bf16,
                fp16=not use_bf16,
                logging_strategy="no",
                save_strategy="no",
                report_to="none",
                disable_tqdm=True,
                seed=settings["seed"],
            ),
        )
        model = trainer.model
        scaler = trainer.accelerator.scaler
        if scaler is not None:
            # A single-step check has no warm-up for the default dynamic loss scale.
            scaler_state = scaler.state_dict()
            scaler_state["scale"] = settings["fp16_initial_scale"]
            scaler.load_state_dict(scaler_state)

        trainable, total = model.get_nb_trainable_parameters()
        if not 0 < trainable < total or any(
            parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()
        ):
            raise RuntimeError("Only LoRA parameters should be trainable.")
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in get_peft_model_state_dict(model, save_embedding_layers=False).items()
        }
        batch = next(iter(trainer.get_train_dataloader()))
        mask = trainer.train_dataset[0]["completion_mask"]
        labels = batch["labels"][0].tolist()
        ids = batch["input_ids"][0].tolist()
        if 0 not in mask or 1 not in mask or labels != [
            token if keep else -100 for token, keep in zip(ids, mask, strict=True)
        ]:
            raise RuntimeError("The SFT batch must mask the prompt and supervise the completion.")
        train_output = trainer.train()
        if trainer.accelerator.optimizer_step_was_skipped:
            raise RuntimeError("Mixed precision skipped the optimizer step.")
        if train_output.global_step != 1 or not math.isfinite(train_output.training_loss):
            raise RuntimeError("SFT did not complete a finite step.")
        after = get_peft_model_state_dict(model, save_embedding_layers=False)
        if not all(torch.isfinite(weight).all() for weight in after.values()):
            raise RuntimeError("Adapter has non-finite weights.")
        if not any(not torch.equal(before[name], weight.cpu()) for name, weight in after.items()):
            raise RuntimeError("No adapter parameter changed.")

        adapter_dir = Path(temp) / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
        model.load_adapter(adapter_dir, adapter_name="roundtrip", is_trainable=False)
        restored = get_peft_model_state_dict(
            model, adapter_name="roundtrip", save_embedding_layers=False
        )
        if after.keys() != restored.keys() or not all(
            torch.equal(weight, restored[name]) for name, weight in after.items()
        ):
            raise RuntimeError("The reloaded adapter differs from the saved adapter.")

    return {
        "model": model_spec,
        "settings": settings,
        "quantized_linear_layers": quantized_layers,
        "generation_new_tokens": new_tokens.numel(),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "optimizer_steps": train_output.global_step,
        "smoke_training_loss": train_output.training_loss,
        "adapter_roundtrip": "passed",
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("environment", "qlora"))
    parser.add_argument("--model", choices=("primary", "fallback"), default="primary")
    parser.add_argument("--fp16", action="store_true", help="Test the T4 precision path on a newer GPU.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists. Choose a new report path.")

    started = datetime.now(timezone.utc)
    config = tomllib.loads((ROOT / "configs/environment.toml").read_text(encoding="utf-8"))
    report = {
        "mode": args.mode,
        "started_utc": started.isoformat(),
        "environment": check_environment(args.fp16),
    }
    if args.mode == "qlora":
        report["qlora"] = check_qlora(config["models"][args.model], config["smoke"], report["environment"])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(f"PASS: {args.mode}. Evidence saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
