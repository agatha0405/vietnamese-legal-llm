"""Benchmark chatbot profiles on the same Qwen revision and regression questions."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter

from evaluate_baseline import ROOT, check_environment, load_qwen, score
from run_checkpoint import read_json, verify, write_json, writer
from train_qlora import FINAL_FILES

sys.path.insert(0, str(ROOT))
from app.inference import chat_config, generate, sample_gpu  # noqa: E402


def run(args):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    torch.set_num_threads(args.cpu_threads)
    check_environment()
    manifest = read_json(args.adapter / "complete.json")
    step = verify(args.adapter, manifest["identity"], FINAL_FILES)
    qwen = manifest["identity"]["qwen"]
    cases = read_json(args.cases)
    configs = {profile: chat_config(qwen, profile) for profile in args.profiles}
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    base, tokenizer, environment = load_qwen(qwen, args.offline)
    torch.cuda.synchronize()
    base_load = perf_counter() - started
    started = perf_counter()
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval()
    torch.cuda.synchronize()
    adapter_load = perf_counter() - started
    local_tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    assert local_tokenizer.chat_template == tokenizer.chat_template
    assert local_tokenizer.get_vocab() == tokenizer.get_vocab()
    eos = qwen["generation"]["eos_token_id"]
    assert tokenizer.eos_token_id in eos
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") in eos
    assert tokenizer.pad_token_id == qwen["generation"]["pad_token_id"]
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"Model loaded in {base_load:.2f}s, adapter in {adapter_load:.2f}s", flush=True)
    with writer(args.output):
        write_json(args.output / "environment.json", {
            "started_utc": started_utc, "environment": environment, "step": step,
            "cpu_threads": torch.get_num_threads(), "base_and_tokenizer_load_s": base_load,
            "adapter_load_s": adapter_load, "profile_configurations": configs,
            "tokenizer_eos_id": tokenizer.eos_token_id, "tokens": tokenizer.convert_ids_to_tokens(eos),
            "adapter_sha256": manifest["files"]["adapter_model.safetensors"],
        })
        for mode in ("base", "fine-tuned"):
            with model.disable_adapter() if mode == "base" else nullcontext():
                messages = make_messages(configs[args.profiles[0]], cases[0]["question"])
                generate(model, tokenizer, configs[args.profiles[0]], messages, warmup=True)
                for case in cases:
                    if args.ids and case["id"] not in args.ids:
                        continue
                    for profile, config in configs.items():
                        if profile == "prompt-only" and case["id"] != "red-unspecified":
                            continue
                        path = args.output / f"{case['id']}-{mode}-{profile}.json"
                        if path.exists():
                            raise FileExistsError(path)
                        done, samples = Event(), []
                        worker = Thread(target=probe, args=(done, samples), daemon=True)
                        worker.start()
                        try:
                            result = generate(model, tokenizer, config, make_messages(config, case["question"]))
                        finally:
                            done.set()
                            worker.join()
                        if not result.prediction:
                            raise RuntimeError(f"Empty output: {path.name}")
                        write_json(path, {
                            "case": case, "mode": mode, "profile": profile,
                            **asdict(result), **sample_gpu(samples),
                            "text_metrics": score(result.prediction, case["reference"]),
                        })
                        print(f"{path.stem}: {result.generated_tokens} tokens, {result.stop_reason}, "
                              f"{result.generate_s:.2f}s", flush=True)


def probe(done, samples):
    while not done.is_set():
        try:
            line = subprocess.check_output(
                ["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip()
            util, memory = map(int, line.split(","))
            samples.append((util, memory))
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        done.wait(1)


def make_messages(config, question):
    return [{"role": "system", "content": config["system_prompt"]}, {"role": "user", "content": question}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "tests" / "chatbot_cases.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("legacy", "prompt-only", "concise"),
                        default=("legacy", "prompt-only", "concise"))
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    run(args)


if __name__ == "__main__":
    main()
