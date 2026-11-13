"""Run and score the frozen test set with the pretrained Qwen model."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
from statistics import mean
from time import perf_counter
import tomllib
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("rouge_l", "token_f1", "exact_match", "inference_time_s")
REVIEW_FIELDS = (
    "legal_factual_correctness", "citation_correctness", "hallucination",
    "appropriate_refusal", "reviewer", "evidence", "notes",
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized_tokens(value):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", value.casefold()))


def load_test(config, data_dir):
    for filename, expected_hash in config["files"].items():
        if digest(data_dir / filename) != expected_hash:
            raise ValueError(f"Dataset checksum mismatch: {filename}")
    master = read_jsonl(data_dir / "legal_qa_master.jsonl")
    by_id = {row["id"]: row for row in master}
    if len(by_id) != len(master):
        raise ValueError("Duplicate IDs in master")
    seen_ids = set()
    group_owners = {}
    question_owners = {}
    evidence_owners = {}
    test = []
    for split in ("train", "validation", "test"):
        for record in read_jsonl(data_dir / f"{split}.jsonl"):
            row = by_id[record["id"]]
            if row["id"] in seen_ids or row["split"] != split or row["verification_status"] != "verified":
                raise ValueError(f"Split ID or verification mismatch: {row['id']}")
            seen_ids.add(row["id"])
            expected_prompt = [
                {"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": row["question"]},
            ]
            if record["prompt"] != expected_prompt or record["completion"] != [
                {"role": "assistant", "content": row["answer"]}
            ]:
                raise ValueError(f"Prompt or reference mismatch: {row['id']}")
            for mapping, key in (
                (group_owners, row["group_id"]),
                (question_owners, tuple(normalized_tokens(row["question"]))),
                (evidence_owners, tuple(normalized_tokens(row["source_text"]))),
            ):
                if not key or mapping.setdefault(key, split) != split:
                    raise ValueError(f"Cross-split overlap: {row['id']}")
            if split == "test":
                test.append(row)
    if seen_ids != set(by_id) or len(test) != config["test_count"]:
        raise ValueError("Incomplete or unexpected evaluation set")
    return test


def score(prediction, reference):
    predicted = normalized_tokens(prediction)
    expected = normalized_tokens(reference)
    if not predicted or not expected:
        return {"rouge_l": 0.0, "token_f1": 0.0, "exact_match": 0}
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for index, other in enumerate(expected, 1):
            current.append(previous[index - 1] + 1 if token == other else max(previous[index], current[-1]))
        previous = current
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    length = len(predicted) + len(expected)
    return {
        "rouge_l": 2 * previous[-1] / length,
        "token_f1": 2 * overlap / length,
        "exact_match": int(predicted == expected),
    }


def load_qwen(config, offline):
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Select a Colab GPU runtime.")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["tokenizer_revision"],
        local_files_only=offline, trust_remote_code=config["trust_remote_code"],
    )
    if not tokenizer.chat_template or tokenizer.pad_token_id != config["generation"]["pad_token_id"]:
        raise ValueError("Unexpected Qwen chat template or pad token")
    quantization = dict(config["quantization"])
    dtype = getattr(torch, quantization["bnb_4bit_compute_dtype"])
    quantization["bnb_4bit_compute_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"],
        local_files_only=offline, trust_remote_code=config["trust_remote_code"],
        use_safetensors=config["use_safetensors"], device_map={"": 0}, dtype=dtype,
        attn_implementation=config["attention"], quantization_config=BitsAndBytesConfig(**quantization),
    ).eval()
    quantized_layers = sum(isinstance(layer, bnb.nn.Linear4bit) for layer in model.modules())
    if not model.is_loaded_in_4bit or quantized_layers != 7 * model.config.num_hidden_layers or model.config._commit_hash != config["model_revision"]:
        raise RuntimeError("Unexpected Qwen model revision or quantization")
    props = torch.cuda.get_device_properties(0)
    return model, tokenizer, {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "bitsandbytes", "accelerate", "huggingface-hub", "tokenizers", "safetensors", "matplotlib")},
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_mib": props.total_memory / 2**20,
        "cuda_build": torch.version.cuda,
        "quantized_layers": quantized_layers,
        "chat_template_sha256": sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        **({"cuda_allocator_config": os.environ["PYTORCH_CUDA_ALLOC_CONF"]} if "PYTORCH_CUDA_ALLOC_CONF" in os.environ else {}),
    }


def generate_answer(model, tokenizer, config, question, *, warmup=False):
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": question},
    ]
    return generate_messages(model, tokenizer, config, messages, warmup=warmup)


def generate_messages(model, tokenizer, config, messages, *, warmup=False):
    import torch
    from transformers import GenerationConfig

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda:0")
    options = dict(config["generation"])
    if warmup:
        options["max_new_tokens"] = 8
    torch.cuda.synchronize()
    started = perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=GenerationConfig(**options))
    torch.cuda.synchronize()
    duration = perf_counter() - started
    tokens = output[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return answer, duration, tokens.numel()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        read_back = list(csv.DictReader(handle))
    expected = [{key: str(value) for key, value in row.items()} for row in rows]
    if read_back != expected:
        raise ValueError(f"CSV roundtrip failed: {path}")


def make_chart(path, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained", width_ratios=[2.4, 1])
    scores = [metrics[key] for key in METRICS[:3]]
    bars = axes[0].bar(["ROUGE-L", "Token F1", "Exact Match"], scores, color=["#236484", "#32947f", "#bc742a"])
    axes[0].bar_label(bars, fmt="%.3f", padding=4)
    axes[0].set(ylim=(0, 1.07), ylabel="Điểm trung bình", title="Độ khớp văn bản")
    latency = metrics["inference_time_s"]
    bars = axes[1].bar(["Qwen 3B"], [latency], color="#236484", width=0.5)
    axes[1].bar_label(bars, fmt="%.2f", padding=4)
    axes[1].set(ylim=(0, latency * 1.18), ylabel="Giây mỗi câu", title="Thời gian sinh")
    fig.suptitle(f"Baseline · {metrics['sample_count']} câu hỏi")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(rows):
    return {"sample_count": len(rows), **{key: mean(row[key] for row in rows) for key in METRICS}}


def finalize(output, test, predictions, manifest):
    if [row["id"] for row in test] != [row["id"] for row in predictions]:
        raise ValueError("Prediction IDs or order differ from the frozen test set")
    rows = []
    for row, result in zip(test, predictions, strict=True):
        if not result["prediction"].strip() or not math.isfinite(result["inference_time_s"]) or result["inference_time_s"] <= 0:
            raise ValueError(f"Invalid prediction or time: {row['id']}")
        if not 1 <= result["generated_tokens"] <= manifest["config"]["generation"]["max_new_tokens"]:
            raise ValueError(f"Invalid generation length: {row['id']}")
        scores = score(result["prediction"], row["answer"])
        if not all(0 <= value <= 1 for value in scores.values()):
            raise ValueError(f"Invalid metrics: {row['id']}")
        rows.append({
            "id": row["id"], "question": row["question"], "reference_answer": row["answer"],
            "base_model_prediction": result["prediction"], "source_url": row["source_url"],
            "source_pdf_url": row["source_pdf_url"], "document_id": row["document_id"],
            "article": row["article"], "clause": row["clause"] or "", "topic": row["topic"],
            "question_type": row["question_type"], **scores,
            "inference_time_s": result["inference_time_s"], "generated_tokens": result["generated_tokens"],
            "legal_review_status": "unverified",
        })
    metrics = {
        **summarize(rows),
        "by_question_type": {
            kind: summarize([row for row in rows if row["question_type"] == kind])
            for kind in sorted({row["question_type"] for row in rows})
        },
        "at_generation_limit": sum(row["generated_tokens"] == manifest["config"]["generation"]["max_new_tokens"] for row in rows),
        "legal_review": {"status": "unverified", "reviewed_samples": 0},
        "run": manifest,
    }
    baseline = output / "baseline"
    write_csv(baseline / "baseline_predictions.csv", rows)
    write_json(baseline / "baseline_metrics.json", metrics)
    if json.loads((baseline / "baseline_metrics.json").read_text(encoding="utf-8")) != metrics:
        raise ValueError("Metrics JSON roundtrip failed")
    rubric = baseline / "legal_review_sheet.csv"
    if not rubric.exists():
        write_csv(rubric, [
            {
                **{key: row[key] for key in ("id", "question", "base_model_prediction", "reference_answer", "source_pdf_url")},
                "review_status": "unverified", **dict.fromkeys(REVIEW_FIELDS, ""),
            }
            for row in rows
        ])
    make_chart(output / "figures" / "baseline_metrics.png", metrics)
    examples_path = baseline / "baseline_examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8")) if examples_path.exists() else []
    (baseline / "baseline_report.md").write_text(make_report(metrics, rows, examples), encoding="utf-8")
    return metrics


def make_report(metrics, rows, examples):
    run = metrics["run"]
    groups = metrics["by_question_type"]
    report = (
        "# Baseline Qwen2.5-3B-Instruct\n\n"
        f"Đã sinh câu trả lời cho {metrics['sample_count']} câu hỏi trên {run['environment']['gpu']} "
        "với NF4 FP16, greedy decoding và giới hạn 256 token mới.\n\n"
        "| Chỉ số | Giá trị trung bình |\n| --- | ---: |\n"
        f"| ROUGE-L | {metrics['rouge_l']:.6f} |\n"
        f"| Token F1 | {metrics['token_f1']:.6f} |\n"
        f"| Exact Match | {metrics['exact_match']:.6f} |\n"
        f"| Thời gian sinh (giây/câu) | {metrics['inference_time_s']:.3f} |\n\n"
        "| Loại câu hỏi | Số mẫu | ROUGE-L | Token F1 | Exact Match | Giây/câu |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"| Viết riêng | {groups['authored']['sample_count']} | {groups['authored']['rouge_l']:.4f} | "
        f"{groups['authored']['token_f1']:.4f} | {groups['authored']['exact_match']:.4f} | "
        f"{groups['authored']['inference_time_s']:.2f} |\n"
        f"| Tra cứu | {groups['provision_lookup']['sample_count']} | {groups['provision_lookup']['rouge_l']:.4f} | "
        f"{groups['provision_lookup']['token_f1']:.4f} | {groups['provision_lookup']['exact_match']:.4f} | "
        f"{groups['provision_lookup']['inference_time_s']:.2f} |\n\n"
        f"Có {metrics['at_generation_limit']} câu chạm giới hạn sinh. "
        "Các điểm tự động đo độ khớp với câu tham chiếu. "
        "Đánh giá pháp lý độc lập đang ở trạng thái `unverified`.\n\n"
        "Xem [rubric và quy trình](../../docs/baseline_evaluation.md) và [phiếu chấm](legal_review_sheet.csv).\n"
    )
    if not examples:
        return report
    by_id = {row["id"]: row for row in rows}
    sections = [report, "\n## Sáu ví dụ đối chiếu với căn cứ\n"]
    for example in examples:
        row = by_id[example["id"]]
        sections.append(
            f"\n### {example['category']} · {row['id']}\n\n"
            f"**Câu hỏi**  \n{row['question']}\n\n"
            f"**Nhận xét sơ bộ**  \n{example['note']}\n\n"
            f"[Trang PDF của căn cứ]({row['source_pdf_url']}) · [Toàn bộ dự đoán](baseline_predictions.csv)\n"
        )
    return "".join(sections)


def check_environment():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line:
            name, expected = line.split("==")
            pins[name] = expected
    for name in ("torch", "transformers", "bitsandbytes", "accelerate", "huggingface-hub", "tokenizers", "safetensors", "matplotlib"):
        if version(name) != pins[name]:
            raise RuntimeError(f"Expected {name}=={pins[name]}, found {version(name)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    config = tomllib.loads((ROOT / "configs/inference.toml").read_text(encoding="utf-8"))
    test = load_test(config, ROOT / "data/processed")
    check_environment()
    if args.check_only:
        assert score("Đường, bộ!", "ĐƯỜNG bộ") == {"rouge_l": 1.0, "token_f1": 1.0, "exact_match": 1}
        assert score("b a a", "a a b") == {"rouge_l": 2 / 3, "token_f1": 1.0, "exact_match": 0}
        assert score("đỗ", "do") == {"rouge_l": 0.0, "token_f1": 0.0, "exact_match": 0}
        print(f"PASS: frozen test, split audit, pinned packages and metric checks ({len(test)} rows)")
        return

    baseline = args.output / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    manifest_path = baseline / "inference_config.json"
    progress_path = baseline / "baseline_generations.jsonl"
    metrics_path = baseline / "baseline_metrics.json"
    complete_outputs = (
        baseline / "baseline_predictions.csv", baseline / "baseline_report.md",
        baseline / "legal_review_sheet.csv", args.output / "figures/baseline_metrics.png",
    )
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        if saved["run"]["config"] != config or saved["sample_count"] != len(test):
            raise ValueError("Saved baseline configuration differs. Use a new --output directory.")
        if not all(path.is_file() for path in complete_outputs):
            raise ValueError("Saved baseline has missing files. Use a new --output directory for reconstruction.")
        print(f"Preserving completed baseline: {metrics_path}", flush=True)
        return
    predictions = read_jsonl(progress_path) if progress_path.exists() else []
    if [row["id"] for row in predictions] != [row["id"] for row in test[:len(predictions)]]:
        raise ValueError("Saved generations are not an ordered test prefix")
    if len(predictions) == len(test):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("Saved run configuration differs from the current configuration")
    else:
        print("Loading pinned Qwen 3B (NF4 FP16)", flush=True)
        model, tokenizer, environment = load_qwen(config, args.offline)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["config"] != config or manifest["environment"] != environment:
                raise ValueError("Resume requires the same configuration and environment")
        else:
            manifest = {
                "config": config, "environment": environment,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(manifest_path, manifest)
        generate_answer(model, tokenizer, config, "Hãy chào bằng tiếng Việt.", warmup=True)
        for index, row in enumerate(test[len(predictions):], len(predictions) + 1):
            answer, seconds, count = generate_answer(model, tokenizer, config, row["question"])
            if not answer:
                raise ValueError(f"Empty prediction: {row['id']}")
            result = {"id": row["id"], "prediction": answer, "inference_time_s": seconds, "generated_tokens": count}
            with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            predictions.append(result)
            print(f"[{index}/{len(test)}] {row['id']} {count} tokens {seconds:.2f}s", flush=True)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    result = finalize(args.output, test, predictions, manifest)
    print(json.dumps({k: result[k] for k in ("sample_count", *METRICS)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
