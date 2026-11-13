"""Evaluate the final fine-tuned adapter against the saved baseline."""

import argparse
from collections import Counter
import math
from pathlib import Path
from uuid import uuid4

from evaluate_baseline import ROOT, METRICS, check_environment, digest, generate_answer, load_qwen, load_test, score, summarize, write_csv
from run_checkpoint import archive, publish, read_json, verify, write_json, writer
from summarize_training import FILES as TRAINING_SUMMARY_FILES
from train_qlora import FINAL_FILES


RESULT_FILES = ("metrics.json", "legal_review_sheet.csv", "legal_review_key.csv")
COMPARISON_FILES = ("comparison_predictions.csv", "comparison_metrics.csv", "error_analysis.csv", "comparison_report.md", "comparison_metrics.png", "training_loss.png")
BASELINE_CSV = ROOT / "outputs/baseline/baseline_predictions.csv"


def evaluation_input(adapter):
    manifest = read_json(adapter / "complete.json")
    verify(adapter, manifest["identity"], FINAL_FILES)
    qwen = manifest["identity"]["qwen"]
    baseline = read_json(ROOT / "outputs/baseline/baseline_metrics.json")
    if qwen != baseline["run"]["config"]:
        raise ValueError("Adapter evaluation requires the same model, tokenizer, prompts and settings as the baseline")
    identity = {"adapter": manifest, "baseline": baseline, "baseline_csv_sha256": digest(BASELINE_CSV), "schema": 1}
    return qwen, baseline, identity


def predict(test, run, adapter, qwen, baseline, offline):
    from peft import PeftModel

    model = tokenizer = None
    environment_file = run / "inference_environment.json"
    for index, row in enumerate(test, 1):
        path = run / f"prediction-{index:03}.json"
        result = None
        if path.exists():
            try:
                saved = read_json(path)
                if (
                    saved["id"] != row["id"] or not saved["prediction"].strip()
                    or not math.isfinite(saved["inference_time_s"]) or saved["inference_time_s"] <= 0
                    or not 1 <= saved["generated_tokens"] <= qwen["generation"]["max_new_tokens"]
                ):
                    raise ValueError("Invalid row")
            except (OSError, ValueError, AttributeError, TypeError, KeyError):
                archive(run, path)
            else:
                result = saved
        if result is None:
            if model is None:
                base, tokenizer, environment = load_qwen(qwen, offline)
                if environment["chat_template_sha256"] != baseline["run"]["environment"]["chat_template_sha256"]:
                    raise RuntimeError("Qwen chat template differs from baseline")
                if environment_file.exists():
                    if read_json(environment_file) != environment:
                        raise ValueError("Resume requires the same inference environment")
                else:
                    write_json(environment_file, environment)
                model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
                generate_answer(model, tokenizer, qwen, "Hãy chào bằng tiếng Việt.", warmup=True)
            answer, seconds, count = generate_answer(model, tokenizer, qwen, row["question"])
            if not answer or not math.isfinite(seconds) or seconds <= 0 or not 1 <= count <= qwen["generation"]["max_new_tokens"]:
                raise ValueError(f"Invalid generation: {row['id']}")
            result = {"id": row["id"], "prediction": answer, "inference_time_s": seconds, "generated_tokens": count}
            write_json(path, result)
        print(f"[{index}/{len(test)}] {row['id']}", flush=True)
        yield {
            "id": row["id"], "question": row["question"], "reference_answer": row["answer"],
            "prediction": result["prediction"], "source_pdf_url": row["source_pdf_url"],
            "question_type": row["question_type"], "topic": row["topic"],
            **score(result["prediction"], row["answer"]),
            "inference_time_s": result["inference_time_s"], "generated_tokens": result["generated_tokens"],
        }


def report_results(run, rows, baseline, identity, training_curve):
    import csv
    import random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shutil import copyfile

    adapter_scores = summarize(rows)
    base_env = baseline["run"]["environment"]
    adapter_env = read_json(run / "inference_environment.json")
    same_environment = adapter_env == base_env and not adapter_env.get("cuda_allocator_config")
    with BASELINE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    if [row["id"] for row in base_rows] != [row["id"] for row in rows]:
        raise ValueError("Baseline CSV does not match adapter evaluation IDs")
    delta = {key: adapter_scores[key] - baseline[key] for key in METRICS[:3]}
    delta["inference_time_s"] = adapter_scores["inference_time_s"] - baseline["inference_time_s"] if same_environment else None
    by_question_type = {
        kind: summarize([row for row in rows if row["question_type"] == kind])
        for kind in sorted({row["question_type"] for row in rows})
    }
    limit_hits = sum(row["generated_tokens"] == identity["adapter"]["identity"]["qwen"]["generation"]["max_new_tokens"] for row in rows)
    staged = run / f".results-{uuid4().hex}"
    staged.mkdir()
    paired = []
    for base, adapted in zip(base_rows, rows, strict=True):
        paired.append({
            **{key: adapted[key] for key in ("id", "question", "reference_answer", "source_pdf_url", "topic", "question_type")},
            "base_prediction": base["base_model_prediction"], "adapter_prediction": adapted["prediction"],
            **{f"base_{key}": float(base[key]) for key in METRICS},
            **{f"adapter_{key}": adapted[key] for key in METRICS},
            **{f"delta_{key}": adapted[key] - float(base[key]) for key in METRICS[:3]},
            "base_generated_tokens": int(base["generated_tokens"]),
            "adapter_generated_tokens": adapted["generated_tokens"],
        })
    write_csv(staged / "comparison_predictions.csv", paired)
    write_csv(staged / "comparison_metrics.csv", [
        {"metric": key, "base": baseline[key], "fine_tuned": adapter_scores[key], "difference": delta[key] if delta[key] is not None else ""}
        for key in METRICS
    ])
    write_json(staged / "metrics.json", {
        "base": {key: baseline[key] for key in ("sample_count", *METRICS)},
        "adapter": adapter_scores, "delta": delta, "adapter_environment": adapter_env,
        "by_question_type": by_question_type, "adapter_at_generation_limit": limit_hits,
        "same_inference_environment": same_environment, "legal_review_status": "unverified",
    })

    review_rows, key_rows = [], []
    rng = random.Random(42)
    for base, adapted in zip(base_rows, rows, strict=True):
        pair = [("base", base["base_model_prediction"]), ("adapter", adapted["prediction"])]
        rng.shuffle(pair)
        for suffix, (source, answer) in zip(("A", "B"), pair, strict=True):
            review_id = f"{adapted['id']}-{suffix}"
            review_rows.append({
                "review_id": review_id, "question": adapted["question"], "prediction": answer,
                "reference_answer": adapted["reference_answer"], "source_pdf_url": adapted["source_pdf_url"],
                "review_status": "unverified", "legal_factual_correctness": "", "citation_correctness": "",
                "hallucination": "", "appropriate_refusal": "", "reviewer": "", "evidence": "",
            })
            key_rows.append({"review_id": review_id, "model": source})
    rng.shuffle(review_rows)
    write_csv(staged / "legal_review_sheet.csv", review_rows)
    write_csv(staged / "legal_review_key.csv", key_rows)
    metrics = METRICS[:3]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    x = range(len(metrics))
    ax.bar([index - 0.19 for index in x], [baseline[key] for key in metrics], width=0.38, label="Base")
    ax.bar([index + 0.19 for index in x], [adapter_scores[key] for key in metrics], width=0.38, label="QLoRA")
    ax.set(xticks=list(x), xticklabels=["ROUGE-L", "Token F1", "Exact Match"], ylim=(0, 1), ylabel="Điểm trung bình")
    ax.legend()
    fig.savefig(staged / "comparison_metrics.png", dpi=160)
    plt.close(fig)
    copyfile(training_curve, staged / "training_loss.png")
    training_state = identity["training_validation"]
    trend = ", ".join(f"{row['step']} → {row['eval_loss']:.4f}" for row in training_state)
    lowest = min(training_state, key=lambda row: row["eval_loss"])
    table = "\n".join(
        f"| {key} | {baseline[key]:.6f} | {adapter_scores[key]:.6f} | "
        f"{format(delta[key], '+.6f') if delta[key] is not None else '—'} |"
        for key in METRICS
    )
    type_table = "\n".join(
        f"| {kind} | {values['sample_count']} | {baseline['by_question_type'][kind]['token_f1']:.4f} | {values['token_f1']:.4f} |"
        for kind, values in by_question_type.items()
    )
    changes = sorted(paired, key=lambda row: (row["delta_token_f1"], row["id"]))
    examples = []
    analysis = []
    for label, cases in (("Chênh lệch F1 thấp nhất", changes[:5]), ("Chênh lệch F1 cao nhất", changes[-5:][::-1])):
        examples.append(f"### {label}\n")
        for row in cases:
            examples.append(f"- {row['id']} ({row['delta_token_f1']:+.4f}). {row['question']} [Căn cứ]({row['source_pdf_url']})")
            analysis.append({
                **row, "selection": label,
                "base_reached_limit": row["base_generated_tokens"] == 256,
                "adapter_reached_limit": row["adapter_generated_tokens"] == 256,
                "error_category": "", "review_status": "unverified", "reviewer_notes": "",
            })
        examples.append("")
    write_csv(staged / "error_analysis.csv", analysis)
    directions = Counter("tăng" if value > 0 else "giảm" if value < 0 else "bằng" for value in (row["delta_token_f1"] for row in paired))
    latency = "Hai lượt chạy dùng cùng môi trường suy luận." if same_environment else "Hai lượt chạy dùng môi trường khác nhau. Thời gian chỉ được báo riêng cho từng lượt."
    (staged / "comparison_report.md").write_text(
        f"# So sánh Base và QLoRA\n\n{len(rows)} câu hỏi theo đúng thứ tự của lượt đánh giá baseline.\n\n"
        "| Chỉ số | Base | QLoRA | Chênh lệch |\n| --- | ---: | ---: | ---: |\n" + table + "\n\n"
        f"{latency} Thời gian tính theo giây/câu.\n\n"
        "| Loại câu hỏi | Số mẫu | F1 Base | F1 QLoRA |\n| --- | ---: | ---: | ---: |\n" + type_table + "\n\n"
        f"Chạm giới hạn sinh: {baseline['at_generation_limit']}/{len(rows)} ở baseline và {limit_hits}/{len(rows)} ở QLoRA.\n\n"
        f"Token F1 tăng ở {directions['tăng']} câu, giảm ở {directions['giảm']} câu và bằng ở {directions['bằng']} câu.\n\n"
        "Đánh giá pháp lý độc lập đang ở trạng thái `unverified`. Phiếu phân tích chọn 10 câu theo Token F1. "
        "Các nhãn lỗi cần đối chiếu nguồn gồm nội dung pháp lý sai, bịa căn cứ, thiếu ý, lạc đề, từ chối sai và quy định hết hiệu lực. "
        "Khả năng từ chối ngoài phạm vi cần tập kiểm tra riêng.\n\n"
        f"Validation loss theo bước: {trend}. Giá trị thấp nhất {lowest['eval_loss']:.4f} ở bước {lowest['step']}. "
        "Đối chiếu đường cong train và validation khi nhận định overfitting.\n\n"
        + "\n".join(examples), encoding="utf-8",
    )
    publish(staged, run / "results", identity, len(rows), (*RESULT_FILES, *COMPARISON_FILES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    adapter, run = args.adapter.resolve(), args.run_dir.resolve()
    qwen, baseline, identity = evaluation_input(adapter)
    by_step = {
        row["step"]: {"step": row["step"], "eval_loss": row["eval_loss"]}
        for row in read_json(adapter / "trainer_state.json")["log_history"] if "eval_loss" in row
    }
    if not by_step or any(not math.isfinite(row["eval_loss"]) for row in by_step.values()):
        raise ValueError("Adapter has no valid validation history")
    identity["training_validation"] = [by_step[step] for step in sorted(by_step)]
    test = load_test(qwen, ROOT / "data/processed")
    check_environment()
    results = run / "results"
    if results.exists():
        print(f"Complete: {results} ({verify(results, identity, (*RESULT_FILES, *COMPARISON_FILES))} rows)")
        return
    if args.status:
        saved = sum((run / f"prediction-{index:03}.json").is_file() for index in range(1, len(test) + 1))
        print(f"Saved predictions: {saved}/{len(test)}. Writer lock: {(run / '.writer').exists()}")
        return
    summary = adapter.parent / "training_summary"
    verify(summary, identity["adapter"], TRAINING_SUMMARY_FILES)
    with writer(run):
        if results.exists():
            print(f"Complete: {results} ({verify(results, identity, (*RESULT_FILES, *COMPARISON_FILES))} rows)")
            return
        manifest = run / "evaluation_config.json"
        if manifest.exists():
            if read_json(manifest) != identity:
                raise ValueError("Use the original adapter or a different evaluation directory")
        else:
            write_json(manifest, identity)
        rows = list(predict(test, run, adapter, qwen, baseline, args.offline))
        report_results(run, rows, baseline, identity, summary / "training_curve.png")


if __name__ == "__main__":
    main()
