"""Run Stage C handoff evaluation: Base vs Fine-tuned, Standalone vs RAG, E2E grounded chatbot, and Gradio UI smoke test."""

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from evaluate_baseline import check_environment, load_qwen, score  # noqa: E402
from run_checkpoint import read_json, verify, write_json, writer  # noqa: E402
from train_qlora import FINAL_FILES  # noqa: E402
from app.app import CONTEXT_LIMIT, build_app, extract_content, prepare_messages  # noqa: E402
from app.grounding import LegalIndex, unverified_sanction_output  # noqa: E402
from app.inference import chat_config, generate  # noqa: E402


CONDITIONS = (
    ("base_standalone", True, False),
    ("finetuned_standalone", False, False),
    ("base_rag", True, True),
    ("finetuned_rag", False, True),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage C GPU regression and Base/Fine-tuned Standalone/RAG evaluation.")
    parser.add_argument("--adapter", type=Path, default=Path("runs/qwen3b-r8-local/final_adapter"))
    parser.add_argument("--cases", type=Path, default=Path("tests/chatbot_handoff_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/chatbot-handoff-20261114"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recheck-only", action="store_true")
    return parser.parse_args()


UI_SESSION_1_SPECS = (
    (
        "ui-fake-999-turn1",
        "Theo Điều 999 Luật Trật tự, an toàn giao thông đường bộ, vượt đèn đỏ bị phạt bao nhiêu?",
        ["không có Điều 999", "89 điều", "ô tô", "xe máy"],
        ["hợp lệ"],
    ),
    (
        "ui-fake-999-turn2-moto",
        "Xe máy",
        ["không có Điều 999", "4.000.000 đến 6.000.000 đồng", "4 điểm", "Điều 7"],
        ["mô tô thuộc nhóm"],
    ),
    (
        "ui-fake-999-turn3-car",
        "Còn ô tô thì sao?",
        ["18.000.000 đến 20.000.000 đồng", "4 điểm", "Điều 6"],
        ["chưa có căn cứ"],
    ),
)

UI_SESSION_2_SPECS = (
    (
        "ui-wrongway-moto",
        "Đi xe máy ngược chiều trên đường một chiều bị xử phạt như thế nào?",
        ["4.000.000 đến 6.000.000 đồng", "2 điểm", "Điều 7"],
        ["chưa có căn cứ"],
    ),
    (
        "ui-rag-priority-road",
        "Mô tả đường ưu tiên.",
        ["Fine-tuned ·", "đường ưu tiên", "giao nhau"],
        ["ngược chiều", "ngoài phạm vi", "chưa có căn cứ"],
    ),
    (
        "ui-out-of-scope",
        "Tôi muốn đăng ký kết hôn thì chuẩn bị giấy tờ gì?",
        ["ngoài phạm vi"],
        ["Nghị định 168"],
    ),
)


def check_substrings(text, must_contain, must_not_contain):
    lowered = text.lower()
    missing = [sub for sub in must_contain if sub.lower() not in lowered]
    forbidden = [sub for sub in must_not_contain if sub.lower() in lowered]
    return not missing and not forbidden, missing, forbidden


def normalize_history(history):
    return [{"role": item["role"], "content": extract_content(item["content"])} for item in history]


def evaluate_model_conditions(args, manifest, step, qwen, cases, legal_index):
    import torch
    from peft import PeftModel

    torch.set_num_threads(args.cpu_threads)
    started = perf_counter()
    base, tokenizer, environment = load_qwen(qwen, args.offline)
    torch.cuda.synchronize()
    load_s = perf_counter() - started
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval()
    torch.cuda.synchronize()
    print(f"[Phase 1] Loaded Qwen + adapter (step {step}) in {load_s:.2f}s", flush=True)

    rows = []
    summaries = {}
    try:
        for cond_name, disable_adapter, use_rag in CONDITIONS:
            cond_rows = []
            with model.disable_adapter() if disable_adapter else nullcontext():
                warmup_msgs = prepare_messages(tokenizer, qwen["system_prompt"], cases[0]["question"], [])
                generate(model, tokenizer, qwen, warmup_msgs, warmup=True)
                for case in cases:
                    msgs = prepare_messages(tokenizer, qwen["system_prompt"], case["question"], case["history"])
                    hits = []
                    if use_rag:
                        hits = legal_index.retrieve(case["question"], top_k=2)
                        msgs = legal_index.augment_messages(msgs, top_k=2, tokenizer=tokenizer, context_limit=CONTEXT_LIMIT)
                    result = generate(model, tokenizer, qwen, msgs)
                    metrics = score(result.prediction, case["reference"])
                    row = {
                        "condition": cond_name,
                        "use_rag": use_rag,
                        "mode": "base" if disable_adapter else "fine-tuned",
                        "case_id": case["id"],
                        "category": case["category"],
                        "question": case["question"],
                        "reference": case["reference"],
                        "rag_citations": [h["citation"] for h in hits],
                        "rag_ids": [h["id"] for h in hits],
                        **asdict(result),
                        "text_metrics": metrics,
                    }
                    rows.append(row)
                    cond_rows.append(row)
                    print(
                        f"  [{cond_name}] {case['id']}: rouge_l={metrics['rouge_l']:.4f}, "
                        f"token_f1={metrics['token_f1']:.4f}, {result.generated_tokens} tok, {result.generate_s:.2f}s",
                        flush=True,
                    )
            summaries[cond_name] = {
                "mean_rouge_l": round(mean(r["text_metrics"]["rouge_l"] for r in cond_rows), 4),
                "mean_token_f1": round(mean(r["text_metrics"]["token_f1"] for r in cond_rows), 4),
                "exact_matches": sum(r["text_metrics"]["exact_match"] for r in cond_rows),
                "mean_generate_s": round(mean(r["generate_s"] for r in cond_rows), 4),
                "mean_generated_tokens": round(mean(r["generated_tokens"] for r in cond_rows), 2),
                "mean_repeated_6gram": round(mean(r["repeated_6gram_fraction"] for r in cond_rows), 4),
                "eos_stops": sum(1 for r in cond_rows if r["stop_reason"] == "eos"),
                "count": len(cond_rows),
            }
    finally:
        del model, base, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return environment, load_s, rows, summaries


def evaluate_e2e_and_ui(args, cases, legal_index):
    import torch
    from gradio_client import Client

    print("[Phase 2] Building full grounded Gradio app...", flush=True)
    ci = build_app(args.adapter, offline=args.offline, profile="grounded", cpu_threads=args.cpu_threads)
    respond = ci.fn

    # Warmup Base & Fine-tuned inside respond once
    respond("Đường bộ là gì?", [], "Fine-tuned", warmup=True)
    respond("Đường bộ là gì?", [], "Base", warmup=True)

    e2e_rows = []
    for case in cases:
        raw_msgs = [*normalize_history(case["history"]), {"role": "user", "content": case["question"]}]
        decision = legal_index.resolve(raw_msgs)
        actual_kind = None if decision is None else decision.kind
        started = perf_counter()
        answer = respond(case["question"], case["history"], "Fine-tuned")
        elapsed = perf_counter() - started
        ok_subs, missing, forbidden = check_substrings(answer, case["must_contain"], case["must_not_contain"])
        ok_kind = actual_kind == case["expected_kind"]
        passed = ok_subs and ok_kind and bool(answer.strip())
        row = {
            "id": case["id"],
            "issue": case["issue"],
            "mode": "Fine-tuned",
            "expected_kind": case["expected_kind"],
            "actual_kind": actual_kind,
            "latency_s": round(elapsed, 4),
            "passed": passed,
            "missing": missing,
            "forbidden": forbidden,
            "unverified_sanction_flagged": unverified_sanction_output(answer) if actual_kind is None else False,
            "response": answer,
        }
        e2e_rows.append(row)
        print(f"  [E2E Fine-tuned] {case['id']}: kind={actual_kind}, passed={passed}, {elapsed:.3f}s", flush=True)

    # Also verify Base and So sánh modes on key representative cases
    mode_checks = []
    for case_id, mode in (
        ("handoff-a-red-car", "Base"),
        ("handoff-a-red-car", "So sánh"),
        ("handoff-d-fake-999-turn2-v6", "So sánh"),
        ("handoff-e-def-priority", "Base"),
        ("handoff-e-def-priority", "So sánh"),
    ):
        case = next(c for c in cases if c["id"] == case_id)
        started = perf_counter()
        answer = respond(case["question"], case["history"], mode)
        elapsed = perf_counter() - started
        ok_subs, missing, forbidden = check_substrings(answer, case["must_contain"], case["must_not_contain"])
        if mode == "So sánh" and case["expected_kind"] is None:
            has_both = answer.startswith("Base ·") and "\n\nFine-tuned ·" in answer
        else:
            has_both = True
        passed = ok_subs and has_both
        mode_checks.append({
            "id": case_id,
            "mode": mode,
            "latency_s": round(elapsed, 4),
            "passed": passed,
            "missing": missing,
            "forbidden": forbidden,
            "response": answer,
        })
        print(f"  [E2E {mode}] {case_id}: passed={passed}, {elapsed:.3f}s", flush=True)

    # Live Gradio v6 HTTP/Client multi-turn UI verification
    print(f"[Phase 3] Launching live Gradio 6.9.0 UI on 127.0.0.1:{args.port} for client smoke check...", flush=True)
    ci.launch(server_name="127.0.0.1", server_port=args.port, prevent_thread_lock=True, quiet=True)
    ui_results = []
    try:
        # Session 1: 3-turn Condition D + Condition C multi-turn verification
        client1 = Client(f"http://127.0.0.1:{args.port}", verbose=False)
        for step_id, q, must_have, must_not in UI_SESSION_1_SPECS:
            out = client1.predict(q, "Fine-tuned", api_name="/respond")
            ok_subs, missing, forbidden = check_substrings(str(out), must_have, must_not)
            ui_results.append({
                "session": "multiturn_fake_citation",
                "step": step_id,
                "question": q,
                "passed": ok_subs,
                "missing": missing,
                "forbidden": forbidden,
                "response": str(out),
            })
            print(f"  [Gradio Client] {step_id}: passed={ok_subs}", flush=True)

        # Session 2: RAG Definition + Over-refusal + Out-of-scope in live UI
        client2 = Client(f"http://127.0.0.1:{args.port}", verbose=False)
        for step_id, q, must_have, must_not in UI_SESSION_2_SPECS:
            out = client2.predict(q, "Fine-tuned", api_name="/respond")
            ok_subs, missing, forbidden = check_substrings(str(out), must_have, must_not)
            ui_results.append({
                "session": "grounded_and_rag",
                "step": step_id,
                "question": q,
                "passed": ok_subs,
                "missing": missing,
                "forbidden": forbidden,
                "response": str(out),
            })
            print(f"  [Gradio Client] {step_id}: passed={ok_subs}", flush=True)
    finally:
        ci.close()
        del ci, respond
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return e2e_rows, mode_checks, ui_results


def write_outputs(args, step, environment, load_s, cmp_rows, cmp_summaries, e2e_rows, mode_checks, ui_results):
    with writer(args.output):
        write_json(args.output / "model_comparison.json", {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_step": step,
            "environment": environment,
            "load_s": round(load_s, 4),
            "summaries": cmp_summaries,
            "rows": cmp_rows,
        })
        write_json(args.output / "e2e_regression.json", {
            "total": len(e2e_rows),
            "passed": sum(1 for r in e2e_rows if r["passed"]),
            "mode_checks_total": len(mode_checks),
            "mode_checks_passed": sum(1 for r in mode_checks if r["passed"]),
            "rows": e2e_rows,
            "mode_checks": mode_checks,
        })
        write_json(args.output / "gradio_ui_smoke.json", {
            "total": len(ui_results),
            "passed": sum(1 for r in ui_results if r["passed"]),
            "rows": ui_results,
        })
        with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "condition", "mode", "use_rag", "mean_rouge_l", "mean_token_f1",
                "exact_matches", "mean_generate_s", "mean_generated_tokens",
                "mean_repeated_6gram", "eos_stops", "count",
            ]
            writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
            writer_csv.writeheader()
            for cond_name, disable_adapter, use_rag in CONDITIONS:
                writer_csv.writerow({
                    "condition": cond_name,
                    "mode": "base" if disable_adapter else "fine-tuned",
                    "use_rag": use_rag,
                    **cmp_summaries[cond_name],
                })
        (args.output / "README.md").write_text(
            render_readme(args, step, cmp_summaries, e2e_rows, mode_checks, ui_results),
            encoding="utf-8",
        )


def render_readme(args, step, cmp_summaries, e2e_rows, mode_checks, ui_results):
    readme_lines = [
        "# Kiểm thử Stage C sau tiếp quản dự án (2026-11-14)",
        "",
        f"- Adapter checkpoint: `{args.adapter}` (step {step}).",
        f"- Tập kiểm thử: `{args.cases}` ({len(cmp_summaries)} cấu hình trên {cmp_summaries['base_standalone']['count']} câu suy luận LLM + {len(e2e_rows)} ca E2E app + {len(ui_results)} lượt `gradio_client`).",
        "",
        "## 1. So sánh Base vs Fine-tuned và Standalone QLoRA vs RAG (ngoài bảng rule xử phạt)",
        "",
        "| Cấu hình | Chế độ | RAG | Mean ROUGE-L | Mean Token F1 | Exact Match | Mean latency (s) | EOS stops |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cond_name, disable_adapter, use_rag in CONDITIONS:
        s = cmp_summaries[cond_name]
        readme_lines.append(
            f"| `{cond_name}` | `{'Base' if disable_adapter else 'Fine-tuned'}` | {'Có' if use_rag else 'Không'} | "
            f"{s['mean_rouge_l']:.4f} | {s['mean_token_f1']:.4f} | {s['exact_matches']}/{s['count']} | "
            f"{s['mean_generate_s']:.2f} | {s['eos_stops']}/{s['count']} |"
        )
    readme_lines.extend([
        "",
        "## 2. Kết quả hồi quy ứng dụng & Gradio UI thực tế",
        "",
        f"- E2E `app/app.py` (`Fine-tuned`, lịch sử Gradio v6 `list[dict]`): **{sum(r['passed'] for r in e2e_rows)}/{len(e2e_rows)} đạt**.",
        f"- Kiểm tra chéo chế độ `Base` và `So sánh`: **{sum(r['passed'] for r in mode_checks)}/{len(mode_checks)} đạt**.",
        f"- Kiểm thử phiên hội thoại sống qua `gradio_client` (`http://127.0.0.1:{args.port}`): **{sum(r['passed'] for r in ui_results)}/{len(ui_results)} lượt đạt**.",
    ])
    return "\n".join(readme_lines) + "\n"


def recheck_existing(args, step, suite):
    if "chatbot-handoff-" not in args.output.name:
        raise ValueError(f"Refusing to recheck outside chatbot-handoff-* output directory: {args.output}")
    cmp_data = read_json(args.output / "model_comparison.json")
    e2e_data = read_json(args.output / "e2e_regression.json")
    ui_data = read_json(args.output / "gradio_ui_smoke.json")

    case_by_id = {c["id"]: c for c in suite["e2e_app_cases"]}
    e2e_rows = []
    for row in e2e_data["rows"]:
        case = case_by_id[row["id"]]
        ok_subs, missing, forbidden = check_substrings(row["response"], case["must_contain"], case["must_not_contain"])
        passed = ok_subs and (row["actual_kind"] == case["expected_kind"]) and bool(row["response"].strip())
        e2e_rows.append({**row, "passed": passed, "missing": missing, "forbidden": forbidden})

    mode_checks = []
    for row in e2e_data["mode_checks"]:
        case = case_by_id[row["id"]]
        ok_subs, missing, forbidden = check_substrings(row["response"], case["must_contain"], case["must_not_contain"])
        has_both = (
            row["response"].startswith("Base ·") and "\n\nFine-tuned ·" in row["response"]
            if row["mode"] == "So sánh" and case["expected_kind"] is None
            else True
        )
        mode_checks.append({**row, "passed": ok_subs and has_both, "missing": missing, "forbidden": forbidden})

    ui_specs = {spec[0]: spec for spec in (*UI_SESSION_1_SPECS, *UI_SESSION_2_SPECS)}
    ui_results = []
    for row in ui_data["rows"]:
        _, _, must_have, must_not = ui_specs[row["step"]]
        ok_subs, missing, forbidden = check_substrings(row["response"], must_have, must_not)
        ui_results.append({**row, "passed": ok_subs, "missing": missing, "forbidden": forbidden})

    e2e_path = args.output / "e2e_regression.json"
    ui_path = args.output / "gradio_ui_smoke.json"
    e2e_path.unlink()
    ui_path.unlink()
    with writer(args.output):
        write_json(e2e_path, {
            "total": len(e2e_rows),
            "passed": sum(1 for r in e2e_rows if r["passed"]),
            "mode_checks_total": len(mode_checks),
            "mode_checks_passed": sum(1 for r in mode_checks if r["passed"]),
            "rows": e2e_rows,
            "mode_checks": mode_checks,
        })
        write_json(ui_path, {
            "total": len(ui_results),
            "passed": sum(1 for r in ui_results if r["passed"]),
            "rows": ui_results,
        })
        (args.output / "README.md").write_text(
            render_readme(args, step, cmp_data["summaries"], e2e_rows, mode_checks, ui_results),
            encoding="utf-8",
        )
    return cmp_data["summaries"], e2e_rows, mode_checks, ui_results


def main():
    args = parse_args()
    check_environment()
    manifest = read_json(args.adapter / "complete.json")
    step = verify(args.adapter, manifest["identity"], FINAL_FILES)
    suite = read_json(args.cases)
    if args.recheck_only:
        cmp_summaries, e2e_rows, mode_checks, ui_results = recheck_existing(args, step, suite)
        print(
            f"Rechecked {args.output}: E2E {sum(r['passed'] for r in e2e_rows)}/{len(e2e_rows)}, "
            f"modes {sum(r['passed'] for r in mode_checks)}/{len(mode_checks)}, "
            f"UI {sum(r['passed'] for r in ui_results)}/{len(ui_results)}"
        )
    else:
        qwen = chat_config(manifest["identity"]["qwen"], "concise")
        legal_index = LegalIndex()
        if args.output.exists() and any(args.output.iterdir()):
            if not args.overwrite or "chatbot-handoff-" not in args.output.name:
                raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
            for old_file in args.output.iterdir():
                if old_file.is_file():
                    old_file.unlink()
        args.output.mkdir(parents=True, exist_ok=True)
        environment, load_s, cmp_rows, cmp_summaries = evaluate_model_conditions(
            args, manifest, step, qwen, suite["model_comparison_cases"], legal_index,
        )
        e2e_rows, mode_checks, ui_results = evaluate_e2e_and_ui(args, suite["e2e_app_cases"], legal_index)
        write_outputs(args, step, environment, load_s, cmp_rows, cmp_summaries, e2e_rows, mode_checks, ui_results)
        print(f"\nSaved Stage C results to {args.output}")
        for cond_name, _, _ in CONDITIONS:
            print(cond_name, cmp_summaries[cond_name])
    failed_e2e = [r["id"] for r in e2e_rows if not r["passed"]]
    failed_mode = [(r["id"], r["mode"]) for r in mode_checks if not r["passed"]]
    failed_ui = [r["step"] for r in ui_results if not r["passed"]]
    if failed_e2e or failed_mode or failed_ui:
        raise AssertionError(f"Failures detected: e2e={failed_e2e}, mode={failed_mode}, ui={failed_ui}")


if __name__ == "__main__":
    main()
