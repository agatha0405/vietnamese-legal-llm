"""Collect out-of-scope responses for blinded human review."""

import argparse
from pathlib import Path
import random
from uuid import uuid4

from evaluate_adapter import evaluation_input
from evaluate_baseline import ROOT, check_environment, digest, generate_answer, load_qwen, read_jsonl, write_csv
from run_checkpoint import publish, read_json, verify, write_json, writer


FILES = ("refusal_review.csv", "refusal_review_key.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    qwen, _, identity = evaluation_input(args.adapter.resolve())
    source = ROOT / "data/processed/out_of_scope_test.jsonl"
    identity = {"evaluation": identity, "out_of_scope_sha256": digest(source)}
    rows = read_jsonl(source)
    if len(rows) != 12 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected the 12 fixed out-of-scope prompts")
    check_environment()
    with writer(run):
        if (run / "results").exists():
            verify(run / "results", identity, FILES)
            return
        model = tokenizer = None
        rng = random.Random(42)
        review, key = [], []
        for row in rows:
            modes = ["base", "adapter"]
            rng.shuffle(modes)
            for code, mode in zip(("A", "B"), modes, strict=True):
                path = run / f"{row['id']}-{mode}.json"
                if not path.exists():
                    if model is None:
                        from peft import PeftModel

                        base, tokenizer, environment = load_qwen(qwen, args.offline)
                        env_path = run / "environment.json"
                        if env_path.exists():
                            if read_json(env_path) != {"environment": environment, "identity": identity}:
                                raise ValueError("Use the same adapter and environment when resuming")
                        else:
                            write_json(env_path, {"environment": environment, "identity": identity})
                        model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval()
                    if mode == "base":
                        with model.disable_adapter():
                            answer, seconds, tokens = generate_answer(model, tokenizer, qwen, row["question"])
                    else:
                        answer, seconds, tokens = generate_answer(model, tokenizer, qwen, row["question"])
                    if not answer:
                        raise RuntimeError(f"Empty output: {row['id']} {mode}")
                    write_json(path, {
                        "identity": identity, "answer": answer, "seconds": seconds, "tokens": tokens,
                    })
                saved = read_json(path)
                if saved["identity"] != identity or not saved["answer"].strip():
                    raise ValueError(f"Invalid saved response: {path}")
                review_id = f"{row['id']}-{code}"
                review.append({
                    "review_id": review_id, "question": row["question"], "prediction": saved["answer"],
                    "expected_behavior": row["expected_behavior"], "appropriate_refusal": "", "status": "unverified",
                })
                key.append({"review_id": review_id, "model": mode})
            print(f"Saved both responses for {row['id']}", flush=True)
        rng.shuffle(review)
        staged = run / f".refusal-{uuid4().hex}"
        staged.mkdir()
        write_csv(staged / "refusal_review.csv", review)
        write_csv(staged / "refusal_review_key.csv", key)
        publish(staged, run / "results", identity, len(review), FILES)


if __name__ == "__main__":
    main()
