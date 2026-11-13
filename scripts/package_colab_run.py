"""Package the latest verified checkpoint and Colab sources without overwriting ZIP files."""

import argparse
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from evaluate_baseline import ROOT
from run_checkpoint import read_json, sha256
from train_qlora import selected_checkpoint


SOURCE_FILES = (
    "configs/environment.toml", "configs/inference.toml", "configs/training.toml",
    "requirements-torch.txt", "requirements.txt", "requirements-analysis.txt",
    "scripts/evaluate_baseline.py", "scripts/train_qlora.py", "scripts/run_checkpoint.py",
    "scripts/evaluate_adapter.py", "scripts/evaluate_refusal.py", "scripts/summarize_training.py", "scripts/demo.py",
    "app/app.py", "data/processed/legal_qa_master.jsonl", "data/processed/train.jsonl",
    "data/processed/validation.jsonl", "data/processed/test.jsonl", "data/processed/out_of_scope_test.jsonl",
    "outputs/baseline/baseline_predictions.csv", "outputs/baseline/baseline_metrics.json",
)


def package(target, entries):
    if target.exists():
        with ZipFile(target) as existing:
            if existing.testzip() is not None or set(existing.namelist()) != set(entries):
                raise ValueError(f"Existing ZIP differs: {target}")
            for name, path in entries.items():
                with existing.open(name) as item:
                    if file_digest(item, "sha256").hexdigest() != sha256(path):
                        raise ValueError(f"Existing ZIP differs: {target}")
        print(f"Preserved {target}")
        return
    with NamedTemporaryFile(dir=target.parent, prefix=".colab-", suffix=".zip", delete=False) as handle:
        staging = Path(handle.name)
    with ZipFile(staging, "w", ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in entries.items():
            archive.write(path, name)
    with ZipFile(staging) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Invalid ZIP: {staging}")
    staging.rename(target)
    print(f"Saved {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/qwen3b-r8-local")
    parser.add_argument("--source-name", default="qlora_colab_source.zip")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    package(artifacts / Path(args.source_name).name, {path: ROOT / path for path in SOURCE_FILES})
    if args.source_only:
        return
    run = args.run_dir.resolve()
    if (run / ".writer").exists():
        raise RuntimeError("Wait for the active training process before packaging its checkpoint")
    identity_path = run / "run_config.json"
    latest, _, _ = selected_checkpoint(run, read_json(identity_path))
    if latest is None:
        raise FileNotFoundError(f"No verified checkpoint in {run}")
    step, checkpoint = latest
    entries = {f"{run.name}/run_config.json": identity_path}
    entries.update({
        f"{run.name}/checkpoint-{step}/{path.relative_to(checkpoint).as_posix()}": path
        for path in sorted(checkpoint.rglob("*")) if path.is_file()
    })
    package(artifacts / f"{run.name}-resume-step{step}.zip", entries)


if __name__ == "__main__":
    main()
