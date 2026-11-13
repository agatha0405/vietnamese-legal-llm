"""Run the Windows workspace pipeline from the latest verified artifact."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from evaluate_baseline import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/qwen3b-r8-local")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "outputs/evaluation/qwen3b-r8")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    adapter = run / "final_adapter"
    env = os.environ.copy()
    cache = ROOT / "artifacts/hf-cache"
    if cache.is_dir():
        env.setdefault("HF_HUB_CACHE", str(cache))
    offline = ["--offline"] if args.offline else []
    commands = [
        ["train_qlora.py", "--run-dir", str(run), *offline],
        ["summarize_training.py", "--adapter", str(adapter)],
        ["demo.py", "--adapter", str(adapter), "--check", *offline],
        ["evaluate_adapter.py", "--adapter", str(adapter), "--run-dir", str(args.eval_dir.resolve()), *offline],
    ]
    for script, *options in commands:
        subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / script), *options], cwd=ROOT, env=env, check=True)
    subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "app/app.py"), "--adapter", str(adapter), "--check", *offline],
        cwd=ROOT, env=env, check=True,
    )


if __name__ == "__main__":
    main()
