"""Export optimizer-step logs and a learning curve from the final adapter."""

import argparse
from pathlib import Path
from uuid import uuid4

from evaluate_baseline import write_csv
from run_checkpoint import publish, read_json, verify, write_json, writer
from train_qlora import FINAL_FILES


FILES = ("training_history.csv", "training_curve.png", "training_summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    adapter = args.adapter.resolve()
    run = adapter.parent
    manifest = read_json(adapter / "complete.json")
    steps = verify(adapter, manifest["identity"], FINAL_FILES)
    target = run / "training_summary"
    if target.exists():
        verify(target, manifest, FILES)
        print(f"Preserving {target}")
        return
    with writer(run):
        if target.exists():
            verify(target, manifest, FILES)
            return
        history = read_json(adapter / "trainer_state.json")["log_history"]
        training = {row["step"]: row["loss"] for row in history if "loss" in row}
        validation = {row["step"]: row["eval_loss"] for row in history if "eval_loss" in row}
        if sorted(training) != list(range(1, steps + 1)) or not validation or max(validation) != steps:
            raise ValueError("Training or validation history is incomplete")
        staged = run / f".summary-{uuid4().hex}"
        staged.mkdir()
        write_csv(staged / "training_history.csv", [
            {"step": step, "loss": training[step], "eval_loss": validation.get(step, "")}
            for step in sorted(training)
        ])
        metrics = read_json(adapter / "training_metrics.json")
        segments = [
            read_json(path) for path in sorted(run.glob("training_segment-*.json"))
        ]
        write_json(staged / "training_summary.json", {
            "step": steps, "final_metrics": metrics,
            "validation_loss_by_step": [{"step": step, "loss": validation[step]} for step in sorted(validation)],
            "mean_logged_train_step_loss": sum(training.values()) / len(training),
            "timed_segments": segments,
            "coverage_note": "Earlier or interrupted segments can lack separate timing data. A resumed Trainer aggregate train_loss is not a whole-run mean.",
        })
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.plot(list(training), list(training.values()), label="Train", alpha=0.7)
        ax.plot(list(validation), list(validation.values()), marker="o", label="Validation")
        ax.set(xlabel="Bước cập nhật optimizer", ylabel="Loss", title="Huấn luyện QLoRA")
        ax.legend()
        fig.savefig(staged / "training_curve.png", dpi=160)
        plt.close(fig)
        publish(staged, target, manifest, steps, FILES)
        print(f"Training curve and logs: {target}")


if __name__ == "__main__":
    main()
