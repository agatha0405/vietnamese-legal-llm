"""Run a local Gradio demo backed by the verified fine-tuned adapter."""

import argparse
from pathlib import Path
from uuid import uuid4

from evaluate_baseline import check_environment, generate_answer, load_qwen
from run_checkpoint import publish, read_json, verify, write_json, writer
from train_qlora import FINAL_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check", action="store_true", help="Generate one answer without starting the server")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    check_environment()
    manifest = read_json(args.adapter / "complete.json")
    verify(args.adapter, manifest["identity"], FINAL_FILES)
    qwen = manifest["identity"]["qwen"]
    smoke = args.adapter.resolve().parent / "demo_check"
    if args.check and smoke.exists():
        verify(smoke, manifest, ("generation.json",))
        print(f"Preserving {smoke}")
        return

    from peft import PeftModel

    base, tokenizer, _ = load_qwen(qwen, args.offline)
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval()

    def respond(message, history):
        question = message.strip()
        if not question:
            return "Hãy nhập một câu hỏi."
        if len(tokenizer.encode(question, add_special_tokens=False)) > 768:
            return "Câu hỏi vượt quá 768 token. Hãy rút gọn câu hỏi."
        return generate_answer(model, tokenizer, qwen, question)[0] or "Mô hình chưa sinh được câu trả lời."

    if args.check:
        question = "Khi nào người lái xe cần bật đèn cảnh báo nguy hiểm?"
        with writer(smoke.parent):
            if smoke.exists():
                verify(smoke, manifest, ("generation.json",))
                return
            answer, seconds, count = generate_answer(model, tokenizer, qwen, question)
            if not answer or count < 1:
                raise RuntimeError("Demo adapter produced no answer")
            staged = smoke.parent / f".demo-check-{uuid4().hex}"
            staged.mkdir()
            write_json(staged / "generation.json", {
                "question": question, "prediction": answer, "inference_time_s": seconds, "generated_tokens": count,
            })
            publish(staged, smoke, manifest, 1, ("generation.json",))
        print(f"{question}\n{answer}\n{count} new tokens in {seconds:.2f}s")
        return

    import gradio as gr

    gr.ChatInterface(
        respond, title="Hỏi đáp pháp luật giao thông Việt Nam",
        description="Demo nghiên cứu. Mỗi lượt xử lý một câu hỏi. Hãy đối chiếu căn cứ với văn bản chính thức.",
        concurrency_limit=1,
    ).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
