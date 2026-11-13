"""Gradio interface for Qwen and its verified traffic-law adapter."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import re
import sys
from threading import Lock
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_baseline import check_environment, generate_messages, load_qwen  # noqa: E402
from run_checkpoint import publish, read_json, verify, write_json, writer  # noqa: E402
from train_qlora import FINAL_FILES  # noqa: E402


MODES = ("Base", "Fine-tuned", "So sánh")
CONTEXT_LIMIT = 1024
RESPONSE_HEADER = re.compile(r"(?:\A|\n\n)(?:Base|Fine-tuned) · \d+\.\d{2} giây\n\n")


def prepare_messages(tokenizer, system, question, history):
    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    initial = [{"role": "system", "content": system}]
    current = [{"role": "user", "content": question}]
    if count(initial + current) > CONTEXT_LIMIT:
        raise ValueError("Câu hỏi quá dài. Hãy rút gọn câu hỏi.")
    previous = []
    pending = None
    for message in history:
        if not isinstance(message["content"], str):
            continue
        if message["role"] == "user":
            pending = {"role": "user", "content": message["content"]}
        elif message["role"] == "assistant" and pending is not None:
            # The last block is the adapter answer in comparison mode.
            answer = RESPONSE_HEADER.split(message["content"])[-1]
            previous.extend((pending, {"role": "assistant", "content": answer}))
            pending = None
    while previous and count(initial + previous + current) > CONTEXT_LIMIT:
        previous = previous[2:]
    return initial + previous + current


def load_chat(adapter, offline=False):
    import gradio as gr
    import torch
    from peft import PeftModel

    check_environment()
    manifest = read_json(adapter / "complete.json")
    step = verify(adapter, manifest["identity"], FINAL_FILES)
    qwen = manifest["identity"]["qwen"]
    base, tokenizer, _ = load_qwen(qwen, offline)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
    lock = Lock()

    def respond(question, history, mode, *, warmup=False):
        question = question.strip()
        if not question:
            raise gr.Error("Hãy nhập một câu hỏi.")
        if mode not in MODES:
            raise gr.Error("Hãy chọn chế độ mô hình.")
        try:
            messages = prepare_messages(tokenizer, qwen["system_prompt"], question, history)
        except ValueError as error:
            raise gr.Error(str(error)) from None
        answers = []
        try:
            with lock:
                for selected in (MODES[:2] if mode == MODES[2] else (mode,)):
                    with model.disable_adapter() if selected == MODES[0] else nullcontext():
                        answer, duration, _ = generate_messages(model, tokenizer, qwen, messages, warmup=warmup)
                    if not answer:
                        raise gr.Error(f"{selected} chưa sinh được câu trả lời.")
                    answers.append(f"{selected} · {duration:.2f} giây\n\n{answer}")
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as error:
            if "out of memory" not in str(error).lower():
                raise
        else:
            return "\n\n".join(answers)
        with lock:
            torch.cuda.empty_cache()
        raise gr.Error("GPU thiếu bộ nhớ. Hãy xóa lịch sử hoặc dùng câu hỏi ngắn hơn.")

    return respond, manifest, step


def build_app(adapter, offline=False):
    import gradio as gr

    respond, manifest, step = load_chat(adapter, offline)
    qwen = manifest["identity"]["qwen"]
    mode = gr.Dropdown(MODES, value="Fine-tuned", label="Mô hình")
    return gr.ChatInterface(
        respond, additional_inputs=[mode],
        title="Hỏi đáp pháp luật giao thông Việt Nam",
        description=(
            f"{qwen['model_id']} · NF4 · LoRA rank {manifest['identity']['adapter']['rank']} · bước {step}. "
            "Đây là hệ thống nghiên cứu, không thay thế tư vấn pháp lý. Hãy đối chiếu nội dung và căn cứ với văn bản chính thức."
        ),
        chatbot=gr.Chatbot(
            buttons=["copy"], sanitize_html=True, render_markdown=False,
            placeholder="Nhập một câu hỏi về giao thông đường bộ.",
        ),
        concurrency_limit=1,
    )


def check_chat(adapter, offline):
    import gradio as gr

    target = adapter.parent / "demo_suite_v2"
    manifest = read_json(adapter / "complete.json")
    verify(adapter, manifest["identity"], FINAL_FILES)
    required = ("cases.json",)
    with writer(adapter.parent):
        if target.exists():
            verify(target, manifest, required)
            print(f"Preserving {target}")
            return
        respond, _, step = load_chat(adapter, offline)
        first = "Khi nào người lái xe cần bật đèn cảnh báo nguy hiểm?"
        response = respond(first, [], "Fine-tuned")
        cases = [
            {"case": "ordinary", "question": first, "response": response},
        ]
        previous = [{"role": "user", "content": first}, {"role": "assistant", "content": response}]
        for name, question, history, mode in (
            ("out_of_scope", "Ngày mai trời ở Hà Nội có mưa không?", [], "Fine-tuned"),
            ("missing_information", "Tôi có được phép dừng ở đây không?", [], "Fine-tuned"),
            ("follow_up", "Nếu xe đang ở trong hầm thì sao?", previous, "Fine-tuned"),
            ("compare", "Phần đường xe chạy là gì?", [], "So sánh"),
        ):
            answer = respond(question, history, mode, warmup=mode == "So sánh")
            if mode == "So sánh" and not (answer.startswith("Base ·") and "\n\nFine-tuned ·" in answer):
                raise ValueError("Comparison did not generate both model responses")
            cases.append({"case": name, "question": question, "response": answer, "max_new_tokens": 8 if mode == "So sánh" else 256})
        try:
            respond("tình huống " * 1600, [], "Fine-tuned")
        except gr.Error as error:
            if "quá dài" not in str(error):
                raise
            cases.append({"case": "length_guard", "message": str(error)})
        else:
            raise ValueError("The app accepted an oversized prompt")
        staged = adapter.parent / f".demo-suite-{uuid4().hex}"
        staged.mkdir()
        write_json(staged / "cases.json", {"checkpoint_step": step, "legal_review_status": "unverified", "cases": cases})
        publish(staged, target, manifest, len(cases), required)
        print(f"Demo responses: {target / 'cases.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.check:
        check_chat(args.adapter.resolve(), args.offline)
    else:
        build_app(args.adapter, args.offline).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
