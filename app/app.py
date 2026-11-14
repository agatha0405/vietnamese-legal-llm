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

if __package__:
    from .grounding import LegalIndex, unverified_sanction_output
    from .inference import NO_SANCTION_SOURCE, chat_config, generate, sanction_reply
else:
    from grounding import LegalIndex, unverified_sanction_output
    from inference import NO_SANCTION_SOURCE, chat_config, generate, sanction_reply


MODES = ("Base", "Fine-tuned", "So sánh")
CONTEXT_LIMIT = 1024
RESPONSE_HEADER = re.compile(r"(?:\A|\n\n)(?:Base|Fine-tuned) · \d+\.\d{2} giây\n\n")
LENGTH_NOTICE = "\n\n[Lượt sinh chạm giới hạn độ dài. Câu trả lời có thể chưa đầy đủ.]"


def extract_content(content):
    """Extract plain text from string or Gradio multimodal message content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = extract_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
    return ""


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
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        raw_content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        text = extract_content(raw_content).strip()
        if not text:
            continue
        if role == "user":
            pending = {"role": "user", "content": text}
        elif role == "assistant" and pending is not None:
            # The last block is the adapter answer in comparison mode.
            answer = RESPONSE_HEADER.split(text)[-1].removesuffix(LENGTH_NOTICE)
            previous.extend((pending, {"role": "assistant", "content": answer}))
            pending = None
    while previous and count(initial + previous + current) > CONTEXT_LIMIT:
        previous = previous[2:]
    return initial + previous + current


def load_chat(adapter, offline=False, *, profile="grounded", cpu_threads=None):
    import gradio as gr
    import torch
    from peft import PeftModel

    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    check_environment()
    manifest = read_json(adapter / "complete.json")
    step = verify(adapter, manifest["identity"], FINAL_FILES)
    qwen = chat_config(manifest["identity"]["qwen"], "concise" if profile == "grounded" else profile)
    legal_index = LegalIndex() if profile == "grounded" else None
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
        if legal_index is not None and (decision := legal_index.resolve(messages)) is not None:
            return decision.render()
        if profile == "concise" and (clarification := sanction_reply(messages)) is not None:
            return clarification[0]
        generation_messages = (
            legal_index.augment_messages(messages, tokenizer=tokenizer, context_limit=CONTEXT_LIMIT)
            if legal_index is not None
            else messages
        )
        answers = []
        try:
            with lock:
                for selected in (MODES[:2] if mode == MODES[2] else (mode,)):
                    with model.disable_adapter() if selected == MODES[0] else nullcontext():
                        if profile == "legacy":
                            answer, duration, _ = generate_messages(model, tokenizer, qwen, messages, warmup=warmup)
                            notice = ""
                        else:
                            result = generate(model, tokenizer, qwen, generation_messages, warmup=warmup)
                            answer, duration = result.prediction, result.generate_s
                            notice = LENGTH_NOTICE if result.stop_reason == "max_new_tokens" and not warmup else ""
                            if legal_index is not None:
                                filtered = legal_index.filter_output(messages, answer)
                                if filtered == NO_SANCTION_SOURCE and unverified_sanction_output(answer):
                                    notice = ""
                                answer = filtered
                    if not answer:
                        raise gr.Error(f"{selected} chưa sinh được câu trả lời.")
                    answers.append(f"{selected} · {duration:.2f} giây\n\n{answer}{notice}")
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as error:
            if "out of memory" not in str(error).lower():
                raise
        else:
            return "\n\n".join(answers)
        with lock:
            torch.cuda.empty_cache()
        raise gr.Error("GPU thiếu bộ nhớ. Hãy xóa lịch sử hoặc dùng câu hỏi ngắn hơn.")

    return respond, manifest, step


def build_app(adapter, offline=False, *, profile="grounded", cpu_threads=None):
    import gradio as gr

    respond, _, _ = load_chat(adapter, offline, profile=profile, cpu_threads=cpu_threads)
    mode = gr.Dropdown(MODES, value="Fine-tuned", label="Mô hình")
    return gr.ChatInterface(
        respond, additional_inputs=[mode],
        title="Hỏi đáp pháp luật giao thông Việt Nam",
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
        respond, _, step = load_chat(adapter, offline, profile="legacy")
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
    parser.add_argument("--legacy", action="store_true", help="Use the original demo prompt and decoding settings")
    parser.add_argument("--cpu-threads", type=int, help="Set PyTorch intra-op threads for this process")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.check:
        check_chat(args.adapter.resolve(), args.offline)
    else:
        profile = "legacy" if args.legacy else "grounded"
        build_app(args.adapter, args.offline, profile=profile, cpu_threads=args.cpu_threads).queue().launch(
            server_name="127.0.0.1", server_port=args.port, share=True,
        )


if __name__ == "__main__":
    main()
