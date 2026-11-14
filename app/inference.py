"""Interactive decoding settings and per-request measurements for the chatbot."""

from dataclasses import dataclass
import re
from statistics import mean
from time import perf_counter, process_time
import unicodedata


CHAT_SYSTEM_PROMPT = (
    "Bạn là chatbot về pháp luật giao thông đường bộ Việt Nam. Trả lời bằng tiếng Việt "
    "tự nhiên, ngắn gọn (1-2 câu), đúng trọng tâm câu hỏi. Nếu hỏi có được phép hay không, "
    "trả lời Có hoặc Không trước rồi giải thích ngắn gọn. Nếu hỏi định nghĩa, chỉ giải thích "
    "khái niệm. Không tự động nêu số điều, khoản, tên văn bản, nguồn tham khảo hoặc mốc thời gian "
    "trừ khi người dùng yêu cầu cụ thể. Nếu thiếu thông tin để trả lời, hãy hỏi lại ngắn gọn "
    "người dùng; nếu chưa có căn cứ đã đối chiếu, nói ngắn gọn là chưa đủ thông tin để trả lời "
    "chính xác; không đoán số tiền, số điều, khoản hoặc văn bản. Nếu ngoài phạm vi giao thông "
    "đường bộ Việt Nam, trả lời ngắn rằng câu hỏi nằm ngoài phạm vi. Dừng sau câu trả lời, không lặp."
)

SANCTION = re.compile(
    r"\b(phạt (?:bao nhiêu|tiền|mấy|gì|thế nào|như thế nào|không)|mức phạt|bị phạt|xử phạt|"
    r"(mất|nộp|đóng) bao nhiêu (tiền|điểm)|trừ (?:bao nhiêu |mấy )?điểm|tước (?:bằng|giấy phép))\b"
)
ROAD_CONTEXT = re.compile(
    r"\b(đèn (?:đỏ|vàng|tín hiệu)|tốc độ|lấn làn|vượt xe|đường (?:bộ|cao tốc)|"
    r"lái xe|bằng lái|giấy phép lái xe|nồng độ cồn|mũ bảo hiểm|đỗ xe|dừng xe|"
    r"đi ngược chiều|biển báo|đăng kiểm|phạt nguội|giao thông)\b"
)
VEHICLE = re.compile(
    r"\b(ô\s*tô|xe\s+(?:máy|gắn máy|mô tô|đạp|tải|buýt|khách|con|bán tải|đầu kéo|"
    r"bốn bánh|4 bánh|ba bánh|3 bánh|chuyên dùng|cứu thương|cứu hỏa)|mô tô|phương tiện tương tự|đi bộ)\b"
)
CLARIFY_VEHICLE = "Bạn đi loại phương tiện nào, chẳng hạn ô tô, xe máy hay xe đạp?"
NO_SANCTION_SOURCE = (
    "Tôi chưa có căn cứ xử phạt đã đối chiếu cho trường hợp này nên chưa thể nêu mức phạt hoặc số điểm bị trừ."
)


def sanction_reply(messages):
    """Avoid unsupported penalties. Handle a vehicle-only reply to our clarification."""
    question = unicodedata.normalize("NFC", messages[-1]["content"]).casefold()
    has_vehicle = VEHICLE.search(question)
    if (has_vehicle and len(question.split()) <= 6 and len(messages) >= 4
            and messages[-2] == {"role": "assistant", "content": CLARIFY_VEHICLE}):
        return NO_SANCTION_SOURCE, "unverified_sanction"
    if SANCTION.search(question) and (has_vehicle or ROAD_CONTEXT.search(question)):
        if has_vehicle:
            return NO_SANCTION_SOURCE, "unverified_sanction"
        return CLARIFY_VEHICLE, "clarify_vehicle"
    return None


def chat_config(qwen, profile):
    if profile == "legacy":
        return qwen
    if profile == "prompt-only":
        return {**qwen, "system_prompt": CHAT_SYSTEM_PROMPT}
    if profile != "concise":
        raise ValueError(f"Unknown inference profile: {profile}")
    return {
        **qwen, "system_prompt": CHAT_SYSTEM_PROMPT,
        "generation": {**qwen["generation"], "max_new_tokens": 96, "no_repeat_ngram_size": 6},
    }


@dataclass(frozen=True)
class Generation:
    prediction: str
    prompt_tokens: int
    generated_tokens: int
    last_token_id: int | None
    stop_reason: str
    tokenize_s: float
    generate_s: float
    decode_s: float
    tokens_per_s: float
    cpu_core_percent: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    repeated_6gram_fraction: float


def generate(model, tokenizer, config, messages, *, warmup=False):
    import torch
    from transformers import GenerationConfig

    started = perf_counter()
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    torch.cuda.synchronize()
    tokenize_s = perf_counter() - started
    prompt_length = inputs["input_ids"].shape[1]
    options = dict(config["generation"])
    if any("tham chiếu đã kiểm chứng" in str(m.get("content", "")) for m in messages):
        # Hugging Face's NoRepeatNGramLogitsProcessor bans 6-grams present in the prompt itself,
        # which prevents copying verified RAG clauses verbatim.
        options.pop("no_repeat_ngram_size", None)
    if warmup:
        options["max_new_tokens"] = 8
    generation_config = GenerationConfig(**options)
    torch.cuda.reset_peak_memory_stats()
    started_cpu = process_time()
    started = perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=generation_config)
    torch.cuda.synchronize()
    duration = perf_counter() - started
    cpu = 100 * (process_time() - started_cpu) / duration
    started = perf_counter()
    tokens = output[0, prompt_length:].tolist()
    prediction = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    last = tokens[-1] if tokens else None
    eos = set(options["eos_token_id"])
    if last in eos:
        reason = "eos"
    elif len(tokens) == options["max_new_tokens"]:
        reason = "max_new_tokens"
    else:
        reason = "other"
    ngrams = list(zip(*(tokens[i:] for i in range(6))))
    repeated = (len(ngrams) - len(set(ngrams))) / len(ngrams) if ngrams else 0.0
    return Generation(
        prediction, prompt_length, len(tokens), last, reason, tokenize_s, duration,
        perf_counter() - started, len(tokens) / duration, cpu,
        torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20, repeated,
    )


def sample_gpu(samples):
    """Summarize device-wide samples. An unavailable sample stays absent."""
    return {
        "gpu_samples": len(samples),
        "gpu_util_mean_pct": mean(s[0] for s in samples) if samples else None,
        "gpu_memory_peak_mib": max((s[1] for s in samples), default=None),
    }
