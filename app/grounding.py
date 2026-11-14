"""Look up reviewed provisions separately from the QLoRA training data."""

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import tomllib
import unicodedata
from urllib.parse import urlparse

if __package__:
    from .inference import CLARIFY_VEHICLE, NO_SANCTION_SOURCE, ROAD_CONTEXT, SANCTION, VEHICLE
else:
    from inference import CLARIFY_VEHICLE, NO_SANCTION_SOURCE, ROAD_CONTEXT, SANCTION, VEHICLE


SOURCE_FILE = Path(__file__).with_name("legal_sources.toml")
CORPUS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "legal_qa_master.jsonl"
VIETNAM_TIME = timezone(timedelta(hours=7))

OUTSIDE = re.compile(
    r"\b(đăng ký kết hôn|ly hôn|thuế thu nhập|kê khai thuế|dự báo thời tiết|"
    r"có mưa không|quyền tác giả|bản quyền|thừa kế|phi công|máy bay|"
    r"xuất cảnh bằng đường hàng không|đường hàng không|thủ tục xuất cảnh|"
    r"nghỉ hằng năm|nghỉ phép|người lao động|bộ luật lao động|"
    r"giấy chứng nhận quyền sử dụng đất|sổ đỏ|"
    r"phương tiện thủy nội địa|đường thủy nội địa|"
    r"thuê căn hộ|tiền đặt cọc|hợp đồng thuê|"
    r"thành lập công ty|công ty cổ phần|đăng ký doanh nghiệp)\b"
)
FOREIGN = re.compile(
    r"(?:\b(?:ở|tại|luật của|pháp luật|nước)\s+(?:nhật bản|nước pháp|mỹ|hoa kỳ|anh quốc|thái lan|hàn quốc|trung quốc|singapore|đức|úc|canada)\b|"
    r"\b(?:ở|tại)\s+pháp\b|"
    r"\b(?:nhật bản|hoa kỳ|anh quốc|thái lan|hàn quốc|trung quốc|singapore|canada)\b|"
    r"^pháp\s+(?:quy định|luật|phạt|áp dụng)\b)"
)
VIETNAM = re.compile(r"\b(việt nam|trong nước)\b")

MOTO = re.compile(r"\b(xe (máy|gắn máy)|mô tô)\b")
CAR = re.compile(r"\b(ô\s*tô|xe (tải|con|buýt|khách|bán tải|đầu kéo))\b")
BICYCLE = re.compile(r"\bxe đạp(?: máy)?\b(?! điện)")
SPECIAL = re.compile(r"\bxe máy chuyên dùng\b")
OTHER_VEHICLE = re.compile(r"\b(xe (điện|đạp điện|máy điện|bốn bánh|4 bánh|ba bánh|3 bánh)|đi bộ)\b")
EXTRA_MOTO = re.compile(r"\bxe máy (điện|chuyên dùng)\b")

ACCIDENT = re.compile(r"\b(tai nạn|va chạm|đâm|tông|thương tích|chết người|bị thương)\b")
NO_ACCIDENT = re.compile(r"\b(không|chưa) (gây |xảy ra |có )?(tai nạn|va chạm)\b")
MINOR = re.compile(r"\b(\d{1,2} tuổi|chưa thành niên|trẻ em|học sinh)\b")
EXPLICIT_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b|\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
OTHER_TIME = re.compile(
    r"\b(năm|tháng|trước|sau|từ)\s+20\d{2}\b|\b(năm sau|năm ngoái|hôm qua|ngày mai)\b|"
    r"\b\d{1,2}[/-]\d{1,2}\b|\btháng \d{1,2}\b"
)
CURRENT = re.compile(r"\b(hiện nay|hiện hành|hôm nay|mới nhất|bây giờ)\b")

FOLLOW_UP = re.compile(r"^(tôi (đi|lái|chạy) |mình (đi|lái|chạy) |dạ |còn |thế |vậy |đi |lái )?(xe |ô tô|mô tô|đi bộ)")
SWITCH_CUE = re.compile(r"(?:^(?:còn|thế|vậy|nếu|trường hợp)\b|\bthì sao\b)")
FACT_REPLY = re.compile(
    r"^(có\b|vâng\b|đúng\b|phải\b|ừ\b|ok\b|(xe |tôi |mình )?(vừa |đang )?(chạy|di chuyển|lưu thông|cầm|dùng)\b|"
    r"không\b|xe đã đỗ\b)"
)
NEGATIVE_FACT_REPLY = re.compile(
    r"^(không\b|chưa\b|không phải\b|xe đã (đỗ|dừng)\b|đang (đỗ|dừng)\b|dùng tai nghe\b|rảnh tay\b)"
)
POSITIVE_FACT_ONLY = re.compile(r"^(có|vâng|đúng|đúng rồi|đúng vậy|phải|phải rồi|có ạ|vâng ạ|ừ)\b[.!?]*$")

PENALTY_CUE = re.compile(r"\b(phạt|chế tài|án phạt|mức xử lý hành chính|tước|tịch thu|tạm giữ|đình chỉ|cảnh cáo)\b")
MONEY_REQUEST = re.compile(r"\b((mất|nộp|đóng|tốn|hết) (bao nhiêu|mấy)|bao nhiêu (tiền|đồng)|trừ bao nhiêu điểm)\b")
OTHER_SANCTION = re.compile(r"\b(tước|tịch thu|tạm giữ|đình chỉ|cảnh cáo)\b")
POINTS_REQUEST = re.compile(
    r"\b(?:có\s+)?(?:bị\s+)?trừ(?:\s+(?:bao nhiêu|mấy))?\s*điểm\b|\bmất(?:\s+(?:bao nhiêu|mấy))?\s*điểm\b"
)
SOURCE_REQUEST = re.compile(
    r"\b(căn cứ|điều khoản|điều mấy|khoản nào|khoản mấy|điểm nào|nghị định nào|luật nào|thông tư nào|"
    r"văn bản nào|trích dẫn|dẫn nguồn|nguồn tham khảo|nguồn nào|nguồn ở đâu|link|url|sửa đổi|hợp nhất|"
    r"hiệu lực|mốc áp dụng|quy định tại đâu|ở điều nào|theo điều nào|thuộc điều nào|"
    r"(?:cho|xin|xem|nêu|kèm|cung cấp)\s+(?:nguồn|căn cứ|điều khoản|trích dẫn))\b"
)

UNVERIFIED_AMOUNT = re.compile(
    r"(?:\d|(?<!đầu\s)\b(?:một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười|vài|trăm|nghìn|ngàn)\b)"
    r"[\d\s.,–—-]*(?:triệu|nghìn|ngàn|tỷ|vnd|vnđ)"
    r"|\d[\d\s.,–—-]*(?:đồng\b|đ\b|k\b|tr\b)"
    r"|\b(?:bị\s+)?trừ\s+(?:từ\s+\d+\s+đến\s+)?(?:\d+|một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười)\s*điểm\b"
    r"|\b(?:bị\s+)?phạt\b[^\n.!?]{0,45}(?<!năm\s)(?<!ngày\s)(?<!tháng\s)(?<!điều\s)(?<!khoản\s)(?<!số\s)(?<!nghị\sđịnh\s)(?<!luật\s)\d+"
    r"|\btước\s+(?:quyền\s+sử\s+dụng|bằng|giấy\s+phép)[^\n.!?]{0,45}\d+"
    r"|[罚罰扣][^\n]{0,25}\d"
)

ARTICLE_REF = re.compile(r"(?:^|[\s,(.;:])điều\s+(\d+)\b(?!\s*(?:bánh|km|triệu|nghìn|ngàn|đồng))")
CLAUSE_REF = re.compile(r"\bkhoản\s+(\d+)\b")
REPEALED_DOC = re.compile(
    r"\b(luật giao thông đường bộ(?: năm)? 2008|23/2008/qh12|"
    r"nghị định(?: số)? 100/2019(?:/nđ-cp)?|nghị định(?: số)? 123/2021(?:/nđ-cp)?)\b"
)
DEFINITION_CUE = re.compile(r"\b(là gì|được hiểu như thế nào|khái niệm|mô tả|giải thích từ ngữ|định nghĩa)\b")
ANAPHORA = re.compile(
    r"(?:^(?:nếu|còn|thế|vậy|trường hợp|khi đó|vậy thì|thế thì)\b|"
    r"\b(?:đó|này|như vậy|vừa nêu|(?:nêu|kể|ở|bên|như) trên|thì sao)\b)"
)
STOPWORDS = frozenset({
    "là", "gì", "của", "và", "hoặc", "theo", "quy", "định", "tại", "về", "cho", "trong",
    "như", "thế", "nào", "khi", "nếu", "thì", "có", "không", "được", "bị", "các", "một",
    "những", "với", "từ", "đến", "trên", "dưới", "này", "đó", "nào", "bao", "nhiêu",
    "trường", "hợp", "việc", "văn", "bản", "số", "mục", "chương",
})

CLARIFY_ACCIDENT = (
    "Bạn vui lòng nêu rõ hành vi vi phạm dẫn đến tai nạn (ví dụ vượt đèn đỏ, đi ngược chiều, nồng độ cồn...) "
    "và hậu quả xảy ra để tôi kiểm tra căn cứ phù hợp. Hệ thống hiện chưa có dữ liệu mức phạt cho trường hợp có tai nạn."
)
CLARIFY_VIOLATION = (
    "Bạn muốn hỏi mức phạt cho hành vi vi phạm cụ thể nào (ví dụ vượt đèn đỏ, không đội mũ bảo hiểm, đi ngược chiều...)?"
)
MISSING_DOC_CITATION = "Bạn vui lòng cho biết tên văn bản pháp luật. Tôi chưa xác minh được điều khoản này."

EXTRA_VEHICLE = re.compile(
    r"\bxe\s+(?:ô\s*tô(?:\s+(?:con|tải|khách))?|máy(?:\s+điện|\s+chuyên\s+dùng)?|"
    r"gắn\s+máy|mô\s*tô|đạp(?:\s+máy|\s+điện)?|điện|tải|buýt|khách|con|bán\s+tải|đầu\s+kéo|"
    r"bốn\s+bánh|4\s+bánh|ba\s+bánh|3\s+bánh)\b"
)
VEHICLE_FOLLOWUP_TOKENS = frozenset({
    "tôi", "mình", "em", "anh", "chị", "bạn", "người", "lái", "điều", "khiển", "chạy", "đi",
    "dạ", "vâng", "ạ", "à", "còn", "thế", "vậy", "thì", "nếu", "là", "bằng", "loại",
    "phương", "tiện", "chiếc", "con", "sao", "như", "nào", "ra", "phạt", "bị", "xử", "mức",
    "bao", "nhiêu", "tiền", "mấy", "đồng", "trừ", "điểm", "hỏi", "về", "cho", "hả", "nhỉ",
    "thôi", "nhé", "với", "trường", "hợp", "khi", "thử", "xem",
})
GENERIC_NON_VIOLATION_TOKENS = frozenset({
    "xe", "người", "lái", "điều", "khiển", "đi", "chạy", "đang", "vừa", "lúc", "tham",
    "gia", "giao", "thông", "đường", "bộ", "tại", "ở", "việt", "nam", "khi", "nếu", "mà",
    "thì", "bị", "xử", "phạt", "hình", "thức", "mức", "bao", "nhiêu", "tiền", "mấy", "điểm",
    "thế", "nào", "như", "ra", "sao", "vậy", "ạ", "có", "phải", "không", "cần", "gì",
    "cho", "hỏi", "về", "hành", "vi", "lỗi", "phạm", "tai", "nạn", "va", "chạm", "gây",
    "xảy", "làm", "dẫn", "đến", "tới", "hậu", "quả", "vụ", "lần", "và", "hay", "hoặc",
    "cùng", "với", "trong", "trên", "mất", "nộp", "đóng", "tốn", "hết", "đồng", "áp",
    "dụng", "hiện", "nay", "mới", "nhất", "cụ", "thể",
})
INLINE_CIT_RE = re.compile(r"\s*\((?:điểm|khoản|điều)\b[^)]*\)", re.IGNORECASE)
SOURCE_PREFIX_RE = re.compile(
    r"^(?:theo\s+)?(?:quy định tại\s+)?(?:điểm\s+[a-zđ]\s+)?(?:khoản\s+\d+(?:,\s*\d+)*(?:\s+và\s+\d+)?\s+)?"
    r"(?:điều\s+\d+|luật\s+|nghị định\s+|thông tư\s+|văn bản\s+)"
    r"(?:(?:luật\s+trật\s+tự,\s*an\s+toàn\s+giao\s+thông\s+đường\s+bộ|[^,:\n]){0,95})"
    r"(?:,\s*quy định như sau[.:]?|[:,])\s*",
    re.IGNORECASE,
)


def _strip_inline_citations(text):
    cleaned = INLINE_CIT_RE.sub("", text)
    return re.sub(r"\s+([.,;:])", r"\1", cleaned).strip()


def _strip_source_prefix(text):
    cleaned = SOURCE_PREFIX_RE.sub("", text.strip())
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


@dataclass(frozen=True)
class Decision:
    kind: str
    text: str
    rule: str | None = None
    citation: str | None = None
    url: str | None = None
    amendment_url: str | None = None
    target_date: date | None = None
    show_source: bool = False

    def render(self, *, include_source=None):
        show = self.show_source if include_source is None else include_source
        if not show or not self.url:
            return self.text
        lines = [self.text]
        if self.target_date:
            lines.append(f"Mốc áp dụng dùng để tra cứu: {self.target_date:%d/%m/%Y}.")
        lines.append(f"Căn cứ: {self.citation}.")
        lines.append(f"Nguồn: {self.url}")
        if self.amendment_url:
            lines.append(f"Văn bản sửa đổi hoặc hợp nhất đã đối chiếu: {self.amendment_url}")
        return "\n".join(lines)


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def sanction_intent(text, *, road_related=False):
    return bool(SANCTION.search(text) or PENALTY_CUE.search(text) or (road_related and MONEY_REQUEST.search(text)))


def unverified_sanction_output(text):
    """Detect common monetary and point claims in the ungrounded model fallback."""
    return bool(UNVERIFIED_AMOUNT.search(normalize(text)))


def official_url(url):
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and (
        host == "chinhphu.vn" or host.endswith(".chinhphu.vn")
        or host == "vbpl.vn" or host.endswith(".vbpl.vn")
    )


def match_all(patterns, text):
    return all(re.search(pattern, text) for pattern in patterns)


def match_any(patterns, text):
    return any(re.search(pattern, text) for pattern in patterns)


def question_date(text, verified_through, today):
    matches = list(EXPLICIT_DATE.finditer(text))
    if len(matches) > 1:
        return None
    if matches:
        year, month, day, dmy_day, dmy_month, dmy_year = matches[0].groups()
        try:
            return date(int(year or dmy_year), int(month or dmy_month), int(day or dmy_day))
        except ValueError:
            return None
    if OTHER_TIME.search(text):
        return None
    return today if CURRENT.search(text) else verified_through


def vehicle_groups(question):
    groups = set()
    if MOTO.search(EXTRA_MOTO.sub("", question)):
        groups.add("moto")
    if CAR.search(question):
        groups.add("car")
    if BICYCLE.search(question):
        groups.add("bicycle")
    if SPECIAL.search(question):
        groups.add("special")
    if OTHER_VEHICLE.search(question) or (not groups and VEHICLE.search(question)):
        groups.add("other")
    return groups


QUERY_SYNONYMS = {
    "xe_máy": ("mô_tô", "gắn_máy"),
    "bằng_lái": ("giấy", "phép", "lái", "xe"),
    "vỉa_hè": ("hè", "phố"),
    "đèn_đỏ": ("tín", "hiệu", "đèn"),
}


def _tokenize_search(text):
    tokens = re.findall(r"[0-9a-zA-Zà-ỹÀ-Ỹ]+", normalize(text))
    filtered = [tok for tok in tokens if (tok not in STOPWORDS and len(tok) > 1) or tok.isdigit()]
    expanded = list(filtered)
    bigrams = []
    for i in range(len(filtered) - 1):
        pair = f"{filtered[i]}_{filtered[i + 1]}"
        if pair == "xe_máy" and i + 2 < len(filtered) and filtered[i + 2] == "chuyên":
            bigrams.append("xe_máy_chuyên")
            continue
        bigrams.append(pair)
        if pair in QUERY_SYNONYMS:
            expanded.extend(QUERY_SYNONYMS[pair])
    return expanded + bigrams


def _trim_excerpt(text, max_chars=750):
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars].rsplit(" ", 1)[0]
    return f"{cut}..."


class LegalIndex:
    def __init__(self, path=SOURCE_FILE, corpus_path=CORPUS_FILE):
        with path.open("rb") as source:
            self.data = tomllib.load(source)
        self.verified_through = self.data["verified_through"]
        for record in (*self.data["rules"], *self.data["provisions"]):
            doc = self.data["documents"][record["document"]]
            if not record["citation"].strip() or not official_url(doc["url"]):
                raise ValueError(f"Missing official citation for {record['id']}")
            if doc.get("amendment_url") and not official_url(doc["amendment_url"]):
                raise ValueError(f"Invalid amendment source for {record['id']}")

        self._records_by_id = {r["id"]: r for r in (*self.data["rules"], *self.data["provisions"])}

        self._doc_aliases = []
        for doc_key, doc in self.data["documents"].items():
            for alias in doc.get("aliases", ()):
                self._doc_aliases.append((normalize(alias), doc_key, doc))
        self._doc_aliases.sort(key=lambda item: len(item[0]), reverse=True)

        self._corpus = []
        self._doc_freqs = Counter()
        self._avg_len = 1.0
        self._build_retrieval_index(corpus_path)

    def _build_retrieval_index(self, corpus_path):
        entries = []
        for item in self.data.get("definitions", ()):
            entries.append({
                "id": item["id"],
                "document_id": self.data["documents"][item["document"]]["number"],
                "article": "2" if "Điều 2 " in item["citation"] else "",
                "clause": "",
                "citation": item["citation"],
                "title": item["title"],
                "text": item["text"],
                "is_definition": "là gì" in normalize(item["title"]) or "Điều 2 " in item["citation"],
                "url": self.data["documents"][item["document"]]["url"],
            })
        if corpus_path and Path(corpus_path).exists():
            with Path(corpus_path).open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("verification_status") != "verified":
                        continue
                    clause = str(row.get("clause") or "").strip()
                    article = str(row.get("article") or "").strip()
                    doc_id = str(row.get("document_id") or "").strip()
                    citation = (
                        f"khoản {clause} Điều {article} {doc_id}"
                        if clause and clause != "0"
                        else f"Điều {article} {doc_id}"
                    )
                    entries.append({
                        "id": row["id"],
                        "document_id": doc_id,
                        "article": article,
                        "clause": clause,
                        "citation": citation,
                        "title": row.get("article_title", ""),
                        "text": row.get("source_text", ""),
                        "is_definition": article == "2" and normalize(row.get("article_title", "")) == "giải thích từ ngữ",
                        "url": row.get("source_url", ""),
                    })

        for entry in entries:
            search_text = f"{entry['title']} {entry['title']} {entry['citation']} {entry['text']}"
            tf = Counter(_tokenize_search(search_text))
            entry["_tf"] = tf
            entry["_len"] = sum(tf.values())
            for term in tf:
                self._doc_freqs[term] += 1
        self._corpus = entries
        if entries:
            self._avg_len = sum(e["_len"] for e in entries) / len(entries) or 1.0

    def _match_documents_in_text(self, normalized_text):
        found = []
        used_spans = []
        for alias, doc_key, doc in self._doc_aliases:
            for m in re.finditer(re.escape(alias), normalized_text):
                start, end = m.span()
                if any(s <= start and end <= e for s, e in used_spans):
                    continue
                used_spans.append((start, end))
                found.append((start, end, doc_key, doc))
        found.sort(key=lambda item: item[0])
        return found

    def _wants_source(self, text):
        return bool(SOURCE_REQUEST.search(normalize(text)))

    def check_citation(self, text):
        normalized = normalize(text)
        want_source = self._wants_source(text)
        if REPEALED_DOC.search(normalized):
            if want_source:
                return (
                    "Văn bản được nêu đã hết hiệu lực đối với lĩnh vực trật tự, an toàn giao thông đường bộ từ ngày 01/01/2025, "
                    "được thay thế bởi Luật 36/2024/QH15, Luật 35/2024/QH15 và Nghị định 168/2024/NĐ-CP."
                )
            return "Văn bản được nêu đã hết hiệu lực đối với lĩnh vực trật tự, an toàn giao thông đường bộ từ ngày 01/01/2025."

        articles = list(ARTICLE_REF.finditer(normalized))
        clauses = list(CLAUSE_REF.finditer(normalized))
        docs = self._match_documents_in_text(normalized)
        max_in_scope = max((d.get("max_article", 89) for d in self.data["documents"].values()), default=89)

        for match in articles:
            art_num = int(match.group(1))
            target_doc = None
            if len(docs) == 1:
                target_doc = docs[0][3]
            elif docs:
                after = [d for d in docs if d[0] >= match.end()]
                target_doc = after[0][3] if after else min(docs, key=lambda d: abs(d[0] - match.start()))[3]

            if art_num <= 0:
                return MISSING_DOC_CITATION if target_doc is None else f"Không tồn tại Điều {art_num}."

            if target_doc is not None and "max_article" in target_doc:
                max_art = target_doc["max_article"]
                if art_num > max_art:
                    doc_label = target_doc["title"] if len(target_doc["title"]) <= 60 else target_doc["number"]
                    return f"{doc_label} chỉ có {max_art} điều, nên không có Điều {art_num}."
            elif art_num > max_in_scope:
                return MISSING_DOC_CITATION

        for match in clauses:
            clause_num = int(match.group(1))
            if clause_num <= 0 or clause_num > 25:
                return MISSING_DOC_CITATION if not docs else f"Không tìm thấy khoản {clause_num} trong văn bản được nêu."
        return None

    def _is_clarification_message(self, assistant_content):
        if CLARIFY_VEHICLE in assistant_content:
            return "vehicle"
        if CLARIFY_VIOLATION in assistant_content:
            return "violation"
        if CLARIFY_ACCIDENT in assistant_content:
            return "accident"
        for topic in self.data["topics"].values():
            clar = topic.get("clarification")
            if clar and clar in assistant_content:
                return "facts"
        return None

    def _collect_clarification_chain(self, messages):
        chain_questions = [normalize(messages[-1]["content"])]
        clar_kind = self._is_clarification_message(messages[-2]["content"])
        if not clar_kind:
            return chain_questions, None
        idx = len(messages) - 3
        while idx >= 0 and messages[idx]["role"] == "user":
            chain_questions.append(normalize(messages[idx]["content"]))
            if idx - 1 >= 0 and messages[idx - 1]["role"] == "assistant" and self._is_clarification_message(messages[idx - 1]["content"]):
                idx -= 2
            else:
                break
        chain_questions.reverse()
        return chain_questions, clar_kind

    def _strip_vehicles(self, text):
        cleaned = EXTRA_VEHICLE.sub(" ", text)
        for pat in (EXTRA_MOTO, MOTO, CAR, BICYCLE, SPECIAL, OTHER_VEHICLE, VEHICLE):
            cleaned = pat.sub(" ", cleaned)
        return " ".join(cleaned.split())

    def _is_vehicle_followup(self, question, *, require_switch=False):
        if DEFINITION_CUE.search(question) or ACCIDENT.search(question):
            return False
        if any(match_any(t["matches"], question) for t in self.data["topics"].values()):
            return False
        if any(match_all(r["matches"], question) and not match_any(r.get("exclude", ()), question) for r in self.data["provisions"]):
            return False
        if not (vehicle_groups(question) or SPECIAL.search(question)):
            return False
        remainder = self._strip_vehicles(question)
        tokens = re.findall(r"[0-9a-zA-Zà-ỹÀ-Ỹ]+", remainder)
        if not all(tok in VEHICLE_FOLLOWUP_TOKENS for tok in tokens):
            return False
        if require_switch:
            return bool(
                SWITCH_CUE.search(question)
                or MONEY_REQUEST.search(question)
                or SANCTION.search(question)
                or PENALTY_CUE.search(question)
            )
        return True

    def _classify_missing_facts(self, question):
        stripped = self._strip_vehicles(question)
        for pat in (SANCTION, PENALTY_CUE, MONEY_REQUEST, OTHER_SANCTION, POINTS_REQUEST):
            stripped = pat.sub(" ", stripped)
        tokens = re.findall(r"[0-9a-zA-Zà-ỹÀ-Ỹ]+", stripped)
        meaningful = [tok for tok in tokens if tok not in GENERIC_NON_VIOLATION_TOKENS]
        if meaningful:
            return None
        if ACCIDENT.search(NO_ACCIDENT.sub("", question)):
            return CLARIFY_ACCIDENT
        return CLARIFY_VIOLATION

    def _with_citation_warning(self, decision, citation_warning):
        if not citation_warning or decision is None:
            return decision
        return Decision(
            decision.kind,
            f"{citation_warning}\n\n{decision.text}",
            decision.rule,
            decision.citation,
            decision.url,
            decision.amendment_url,
            decision.target_date,
            decision.show_source,
        )

    def resolve(self, messages, *, today=None):
        question = normalize(messages[-1]["content"])
        want_source = self._wants_source(question)
        want_points = bool(POINTS_REQUEST.search(question))
        outside_topic = OUTSIDE.search(question) and not (ROAD_CONTEXT.search(question) or VEHICLE.search(question))
        outside_country = FOREIGN.search(question) and not VIETNAM.search(question)
        if outside_topic or outside_country:
            return Decision("outside", "Câu hỏi này nằm ngoài phạm vi pháp luật giao thông đường bộ Việt Nam.")

        bare_affirmative = False
        if len(messages) >= 3 and messages[-2]["role"] == "assistant" and messages[-3]["role"] == "user" and not (sanction_intent(question) and any(
            match_any(topic["matches"], question) for topic in self.data["topics"].values()
        )):
            previous_reply = messages[-2]["content"]
            clar_kind = self._is_clarification_message(previous_reply)
            if clar_kind == "vehicle":
                if self._is_vehicle_followup(question, require_switch=False) and len(question.split()) <= 15:
                    chain, _ = self._collect_clarification_chain(messages)
                    question = " ".join(chain)
            elif clar_kind == "facts":
                if NEGATIVE_FACT_REPLY.search(question) and not POSITIVE_FACT_ONLY.search(question):
                    return Decision("no_source", NO_SANCTION_SOURCE)
                if FACT_REPLY.search(question) or POSITIVE_FACT_ONLY.search(question):
                    bare_affirmative = bool(POSITIVE_FACT_ONLY.search(question))
                    chain, _ = self._collect_clarification_chain(messages)
                    question = " ".join(chain)
            elif clar_kind in ("violation", "accident"):
                if not DEFINITION_CUE.search(question) and (
                    any(match_any(t["matches"], question) for t in self.data["topics"].values())
                    or sanction_intent(question, road_related=True)
                    or len(question.split()) <= 10
                ):
                    chain, _ = self._collect_clarification_chain(messages)
                    question = " ".join(chain)
            else:
                prev_decision = self.resolve(messages[:-2], today=today) if NO_SANCTION_SOURCE not in previous_reply else None
                grounded_prev = (
                    (prev_decision is not None and prev_decision.kind == "grounded")
                    or ("Căn cứ:" in previous_reply and "Mốc áp dụng dùng để tra cứu:" in previous_reply)
                )
                is_detail_followup = (
                    grounded_prev
                    and (want_points or want_source)
                    and not vehicle_groups(question)
                    and not SPECIAL.search(question)
                    and not any(match_any(t["matches"], question) for t in self.data["topics"].values())
                    and not DEFINITION_CUE.search(question)
                    and len(question.split()) <= 14
                )
                if is_detail_followup and prev_decision is not None and prev_decision.rule in self._records_by_id:
                    rule = self._records_by_id[prev_decision.rule]
                    if want_points and "points" not in rule:
                        return Decision("no_source", NO_SANCTION_SOURCE)
                    ref_date = prev_decision.target_date or (today or self.verified_through)
                    rendered = self.render_rule(
                        rule,
                        ref_date,
                        include_points=want_points or want_source,
                        show_source=want_source,
                    )
                    return rendered
                if grounded_prev and self._is_vehicle_followup(question, require_switch=True):
                    prior_topic_text = ""
                    for m in reversed(messages[:-1]):
                        if m["role"] == "user":
                            norm_u = normalize(m["content"])
                            if any(match_any(t["matches"], norm_u) for t in self.data["topics"].values()):
                                prior_topic_text = self._strip_vehicles(norm_u)
                                break
                    if prior_topic_text:
                        question = f"{prior_topic_text} {question}".strip()
                    else:
                        return Decision("no_source", NO_SANCTION_SOURCE)
                elif sanction_intent(normalize(messages[-3]["content"]), road_related=True) and (
                    (
                        not vehicle_groups(question)
                        and not SPECIAL.search(question)
                        and not DEFINITION_CUE.search(question)
                        and not ACCIDENT.search(question)
                        and (
                            SWITCH_CUE.search(question)
                            or re.fullmatch(r"(?:(?:phạt|mất|nộp)\s+)?(?:bao nhiêu|mấy)(?:\s+(?:tiền|điểm))?[?!.]*", question)
                        )
                    )
                    or self._is_vehicle_followup(question, require_switch=True)
                ):
                    return Decision("no_source", NO_SANCTION_SOURCE)

        citation_warning = self.check_citation(question)
        target_date = question_date(question, self.verified_through, today or datetime.now(VIETNAM_TIME).date())
        topics = [name for name, topic in self.data["topics"].items() if match_any(topic["matches"], question)]
        vehicles = vehicle_groups(question)
        road_related = bool(topics or vehicles or ROAD_CONTEXT.search(question))

        if sanction_intent(question, road_related=road_related):
            if not road_related:
                return Decision("no_source", NO_SANCTION_SOURCE)
            if len(topics) > 1:
                filtered_topics = [
                    name for name in topics
                    if not match_any(self.data["topics"][name].get("exclude", ()), question)
                ]
                if len(filtered_topics) == 1:
                    topics = filtered_topics
            if len(vehicles) != 1:
                return self._with_citation_warning(Decision("clarify", CLARIFY_VEHICLE), citation_warning)
            if len(topics) == 0:
                if clarify_msg := self._classify_missing_facts(question):
                    return self._with_citation_warning(Decision("clarify", clarify_msg), citation_warning)
                return self._with_citation_warning(Decision("no_source", NO_SANCTION_SOURCE), citation_warning)
            if len(topics) > 1:
                return self._with_citation_warning(Decision("no_source", NO_SANCTION_SOURCE), citation_warning)
            topic = self.data["topics"][topics[0]]
            if (
                match_any(topic.get("exclude", ()), question)
                or ACCIDENT.search(NO_ACCIDENT.sub("", question))
                or MINOR.search(question)
                or FOREIGN.search(question)
            ):
                return self._with_citation_warning(Decision("no_source", NO_SANCTION_SOURCE), citation_warning)
            if not (bare_affirmative or match_all(topic.get("facts", ()), question)):
                return self._with_citation_warning(Decision("clarify", topic["clarification"]), citation_warning)
            candidates = [r for r in self.data["rules"] if r["topic"] == topics[0] and r["vehicle"] in vehicles]
            if OTHER_SANCTION.search(question) and not any(r.get("supports_other_sanction") for r in candidates):
                return self._with_citation_warning(Decision("no_source", NO_SANCTION_SOURCE), citation_warning)
        else:
            candidates = [
                r for r in self.data["provisions"]
                if match_all(r["matches"], question) and not match_any(r.get("exclude", ()), question)
            ]
            if not candidates:
                if citation_warning:
                    return Decision("invalid_citation", citation_warning)
                return None

        matches = [
            r for r in candidates
            if target_date and r["effective_from"] <= target_date <= self.verified_through
            and (not r.get("effective_to") or target_date < r["effective_to"])
        ]
        if len(matches) != 1:
            msg = (
                f"Chưa có căn cứ đã đối chiếu cho tình huống hoặc thời điểm này. Mốc rà soát nguồn: {self.verified_through:%d/%m/%Y}."
                if want_source
                else "Chưa có căn cứ đã đối chiếu cho tình huống hoặc thời điểm này."
            )
            return self._with_citation_warning(Decision("no_source", msg), citation_warning)
        if want_points and "points" not in matches[0]:
            return self._with_citation_warning(Decision("no_source", NO_SANCTION_SOURCE), citation_warning)
        decision = self.render_rule(
            matches[0],
            target_date,
            include_points=want_points,
            show_source=want_source,
        )
        return self._with_citation_warning(decision, citation_warning)

    def render_rule(self, rule, target_date, *, include_points=False, show_source=False):
        doc = self.data["documents"][rule["document"]]
        text = rule.get("fact")
        if text and not show_source:
            text = _strip_inline_citations(text)
        if "fine_vnd" in rule:
            condition = self.data["topics"][rule["topic"]]["condition"].lower()
            low, high = (f"{amount:,}".replace(",", ".") for amount in rule["fine_vnd"])
            text = f"Người điều khiển {rule['vehicle_label']} {condition} bị phạt từ {low} đến {high} đồng"
            if (include_points or show_source) and "points" in rule:
                points = rule["points"]
                if points > 0:
                    text += f" và trừ {points} điểm giấy phép lái xe"
                    if rule.get("points_condition"):
                        text += f" {rule['points_condition']}"
                else:
                    text += " (không áp dụng trừ điểm giấy phép lái xe)"
            text += "."
        return Decision(
            "grounded",
            text,
            rule["id"],
            f"{rule['citation']} {doc['number']}",
            doc["url"],
            doc.get("amendment_url"),
            target_date=target_date,
            show_source=show_source,
        )

    def _extract_definition_subject(self, normalized_query):
        cleaned = normalized_query
        if m := SOURCE_REQUEST.search(cleaned):
            prefix = re.sub(
                r"(?:,?\s*(?:và|kèm|với|đồng thời|cho|xin|nêu|trích\s+dẫn|cung\s+cấp|biết|xem|rõ|tôi|mình|em))+\s*$",
                "",
                cleaned[:m.start()],
            )
            cleaned = prefix if prefix.strip() else cleaned[m.end():]
        for alias, _, _ in self._doc_aliases:
            cleaned = re.sub(
                rf"\b(?:theo|tại|trong|của)\s+(?:quy định(?:\s+(?:của|tại))?\s+)?{re.escape(alias)}\b",
                " ",
                cleaned,
            )
        cleaned = DEFINITION_CUE.sub(" ", cleaned)
        cleaned = re.sub(r"[?!.,:;()\"'“”]+", " ", cleaned)
        cleaned = re.sub(
            r"^(?:cho\s+(?:tôi|mình|em|biết|hỏi)|xin\s+hỏi|hãy|vui\s+lòng|trong\s+luật\s+giao\s+thông)\s+",
            "",
            " ".join(cleaned.split()),
        )
        return cleaned.strip()

    def _definition_matches_subject(self, subject, entry):
        if not subject:
            return entry["is_definition"]
        variants = [subject]
        if subject == "xe máy":
            variants.extend(("xe mô tô", "xe gắn máy"))
        norm_title = normalize(re.sub(r"\s+là gì$", "", entry["title"]))
        norm_text = normalize(entry["text"])
        for term in variants:
            if entry["is_definition"] and term in norm_title:
                return True
            if re.search(rf"(?:^|[.,;:])\s*(?:theo\s+[^,]+,\s*)?{re.escape(term)}[^.:;\n]{{0,25}}\s+(?:là|bao gồm|gồm|thuộc)\b", norm_text):
                return True
        return False

    def retrieve(self, query, *, top_k=2):
        if not self._corpus:
            return []
        normalized_query = normalize(query)
        q_tokens = _tokenize_search(normalized_query)
        if not q_tokens:
            return []
        n_docs = len(self._corpus)
        k1 = 1.5
        b = 0.75
        art_matches = {m.group(1) for m in ARTICLE_REF.finditer(normalized_query)}
        clause_matches = {m.group(1) for m in CLAUSE_REF.finditer(normalized_query)}
        doc_matches = {doc["number"] for _, _, _, doc in self._match_documents_in_text(normalized_query)}
        is_def_query = bool(DEFINITION_CUE.search(normalized_query))

        scored = []
        q_term_counts = Counter(q_tokens)
        for entry in self._corpus:
            tf = entry["_tf"]
            score = 0.0
            doc_len = entry["_len"] or 1
            for term, q_freq in q_term_counts.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                df = self._doc_freqs.get(term, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1.0 - b + b * doc_len / self._avg_len)
                score += idf * (freq * (k1 + 1.0) / denom) * min(q_freq, 2)

            if art_matches and entry["article"] in art_matches:
                score += 6.0
                if doc_matches and entry["document_id"] in doc_matches:
                    score += 4.0
                if clause_matches and entry["clause"] in clause_matches:
                    score += 5.0
            if is_def_query and entry["is_definition"]:
                score += 4.0
                if normalize(entry["title"].replace(" là gì", "")) in normalized_query:
                    score += 6.0
                elif "xe máy" in normalized_query and entry["id"] in ("def-55-34-1-xe-mo-to", "def-55-34-1-xe-gan-may"):
                    score += 6.0

            if score >= 3.0:
                scored.append((score, entry))

        if is_def_query and not art_matches:
            subject = self._extract_definition_subject(normalized_query)
            if subject:
                scored = [(sc, entry) for sc, entry in scored if self._definition_matches_subject(subject, entry)]

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen_citations = set()
        top_score = scored[0][0] if scored else 0.0
        top_is_def = bool(scored and scored[0][1]["is_definition"])
        for sc, entry in scored:
            cit_key = re.sub(r"\bluật\s+", "", normalize(entry["citation"]))
            if cit_key in seen_citations:
                continue
            if selected:
                if sc < top_score * 0.72:
                    continue
                if is_def_query and top_is_def and not entry["is_definition"]:
                    continue
            seen_citations.add(cit_key)
            selected.append(entry)
            if len(selected) >= top_k:
                break
        return selected

    def augment_messages(self, messages, *, top_k=2, tokenizer=None, context_limit=1024):
        if not messages:
            return messages
        user_turns = [normalize(m["content"]) for m in messages if m["role"] == "user"]
        if not user_turns:
            return messages
        current = user_turns[-1]
        want_source = self._wants_source(messages[-1]["content"])
        is_source_followup = (
            len(user_turns) >= 2
            and want_source
            and len(current.split()) <= 12
            and not any(match_any(t["matches"], current) for t in self.data["topics"].values())
            and not DEFINITION_CUE.search(current)
        )
        is_followup = len(user_turns) >= 2 and (
            bool(ANAPHORA.search(current))
            or is_source_followup
            or (len(current.split()) <= 3 and not DEFINITION_CUE.search(current))
        )
        query = f"{user_turns[-2]} {current}" if is_followup else current
        isolated = (
            [dict(messages[0]), dict(messages[-1])]
            if (not is_followup and len(messages) > 2)
            else None
        )

        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            if isolated is not None:
                return isolated if messages[0]["role"] == "system" else [dict(messages[-1])]
            return messages

        def build_with_hits(selected_hits, max_chars):
            sys_idx = 0 if messages[0]["role"] == "system" else -1
            base_system = messages[sys_idx]["content"] if sys_idx == 0 else ""
            if want_source:
                blocks = [f"[{hit['citation']}] {_trim_excerpt(hit['text'], max_chars=max_chars)}" for hit in selected_hits]
                rag_instruction = (
                    "\n\n[Căn cứ pháp lý tham chiếu đã kiểm chứng]\n"
                    + "\n".join(blocks)
                    + "\nHướng dẫn: Nếu căn cứ trên trả lời trực tiếp câu hỏi, hãy bám sát đúng câu chữ của căn cứ đó và ghi đúng Điều, khoản, số hiệu văn bản từ phần [Căn cứ]. Không bịa thêm số điều hoặc mức phạt."
                )
            else:
                blocks = [_trim_excerpt(_strip_source_prefix(hit["text"]), max_chars=max_chars) for hit in selected_hits]
                rag_instruction = (
                    "\n\n[Thông tin tham chiếu đã kiểm chứng]\n"
                    + "\n".join(blocks)
                    + "\nHướng dẫn: Dùng đúng nội dung thông tin trên để trả lời ngắn gọn (1-2 câu), đúng trọng tâm; nếu hỏi có được phép hay không thì trả lời Có hoặc Không trước rồi giải thích ngắn gọn. Không tự động nêu số điều, khoản, tên văn bản hoặc nguồn tham khảo."
                )
            augmented = [dict(m) for m in messages]
            if sys_idx == 0:
                augmented[0]["content"] = base_system + rag_instruction
            else:
                augmented.insert(0, {"role": "system", "content": rag_instruction.strip()})
            if not is_followup and len(augmented) > 2:
                augmented = [augmented[0], dict(messages[-1])]
            if want_source:
                citations_str = ", ".join(h["citation"] for h in selected_hits)
                refs_str = " ".join(_trim_excerpt(h["text"], max_chars=max_chars) for h in selected_hits)
                suffix = f"Căn cứ tham chiếu ({citations_str}): {refs_str}"
            else:
                refs_str = " ".join(_trim_excerpt(_strip_source_prefix(h["text"]), max_chars=max_chars) for h in selected_hits)
                suffix = f"Thông tin tham chiếu: {refs_str}"
            augmented[-1] = {
                "role": "user",
                "content": f"{messages[-1]['content']}\n\n{suffix}",
            }
            return augmented

        for char_limit in (700, 420, 260):
            for count in (len(hits), 1):
                candidate = build_with_hits(hits[:count], char_limit)
                if tokenizer is None:
                    return candidate
                toks = len(tokenizer.apply_chat_template(candidate, tokenize=True, add_generation_prompt=True))
                if toks <= context_limit - 32:
                    return candidate
        if isolated is not None:
            return isolated if messages[0]["role"] == "system" else [dict(messages[-1])]
        return messages

    def filter_output(self, messages, answer):
        if unverified_sanction_output(answer):
            return NO_SANCTION_SOURCE
        last_user = messages[-1]["content"] if messages else ""
        norm_user = normalize(last_user)
        if not (self._wants_source(last_user) or ARTICLE_REF.search(norm_user) or CLAUSE_REF.search(norm_user)):
            cleaned = _strip_source_prefix(answer)
            if cleaned:
                return cleaned
        return answer
