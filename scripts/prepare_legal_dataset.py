"""Build and verify the date-pinned Vietnamese road-law QA dataset."""

import argparse
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
import json
from pathlib import Path
from random import Random
import re
import statistics
import tomllib
import unicodedata

from datasets import load_dataset
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from legal_sources import clean, effective_date, source_file, split_articles


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
REQUIRED = {
    "id", "question", "answer", "source_url", "document_id", "article",
    "effective_date", "verification_status", "topic",
}
SYSTEM = (
    "Bạn trả lời câu hỏi về pháp luật giao thông đường bộ Việt Nam bằng tiếng Việt. "
    "Nêu căn cứ điều, khoản và số hiệu văn bản. Giữ đầy đủ điều kiện và ngoại lệ. "
    "Mốc áp dụng của bộ dữ liệu là ngày 13/11/2026."
)
NEAR_DUPLICATE = 0.80


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def lookup_question(source, heading, locator, index):
    title = heading[0].lower() + heading[1:]
    templates = (
        f"Về {title}, {locator} của {source['title']} số {source['document_id']} quy định gì?",
        f"Quy định về {title} tại {locator} văn bản số {source['document_id']} có nội dung gì?",
        f"Tra cứu nội dung {locator} văn bản số {source['document_id']} về {title} như thế nào?",
    )
    return templates[index % len(templates)]


def extract_candidates(config, raw_dir, download, authored):
    candidates = defaultdict(list)
    discarded = Counter()
    used = set()
    for source in config["sources"]:
        articles, notes = split_articles(
            source_file(source, raw_dir, download), source["last_article"]
        )
        for article, provision in articles.items():
            number = int(re.match(r"\d+", article)[0])
            if number in source["exclude"]:
                continue
            heading = provision.heading.text
            starts = [int(key) for key in source["topics"] if int(key) <= number]
            topic = source["topics"][str(max(starts))]
            for clause, passage in provision.clauses.items():
                if clause == "0" and len(provision.clauses) > 1:
                    continue
                excerpt = passage.text
                if (
                    len(excerpt) < 30
                    or not excerpt.endswith((".", ";", "?", "”"))
                    or re.search(r"\(\s*(?:được\s+)?bãi bỏ\s*\)|[\ufffd\u25a1]|[.]{3}", excerpt, re.I)
                ):
                    discarded["extraction_or_scope"] += 1
                    continue
                note_ids = provision.heading.notes | passage.notes
                effective = effective_date(source, article, clause, note_ids, notes)
                if effective is None or date.fromisoformat(effective) > date.fromisoformat(config["as_of"]):
                    discarded["uncertain_or_future_effective_date"] += 1
                    continue
                locator = f"Điều {article}" if clause == "0" else f"khoản {clause} Điều {article}"
                key = f"{source['key']}:{article}.{clause}"
                pages = sorted(passage.pages)
                question = authored.get(key)
                if question:
                    used.add(key)
                candidates[source["key"]].append({
                    "id": f"vn-road-{source['key']}-{article}-{clause}",
                    "question": clean(question or lookup_question(source, heading, locator, len(candidates[source["key"]]))),
                    "answer": f"Theo {locator} văn bản số {source['document_id']}, quy định như sau.\n\n{excerpt}",
                    "source_url": source["source_url"],
                    "document_id": source["document_id"],
                    "article": article,
                    "clause": None if clause == "0" else clause,
                    "effective_date": effective,
                    "verification_status": "verified",
                    "verification_method": "official_pdf_extractive",
                    "topic": topic,
                    "as_of_date": config["as_of"],
                    "source_edition": source["edition"],
                    "source_sha256": source["sha256"],
                    "source_pdf_url": f"{source['pdf_url']}#page={pages[0]}",
                    "pdf_pages": pages,
                    "article_title": heading,
                    "source_text": excerpt,
                    "source_footnotes": sorted(note_ids, key=int),
                    "question_type": "authored" if question else "provision_lookup",
                })
    if authored.keys() - used:
        raise ValueError(f"Authored questions without a retained provision: {sorted(authored.keys() - used)}")
    return candidates, discarded


def tokenize(row, tokenizer):
    prompt = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["question"]},
    ]
    completion = [{"role": "assistant", "content": row["answer"]}]
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(prompt + completion, tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError(f"Chat template prefix mismatch for {row['id']}")
    eos = full.index(tokenizer.eos_token_id, len(prefix))
    if eos <= len(prefix) or tokenizer.eos_token_id in full[eos + 1:]:
        raise ValueError(f"Invalid assistant turn for {row['id']}")
    return {
        "id": row["id"],
        "prompt": prompt,
        "completion": completion,
        "input_ids": full,
        "completion_mask": [0] * len(prefix) + [1] * (eos + 1 - len(prefix)) + [0] * (len(full) - eos - 1),
    }


def select_rows(config, candidates, discarded, tokenizer):
    rng = Random(config["seed"])
    selected = []
    encoded = {}
    seen_questions = set()
    seen_evidence = set()
    for source in config["sources"]:
        by_article = defaultdict(list)
        authored = []
        for row in candidates[source["key"]]:
            if row["question_type"] == "authored":
                authored.append(row)
            else:
                by_article[row["article"]].append(row)
        order = list(by_article)
        rng.shuffle(order)
        for rows in by_article.values():
            rng.shuffle(rows)
        ordered = authored + [
            by_article[article][offset]
            for offset in range(max(map(len, by_article.values()), default=0))
            for article in order
            if offset < len(by_article[article])
        ]
        count = 0
        for row in ordered:
            question_key = row["question"].casefold()
            evidence_key = row["source_text"].casefold()
            if question_key in seen_questions or evidence_key in seen_evidence:
                discarded["exact_duplicate"] += 1
                continue
            example = tokenize(row, tokenizer)
            if len(example["input_ids"]) > config["token_limit"]:
                discarded["token_limit"] += 1
                if row["question_type"] == "authored":
                    raise ValueError(f"Authored row exceeds the token budget: {row['id']}")
                continue
            row["question_words"] = len(row["question"].split())
            row["answer_words"] = len(row["answer"].split())
            row["sequence_tokens"] = len(example["input_ids"])
            row["assistant_tokens"] = sum(example["completion_mask"])
            selected.append(row)
            encoded[row["id"]] = example
            seen_questions.add(question_key)
            seen_evidence.add(evidence_key)
            count += 1
            if count == source["quota"]:
                break
        if count != source["quota"]:
            raise ValueError(f"{source['document_id']} has {count} eligible rows, target is {source['quota']}")
    if len(selected) != config["sample_target"]:
        raise ValueError("Source quotas do not match sample_target")
    return selected, encoded


def shingles(text):
    tokens = re.findall(r"\w+", text.casefold())
    return set(zip(tokens, tokens[1:], tokens[2:]))


def near_duplicate_pairs(rows):
    features = {row["id"]: shingles(row["source_text"]) for row in rows}
    for left, right in combinations(rows, 2):
        if (left["document_id"], left["article"]) == (right["document_id"], right["article"]):
            continue
        a, b = features[left["id"]], features[right["id"]]
        if min(len(a), len(b)) >= NEAR_DUPLICATE * max(len(a), len(b)) and min(len(a), len(b)) >= 3:
            if len(a & b) / len(a | b) >= NEAR_DUPLICATE:
                yield left, right


def group_rows(rows, linked_articles):
    def article_key(row):
        return f"{row['document_id']}:{row['article']}"

    parents = {article_key(row): article_key(row) for row in rows}

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    for left, right in near_duplicate_pairs(rows):
        a, b = sorted((root(article_key(left)), root(article_key(right))))
        parents[b] = a
    for linked in linked_articles:
        keys = sorted({root(key) for key in linked if key in parents})
        for key in keys[1:]:
            parents[key] = keys[0]
    groups = defaultdict(list)
    for row in rows:
        key = root(article_key(row))
        row["group_id"] = key
        groups[key].append(row)
    return groups


def subset(order, groups, size):
    ways = {0: ()}
    for key in order:
        for count, keys in list(ways.items()):
            total = count + len(groups[key])
            if total <= size and total not in ways:
                ways[total] = (*keys, key)
        if size in ways:
            return set(ways[size])
    return None


def stratification(rows):
    features = Counter()
    for row in rows:
        for field in ("topic", "document_id", "question_type"):
            features[field, row[field]] += 1
    return features


def assign_splits(rows, groups, seed):
    total_features = stratification(rows)
    priority_test_topics = ("tai_nan_giao_thong", "hanh_vi_bi_cam", "nguoi_lai_va_giay_phep")
    n = len(rows) // 10
    rng = Random(seed)
    order = sorted(groups)
    best = None
    best_score = float("inf")
    for _ in range(3000):
        rng.shuffle(order)
        test = subset(order, groups, n)
        if test is None:
            continue
        validation = subset([key for key in order if key not in test], groups, n)
        if validation is None:
            continue
        score = 0
        holdout = Counter()
        for ids in (test, validation):
            features = stratification(row for key in ids for row in groups[key])
            holdout.update(features)
            for feature, count in total_features.items():
                expected = count * n / len(rows)
                score += (features[feature] - expected) ** 2 / max(expected, 1)
                if expected >= 1 and features[feature] == 0:
                    score += 3
            score += 4 * max(0, 10 - features["question_type", "authored"]) ** 2
            if ids == test:
                score += 18 * sum(features["topic", topic] == 0 for topic in priority_test_topics)
        score += 30 * sum(
            field == "topic" and holdout[field, value] == count
            for (field, value), count in total_features.items()
        )
        if score < best_score:
            best_score = score
            best = test, validation
    if best is None:
        raise ValueError("No exact 80/10/10 partition found with whole groups")
    test, validation = best
    for row in rows:
        key = row["group_id"]
        row["split"] = "test" if key in test else "validation" if key in validation else "train"


def verify(master, splits, candidates, tokenizer):
    expected = {row["id"]: row for rows in candidates.values() for row in rows}
    by_id = {row["id"]: row for row in master}
    if len(by_id) != len(master):
        raise ValueError("Duplicate master IDs")
    for field in ("question", "source_text"):
        if len({clean(row[field]).casefold() for row in master}) != len(master):
            raise ValueError(f"Duplicate normalized {field}")
    evidence_fields = REQUIRED | {
        "source_text", "source_sha256", "source_pdf_url", "source_edition", "pdf_pages", "clause",
    }
    for row in master:
        if any(not row.get(key) for key in REQUIRED):
            raise ValueError(f"Missing or empty field: {row['id']}")
        if any(not unicodedata.is_normalized("NFC", row[field]) for field in ("question", "answer", "source_text")):
            raise ValueError(f"Non-NFC text: {row['id']}")
        if any(row.get(key) != expected[row["id"]][key] for key in evidence_fields):
            raise ValueError(f"Source verification failed: {row['id']}")
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id, completion_only_loss=True
    )
    seen_ids = set()
    group_sets = {}
    for name, examples in splits.items():
        group_sets[name] = set()
        for start in range(0, len(examples), 16):
            batch = examples[start:start + 16]
            for example in batch:
                row = by_id[example["id"]]
                if row["split"] != name or example["id"] in seen_ids or example != tokenize(row, tokenizer):
                    raise ValueError(f"Split or token mismatch: {example['id']}")
                seen_ids.add(row["id"])
                group_sets[name].add(row["group_id"])
            labels = collator(batch)["labels"].tolist()
            for example, actual in zip(batch, labels, strict=True):
                target = [
                    token if mask else -100
                    for token, mask in zip(example["input_ids"], example["completion_mask"], strict=True)
                ]
                if actual[:len(target)] != target or any(x != -100 for x in actual[len(target):]):
                    raise ValueError(f"TRL loss mask mismatch: {example['id']}")
    if seen_ids != set(by_id):
        raise ValueError("Incomplete splits")
    for left, right in combinations(SPLITS, 2):
        if group_sets[left] & group_sets[right]:
            raise ValueError(f"Group leakage: {left}/{right}")
    for left, right in near_duplicate_pairs(master):
        if left["split"] != right["split"]:
            raise ValueError(f"Near-duplicate leakage: {left['id']}/{right['id']}")
    return {name: len(keys) for name, keys in group_sets.items()}


def stats(master, splits, groups, discarded, candidates, out_of_scope):
    return {
        "total": len(master),
        "splits": {key: len(value) for key, value in splits.items()},
        "groups": groups,
        "verification_status": {
            status: sum(row["verification_status"] == status for row in master)
            for status in ("verified", "unverified")
        },
        "question_types": {name: dict(Counter(row["question_type"] for row in master if row["split"] == name)) for name in SPLITS},
        "sources": dict(Counter(row["document_id"] for row in master)),
        "topics": dict(Counter(row["topic"] for row in master)),
        "topics_by_split": {name: dict(Counter(row["topic"] for row in master if row["split"] == name)) for name in SPLITS},
        "lengths": {
            field: {"min": min(values), "median": statistics.median(values), "max": max(values)}
            for field in ("question_words", "answer_words", "sequence_tokens", "assistant_tokens")
            for values in [[row[field] for row in master]]
        },
        "eligible_before_quota": {key: len(value) for key, value in candidates.items()},
        "filtered": dict(discarded),
        "cross_split_group_overlap": 0,
        "cross_split_near_duplicate_overlap": 0,
        "trl_mask_checked": len(master),
        "out_of_scope_test": len(out_of_scope),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing checksum-pinned official PDFs")
    parser.add_argument("--offline", action="store_true", help="Use the cached pinned Qwen tokenizer")
    parser.add_argument("--verify-only", action="store_true", help="Reopen and verify the current files")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    if not args.verify_only and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Preserving existing data in {args.output}. Use --verify-only or a new --output directory.")
    config = tomllib.loads((ROOT / "configs/dataset.toml").read_text(encoding="utf-8"))
    model = tomllib.loads((ROOT / "configs/environment.toml").read_text(encoding="utf-8"))["models"]["primary"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"], local_files_only=args.offline)
    candidates, discarded = extract_candidates(
        config, ROOT / "data/raw/official_documents", args.download, load_json(ROOT / "data/authored_questions.json")
    )
    master, encoded = select_rows(config, candidates, discarded, tokenizer)
    groups = group_rows(master, config["linked_articles"])
    assign_splits(master, groups, config["seed"])
    expected_splits = {
        name: [encoded[row["id"]] for row in master if row["split"] == name] for name in SPLITS
    }
    out_of_scope = [
        {
            "id": f"vn-road-oos-{index:02}",
            "question": clean(question),
            "expected_behavior": "state_out_of_scope",
        }
        for index, question in enumerate(load_json(ROOT / "data/out_of_scope_questions.json"), 1)
    ]
    all_questions = [row["question"].casefold() for row in master + out_of_scope]
    if any(not row["question"] for row in out_of_scope) or len(set(all_questions)) != len(all_questions):
        raise ValueError("Empty or duplicate out-of-scope question")
    if not args.verify_only:
        args.output.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output / "legal_qa_master.jsonl", master)
        for name, examples in expected_splits.items():
            write_jsonl(args.output / f"{name}.jsonl", examples)
        write_jsonl(args.output / "out_of_scope_test.jsonl", out_of_scope)
    stored = [json.loads(line) for line in (args.output / "legal_qa_master.jsonl").read_text(encoding="utf-8").splitlines()]
    loaded_splits = load_dataset(
        "json", data_files={name: str(args.output / f"{name}.jsonl") for name in SPLITS}
    )
    splits = {name: list(loaded_splits[name]) for name in SPLITS}
    if stored != master:
        raise ValueError("Stored master differs from the reproducible build")
    stored_oos = [
        json.loads(line)
        for line in (args.output / "out_of_scope_test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if stored_oos != out_of_scope:
        raise ValueError("Stored out-of-scope set differs from the reproducible build")
    for name in SPLITS:
        if len(splits[name]) != len(expected_splits[name]):
            raise ValueError(f"Unexpected {name} split size")
    checked_groups = verify(stored, splits, candidates, tokenizer)
    summary = stats(stored, splits, checked_groups, discarded, candidates, out_of_scope)
    summary["tokenizer"] = model
    summary["as_of_date"] = config["as_of"]
    summary["seed"] = config["seed"]
    if not args.verify_only:
        (args.output / "statistics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif summary != load_json(args.output / "statistics.json"):
        raise ValueError("Stored statistics differ from verified statistics")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
