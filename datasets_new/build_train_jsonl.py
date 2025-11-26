import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# Ensure repo root + src are importable so we can reuse config/helpers for token stats
REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "src"):
    if extra.exists():
        extra_str = str(extra)
        if extra_str not in sys.path:
            sys.path.insert(0, extra_str)

print("Tokenizer import begin")

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency for token stats
    AutoTokenizer = None

from config.config import MODEL_NAME
from config.training_config import SYSTEM_PROMPT, MAX_RESPONSE_LEN
from src.helpers.build_messages import build_messages

STRUCTURE_FILE = "structure.enriched.json"
TOPICS_DIR = "topics"

def load_structure(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Missing dataset file: {path}")
    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON, skipping: {path}")
    return []

def pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _replace_placeholders(text, mapping):
    if not isinstance(text, str) or not text:
        return text
    for k, v in mapping.items():
        if v is None:
            continue  # allow leaving some placeholders untouched
        text = text.replace(k, v)
    return text

def _apply_identity_placeholders(row, identity_map):
    """
    Apply ${NAME} ${SPEC} ${TRAINER} replacements to identity.* topics.
    Works on question, think, and output fields.
    """
    topic = row.get("topic", "") or ""
    category = row.get("category", "") or ""

    if topic.startswith("identity.") or category == "identity":
        row["question"] = _replace_placeholders(row.get("question", ""), identity_map)
        row["think"] = _replace_placeholders(row.get("think", ""), identity_map)
        row["output"] = _replace_placeholders(row.get("output", ""), identity_map)
    return row


def _meta_block(ex: dict) -> str:
    """Match the metadata header used during training for accurate token stats."""
    tags = ex.get("tags", [])
    if isinstance(tags, list):
        tags_str = ", ".join(map(str, tags))
    else:
        tags_str = str(tags) if tags is not None else ""
    return (
        "[META]\n"
        f"category: {ex.get('category','')}\n"
        f"subcategory: {ex.get('subcategory','')}\n"
        f"topic: {ex.get('topic','')}\n"
        f"content_type: {ex.get('content_type','')}\n"
        f"difficulty: {ex.get('difficulty','')}\n"
        f"tags: {tags_str}\n"
        "[/META]\n\n"
    )


def _format_messages_for_token_stats(row):
    user_content = _meta_block(row) + row.get("question", "")
    think = row.get("think", "")
    output = row.get("output", "")
    response = f"<think>{think}</think><output>{output}</output>"
    return build_messages(SYSTEM_PROMPT, user_content, response)


def _load_tokenizer_for_stats(tokenizer_path: str):
    if AutoTokenizer is None:
        raise RuntimeError(
            "transformers is required for token statistics. Install transformers or rerun with --no-stats."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        add_bos_token=False,
        add_eos_token=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def _compute_token_summary(rows, tokenizer_path, max_length):
    if not rows:
        return {"total_tokens": 0, "avg_tokens": 0.0, "truncated": 0, "tokenizer_name": tokenizer_path}

    tokenizer = _load_tokenizer_for_stats(tokenizer_path)
    total_tokens = 0
    truncated = 0

    for row in rows:
        messages = _format_messages_for_token_stats(row)
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            max_length=max_length,
            truncation=True,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if len(token_ids) >= max_length:
            truncated += 1
        total_tokens += len(token_ids)

    avg_tokens = total_tokens / len(rows) if rows else 0.0
    return {
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "truncated": truncated,
        "tokenizer_name": getattr(tokenizer, "name_or_path", tokenizer_path),
    }

def normalize_entry(
    entry, *,
    category, subcat, ctype, meta_tags, source_version, fallback_topic, next_id
):
    # Core fields (tolerate minor schema variants)
    q = pick(entry, "question", "q", default=None)
    think = pick(entry, "think", "rationale", "reasoning", default=None)
    out = pick(entry, "output", "answer", "a", default=None)

    # If any are missing, warn and skip by returning None
    if not q or not out:
        print(f"⚠️ Skipping sample without required fields (question/output) in {category}/{subcat}/{ctype}")
        return None

    row = {
        "id": entry.get("id", next_id),
        "category": category,
        "subcategory": subcat,
        "content_type": ctype,
        "topic": entry.get("topic", fallback_topic),
        # Coerce difficulty to an int for consistent column typing across the JSONL
        "difficulty": None,
        "question": q,
        "think": think if think is not None else "",
        "output": out,
        # Use metadata tags; if entry has tags, union them with meta tags (de-dup, keep order).
        "tags": list(dict.fromkeys((entry.get("tags") or []) + (meta_tags or []))),
        "source_version": source_version,
    }
    # Normalize difficulty into an int (datasets/pyarrow expects consistent types)
    raw_diff = entry.get("difficulty", 3)
    try:
        diff = int(raw_diff)
    except Exception:
        # allow common text values, otherwise fallback to default 3
        if isinstance(raw_diff, str):
            v = raw_diff.strip().lower()
            mapping = {"easy": 1, "medium": 3, "hard": 5}
            diff = mapping.get(v, 3)
            if v not in mapping:
                print(f"⚠️ Non-numeric difficulty '{raw_diff}' in {category}/{subcat}/{ctype}, using {diff}")
        else:
            diff = 3

    row["difficulty"] = diff
    return row

def export_jsonl(output_path, rows):
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Advanced Stats & Pretty Printing ----------

def _pct(n, d):
    return 0.0 if d == 0 else (n / d) * 100.0

def _bar(pct, width=24):
    # simple ascii bars for readability
    blocks = int(round((pct / 100.0) * width))
    return "█" * blocks + " " * (width - blocks)

def _print_header(title, char="="):
    line = char * 88
    print(f"\n{line}\n{title}\n{line}")

def _print_subheader(title):
    print(f"\n— {title} —")

def _format_row(left, right):
    return f"{left:<58} {right:>28}"

def _print_topic_breakdown(rows):
    total = len(rows)
    if total == 0:
        print("No rows to analyze.")
        return

    # Determine difficulty range present
    diffs_present = sorted({int(r.get("difficulty", 0)) for r in rows})
    min_d, max_d = diffs_present[0], diffs_present[-1]

    # Group by topic
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r.get("topic", "unknown")].append(r)

    topics_sorted = sorted(by_topic.keys())

    _print_header("PER-TOPIC STATISTICS")
    for idx, topic in enumerate(topics_sorted, start=1):
        topic_rows = by_topic[topic]
        tcount = len(topic_rows)
        tshare = _pct(tcount, total)

        print(_format_row(f"[{idx:02d}] Topic: {topic}", f"Count: {tcount}  ({tshare:.2f}%)"))
        # Difficulty table for this topic
        counts = Counter(int(r.get("difficulty", 0)) for r in topic_rows)
        # Pretty table
        print("  Difficulty Distribution:")
        for d in range(min_d, max_d + 1):
            c = counts.get(d, 0)
            pct = _pct(c, tcount)
            bar = _bar(pct)
            print(f"    • Diff {d:>2}: {c:>5}  ({pct:6.2f}%)  |{bar}|")
        print("")

def _print_global_breakdown(rows, token_summary=None):
    total = len(rows)
    if total == 0:
        return
    diffs_present = sorted({int(r.get("difficulty", 0)) for r in rows})
    min_d, max_d = diffs_present[0], diffs_present[-1]

    # Topic counts
    topic_counts = Counter(r.get("topic", "unknown") for r in rows)
    # Difficulty counts
    diff_counts = Counter(int(r.get("difficulty", 0)) for r in rows)

    _print_header("GLOBAL STATISTICS")
    _print_subheader("Topics (share of total)")
    for idx, (topic, cnt) in enumerate(sorted(topic_counts.items()), start=1):
        pct = _pct(cnt, total)
        bar = _bar(pct)
        print(f"  [{idx:02d}] {topic:<40} {cnt:>6}  ({pct:6.2f}%)  |{bar}|")

    _print_subheader("Difficulties (share of total)")
    for d in range(min_d, max_d + 1):
        cnt = diff_counts.get(d, 0)
        pct = _pct(cnt, total)
        bar = _bar(pct)
        print(f"  Diff {d:>2}  {cnt:>8}  ({pct:6.2f}%)  |{bar}|")

    print("\n" + "-" * 88)
    print(f"TOTAL SAMPLES: {total:,}")
    if token_summary:
        total_tokens = token_summary.get("total_tokens", 0)
        avg_tokens = token_summary.get("avg_tokens", 0.0)
        print(f"TOTAL TOKENS: {total_tokens:,}")
        print(f"AVG TOKENS/SAMPLE: {avg_tokens:,.2f}")
        truncated = token_summary.get("truncated", 0)
        if truncated:
            print(f"⚠️ Truncated samples at max_length ({truncated})")

def main():
    ap = argparse.ArgumentParser(description="Flatten topics/* into JSONL with per-sample metadata + stats.")
    ap.add_argument("--structure", default=STRUCTURE_FILE, help="Path to structure.enriched.json")
    ap.add_argument("--topics-dir", default=TOPICS_DIR, help="Path to topics directory")
    ap.add_argument("--output", default="train_data.jsonl", help="Output JSONL filename")
    ap.add_argument("--source-version", default="v1.0", help="Version stamp to include on each row")
    ap.add_argument("--start-id", type=int, default=10_000, help="Start ID if a sample lacks id")
    ap.add_argument("--no-stats", action="store_true", help="Suppress advanced console statistics")
    ap.add_argument(
        "--tokenizer-path",
        default=MODEL_NAME,
        help="HF model ID or local path used to estimate total tokens",
    )

    args = ap.parse_args()

    identity_map = {
        "${NAME}": "Oliver",
        "${SPEC}": "AI",
        "${TRAINER}": "LLMDataForge",
    }

    structure = load_structure(args.structure)

    rows = []
    next_id = args.start_id
    seen_ids = set()

    for category, subcats in structure.items():
        for subcat, meta in subcats.items():
            content_types = meta.get("content_type", [])
            if isinstance(content_types, str):
                content_types = [content_types]
            meta_tags = meta.get("tags", [])
            subcat_dir = os.path.join(args.topics_dir, category, subcat)

            for ctype in content_types:
                filename = f"{category}.{subcat}.{ctype}.json"
                file_path = os.path.join(subcat_dir, filename)
                data = read_json_file(file_path)

                if not isinstance(data, list):
                    print(f"⚠️ Expected a list in {file_path}; skipping.")
                    continue

                fallback_topic = f"{category}.{subcat}"

                for entry in data:
                    row = normalize_entry(
                        entry,
                        category=category,
                        subcat=subcat,
                        ctype=ctype,
                        meta_tags=meta_tags,
                        source_version=args.source_version,
                        fallback_topic=fallback_topic,
                        next_id=next_id
                    )
                    if row is None:
                        continue

                    # Apply identity placeholder replacements only for identity.* topics
                    row = _apply_identity_placeholders(row, identity_map)

                    # Ensure unique IDs; if clash, reassign sequentially
                    rid = row["id"]
                    if rid in seen_ids:
                        rid = next_id
                        row["id"] = rid
                    seen_ids.add(rid)
                    next_id = max(next_id + 1, rid + 1)

                    rows.append(row)

    # Export first so file is guaranteed even if stats printing crashes.
    export_jsonl(args.output, rows)
    print(f"✅ Wrote {len(rows)} rows to {args.output}")

    # Advanced stats (unless suppressed)
    if not args.no_stats:
        token_summary = None
        try:
            token_summary = _compute_token_summary(rows, args.tokenizer_path, MAX_RESPONSE_LEN)
        except Exception as exc:
            print(f"⚠️ Token statistic calculation failed: {exc}")
        _print_topic_breakdown(rows)
        _print_global_breakdown(rows, token_summary)

if __name__ == "__main__":
    main()
