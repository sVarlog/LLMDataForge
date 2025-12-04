import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from config.training_config import OUTPUT_BASE_DIR
from src.helpers.loggers import log, debug


def _cast_diff(v, default: int = 3) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _meta_block(ex: dict) -> str:
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


def _extract_between(text: str, open_tag: str, close_tag: str) -> str:
    m = re.search(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), text, flags=re.DOTALL)
    return (m.group(1).strip() if m else "").strip()


def is_structured_output(text: str) -> bool:
    m = re.search(r"<\|im_start\|><\|assistant\|>\s*(.*)", text, re.DOTALL)
    segment = m.group(1) if m else text
    has_think = ("<think>" in segment and "</think>" in segment)
    has_output = ("<output>" in segment and "</output>" in segment)
    return has_output and has_think


def find_token_sequence(token_ids: List[int], seq_ids: List[int]) -> int:
    if not seq_ids:
        return -1
    for i in range(len(token_ids) - len(seq_ids) + 1):
        if token_ids[i : i + len(seq_ids)] == seq_ids:
            return i
    return -1


def prepare_output_dir() -> Path:
    try:
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        os.makedirs(str(OUTPUT_BASE_DIR), exist_ok=True)

    existing_dirs = [d for d in OUTPUT_BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("training-")]
    nums: List[int] = []
    for d in existing_dirs:
        try:
            nums.append(int(d.name.split("-")[1]))
        except Exception:
            continue
    next_training_num = (max(nums) + 1) if nums else (len(existing_dirs) + 1)

    output_dir = OUTPUT_BASE_DIR / f"training-{next_training_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint-1").mkdir(parents=True, exist_ok=True)
    return output_dir


def find_last_training_dir() -> Optional[Path]:
    if not os.path.exists(OUTPUT_BASE_DIR):
        return None
    dirs = [d for d in os.listdir(OUTPUT_BASE_DIR) if d.startswith("training-") and os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]
    if not dirs:
        return None
    nums = [int(d.split("-")[1]) for d in dirs if d.split("-")[1].isdigit()]
    if not nums:
        return None
    last = max(nums)
    return OUTPUT_BASE_DIR / f"training-{last}"


def find_latest_checkpoint(training_dir: Path) -> Optional[Path]:
    if not training_dir or not training_dir.exists():
        return None
    chkp_dirs = [p for p in training_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not chkp_dirs:
        return None

    def _num(p: Path) -> int:
        try:
            return int(p.name.split("-")[1])
        except Exception:
            return -1

    chkp_dirs.sort(key=_num)
    return chkp_dirs[-1]


def build_bad_words_ids(tokenizer) -> List[List[int]]:
    bad = ["<|im_start|>", "<|user|>", "<|system|>", "<|im_im|>", "[META]", "[/META]"]
    out: List[List[int]] = []
    for s in bad:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            out.append(ids)
    return out


class _TeeStream:
    """Mirror writes to both the wrapped stream and a sink, without breaking progress bars."""

    def __init__(self, stream, sink_fh_getter):
        self._stream = stream
        self._sink_getter = sink_fh_getter

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            fh = self._sink_getter()
            if fh:
                fh.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            fh = self._sink_getter()
            if fh:
                fh.flush()
        except Exception:
            pass
