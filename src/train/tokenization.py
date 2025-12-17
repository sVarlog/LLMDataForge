from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer
from tokenizers import Tokenizer

from config.config import MODEL_NAME
from config.training_config import (
    SYSTEM_PROMPT,
    ANCHOR_INTO_OUTPUT,
    SUPERVISE_OUTPUT_ONLY,
    ASSISTANT_OPEN_WITH_NL,
    ASSISTANT_OPEN_NO_NL,
    MAX_RESPONSE_LEN,
)
from src.helpers.build_messages import build_messages
from src.helpers.loggers import log, debug
from .constants import DIFFICULTY_TO_LOSS_WEIGHT
from .utils import _cast_diff, _meta_block, find_token_sequence


def load_and_prepare_tokenizer(output_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        add_bos_token=False,
        add_eos_token=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def format_and_tokenize(
    messages: List[Dict[str, str]],
    tokenizer,
    return_tensors: bool = False,
    add_generation_prompt: bool = False,
    canonical_assistant_ids: List[int] | None = None,
) -> Tuple[str, Any]:
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )

    if add_generation_prompt and canonical_assistant_ids is not None:
        tmp = tokenizer(formatted_text, return_tensors=None, add_special_tokens=False)
        ids = tmp["input_ids"]
        ids_list = ids[0] if hasattr(ids, "__len__") and not isinstance(ids, list) else ids
        pos = find_token_sequence(ids_list, canonical_assistant_ids)
        if pos == -1:
            log("⚠️ formatted_text does NOT contain canonical assistant marker")
            debug("formatted_text repr: " + repr(formatted_text[-200:]))
            debug("canonical_assistant_ids tokens: " + str(tokenizer.convert_ids_to_tokens(canonical_assistant_ids)))
        else:
            debug("formatted_text contains canonical assistant marker at token pos " + str(pos))

    if add_generation_prompt and ANCHOR_INTO_OUTPUT:
        pass

    if return_tensors:
        tokenized = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False)
    else:
        tokenized = tokenizer(
            formatted_text,
            padding="longest",
            truncation=True,
            max_length=MAX_RESPONSE_LEN,
            return_tensors=None,
            add_special_tokens=False,
        )

    return formatted_text, tokenized


def tokenize_function(ex: dict, tokenizer, canonical_assistant_ids: List[int]):
    diff_int = _cast_diff(ex.get("difficulty", 3))
    loss_weight = DIFFICULTY_TO_LOSS_WEIGHT.get(diff_int, 1.0)

    user_content = _meta_block(ex) + ex["question"]
    response = f"<think>{ex['think']}</think><output>{ex['output']}</output>"
    messages = build_messages(SYSTEM_PROMPT, user_content, response)

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        max_length=MAX_RESPONSE_LEN,
        truncation=True,
    )

    im_end_marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    start_pos = find_token_sequence(token_ids, canonical_assistant_ids)

    if start_pos == -1:
        cand_with_nl = tokenizer.encode(ASSISTANT_OPEN_WITH_NL, add_special_tokens=False)
        cand_no_nl = tokenizer.encode(ASSISTANT_OPEN_NO_NL, add_special_tokens=False)
        start_pos = find_token_sequence(token_ids, cand_with_nl)
        used_marker = cand_with_nl
        if start_pos == -1:
            start_pos = find_token_sequence(token_ids, cand_no_nl)
            used_marker = cand_no_nl
        if start_pos == -1:
            log("❌ Could not find assistant marker in tokens (tokenize_function)")
            tail = token_ids[-120:] if len(token_ids) > 120 else token_ids
            log("tail ids:", tail)
            log("tail toks:", tokenizer.convert_ids_to_tokens(tail))
            raise AssertionError("❌ Could not find assistant marker in tokens")
    else:
        used_marker = canonical_assistant_ids

    start_idx = start_pos + len(used_marker)
    end_idx = -1
    for i in range(start_idx, len(token_ids) - len(im_end_marker) + 1):
        if token_ids[i : i + len(im_end_marker)] == im_end_marker:
            end_idx = i
            break
    if end_idx == -1:
        end_idx = len(token_ids)

    labels = [-100] * len(token_ids)

    if SUPERVISE_OUTPUT_ONLY:
        out_open = tokenizer.encode("<output>", add_special_tokens=False)
        out_close = tokenizer.encode("</output>", add_special_tokens=False)
        start_out = find_token_sequence(token_ids[start_idx:end_idx], out_open)
        end_out = find_token_sequence(token_ids[start_idx:end_idx], out_close)
        if start_out != -1 and end_out != -1:
            o_s = start_idx + start_out
            o_e = start_idx + end_out + len(out_close)
            labels[o_s:o_e] = token_ids[o_s:o_e]
        else:
            labels[start_idx:end_idx] = token_ids[start_idx:end_idx]
            debug("Could not find explicit <output> tags — supervising whole assistant span")
    else:
        labels[start_idx:end_idx] = token_ids[start_idx:end_idx]

    attention_mask = [1] * len(token_ids)

    return {
        "input_ids": token_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "loss_weight": loss_weight,
    }


def dump_tokenizer_artifacts(tokenizer, save_dir: Path):
    log(f"🔧 Saving chat template + tokenizer to {save_dir}")
    tokenizer.save_pretrained(save_dir)

    tmpl_path = Path(save_dir) / "chat_template.jinja"
    tokenizer.chat_template = tmpl_path.read_text(encoding="utf-8")
    tokenizer.init_kwargs["chat_template"] = tokenizer.chat_template

    fast_tok = Tokenizer.from_file(str(save_dir / "tokenizer.json"))
    bpe = fast_tok.model
    bpe_folder = save_dir / "bpe-tokenizer"
    bpe_folder.mkdir(exist_ok=True)
    bpe.save(str(bpe_folder))
    (bpe_folder / "vocab.json").rename(save_dir / "vocab.json")
    (bpe_folder / "merges.txt").rename(save_dir / "merges.txt")
    bpe_folder.rmdir()
    log(f"✅ Chat template + vocab/merges dumped to {save_dir}")
