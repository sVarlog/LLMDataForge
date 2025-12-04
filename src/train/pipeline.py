import sys
import logging as pylog
from pathlib import Path

from datasets import load_dataset
from transformers import logging

import _bootstrap  # noqa: F401
import config.training_config as tc
from config.training_config import (
    SYSTEM_PROMPT,
    DATA_PATH,
    ASSISTANT_OPEN_WITH_NL,
    ASSISTANT_OPEN_NO_NL,
    TRAINING_NEW,
)
from src.helpers.persist_chat_template import persist_chat_template
from src.helpers.loggers import log, debug, close_logs
from src.train.tokenization import (
    load_and_prepare_tokenizer,
    format_and_tokenize,
    tokenize_function,
    dump_tokenizer_artifacts,
)
from src.train.model_loader import load_model_and_prepare_for_qora
from src.train.trainer_core import train_model
from src.train.utils import (
    _TeeStream,
    prepare_output_dir,
    find_last_training_dir,
    find_latest_checkpoint,
    find_token_sequence,
)

FINAL_LOG_FH = None
_ORIG_STDOUT = None
_ORIG_STDERR = None


def _resolve_output_and_checkpoint():
    resume_checkpoint = None
    output_dir = None
    try:
        from config import config as cfg

        candidate = cfg.resolve_adapter_checkpoint()
        if candidate is not None and TRAINING_NEW is False:
            resume_checkpoint = candidate
            output_dir = candidate.parent
            log(f"Found existing adapter checkpoint via config: {candidate}")
        else:
            if not TRAINING_NEW:
                last = find_last_training_dir()
                if last is not None:
                    log(f"TRAINING_NEW is False — reusing last training dir: {last}")
                    output_dir = last
                    ck = find_latest_checkpoint(output_dir)
                    if ck is not None:
                        resume_checkpoint = ck
                        log(f"Found latest checkpoint: {resume_checkpoint}")
                else:
                    log("TRAINING_NEW is False but no previous training dir found; creating new one")
                    output_dir = prepare_output_dir()
            else:
                output_dir = prepare_output_dir()
    except Exception:
        if not TRAINING_NEW:
            last = find_last_training_dir()
            if last is not None:
                output_dir = last
                ck = find_latest_checkpoint(output_dir)
                if ck is not None:
                    resume_checkpoint = ck
            else:
                output_dir = prepare_output_dir()
        else:
            output_dir = prepare_output_dir()
    return output_dir, resume_checkpoint


def _init_logging(output_dir: Path):
    global FINAL_LOG_FH, _ORIG_STDOUT, _ORIG_STDERR
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    FINAL_LOG_FH = open(logs_dir / "finalLog.txt", "a", encoding="utf-8")

    _ORIG_STDOUT, _ORIG_STDERR = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(_ORIG_STDOUT, lambda: FINAL_LOG_FH)
    sys.stderr = _TeeStream(_ORIG_STDERR, lambda: FINAL_LOG_FH)
    tc.FINAL_LOG_FH = FINAL_LOG_FH
    tc._ORIG_STDOUT = _ORIG_STDOUT
    tc._ORIG_STDERR = _ORIG_STDERR
    tc.TEE_ACTIVE = True

    pylog.basicConfig(level=pylog.WARNING)
    tf_logger = pylog.getLogger("transformers")
    tf_logger.setLevel(pylog.WARNING)
    if not any(
        isinstance(h, pylog.StreamHandler) and getattr(h.stream, "name", "") == FINAL_LOG_FH.name
        for h in tf_logger.handlers
        if hasattr(h, "stream")
    ):
        tf_logger.addHandler(pylog.StreamHandler(FINAL_LOG_FH))

    logging.set_verbosity_warning()
    return FINAL_LOG_FH


def _detect_canonical_assistant_ids(tokenizer, s_no, s_nl):
    fmt, tok = format_and_tokenize(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "__DETECT__"}],
        tokenizer,
        return_tensors=True,
        add_generation_prompt=True,
        canonical_assistant_ids=None,
    )
    fmt_ids = tok["input_ids"][0].tolist()
    pos_no = find_token_sequence(fmt_ids, s_no)
    pos_nl = find_token_sequence(fmt_ids, s_nl)

    if pos_no != -1 and pos_nl == -1:
        canonical_assistant_ids = s_no
        debug("Canonical assistant marker: no-newline variant")
    elif pos_nl != -1 and pos_no == -1:
        canonical_assistant_ids = s_nl
        debug("Canonical assistant marker: newline variant")
    elif pos_nl != -1 and pos_no != -1:
        canonical_assistant_ids = s_no
        debug("Both variants in template; picking no-newline as canonical")
    else:
        canonical_assistant_ids = s_nl
        debug("No variant found in detection; defaulting to newline variant (best-effort)")
    return canonical_assistant_ids


def init_training():
    log("Preparing output directory")
    output_dir, resume_checkpoint = _resolve_output_and_checkpoint()

    final_log_fh = _init_logging(output_dir)

    log("Loading tokenizer and adding special tags")
    tokenizer = load_and_prepare_tokenizer(output_dir)

    s_nl = tokenizer.encode(ASSISTANT_OPEN_WITH_NL, add_special_tokens=False)
    s_no = tokenizer.encode(ASSISTANT_OPEN_NO_NL, add_special_tokens=False)

    log("Saving chat template to tokenizer")
    persist_chat_template(tokenizer, output_dir)
    dump_tokenizer_artifacts(tokenizer, output_dir)

    canonical_assistant_ids = _detect_canonical_assistant_ids(tokenizer, s_no, s_nl)

    log("Loading and tokenizing dataset")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    train_dataset = dataset

    remove_cols = [c for c in dataset.column_names if c not in ("input_ids", "labels", "attention_mask", "loss_weight")]
    dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, canonical_assistant_ids),
        remove_columns=remove_cols,
        batched=False,
    )

    log(f"Dataset loaded with {len(dataset)} examples.")
    log(f"Sample tokenized example: {dataset[0]}")

    stop_ids = tokenizer.encode("</output>", add_special_tokens=False)
    log(f"stop ids: {stop_ids}, {tokenizer.convert_ids_to_tokens(stop_ids)}")

    log("Loading model and applying LoRA")
    model = load_model_and_prepare_for_qora(tokenizer, output_dir)

    log("Training model")
    train_model(
        model,
        tokenizer,
        dataset,
        output_dir,
        canonical_assistant_ids,
        train_dataset,
        resume_from_checkpoint=resume_checkpoint,
        final_log_fh=final_log_fh,
    )

    close_logs()


def start_training():
    log("=== Starting training run ===")
    init_training()
