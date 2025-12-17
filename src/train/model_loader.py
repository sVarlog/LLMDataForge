import json
import os
from time import time
from pathlib import Path

import torch
from transformers import logging, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from config.config import MODEL_NAME
from config.training_config import LORA_CONFIG_PATH
from src.helpers.loggers import log


def check_lora_modules(model, lora_config_path: str):
    with open(lora_config_path, "r") as f:
        lora_cfg = LoraConfig(**json.load(f))
    all_module_names = [name for name, _ in model.named_modules()]
    found, missing = [], []

    log("Checking LoRA target modules against the model…")
    for target in lora_cfg.target_modules:
        matches = [mn for mn in all_module_names if target in mn]
        if matches:
            found.append(target)
            snippet = matches[:3] + (["…"] if len(matches) > 3 else [])
            log(f"  ✔ `{target}` matched in: {snippet}")
        else:
            missing.append(target)
            log(f"  ❌ `{target}` NOT found in model modules!")
    log(f"✅ Modules to be LoRA‐tuned : {found}")

    if missing:
        log(f"⚠️ Warning: these targets were missing and will be skipped: {missing}")

    return lora_cfg


def load_model_and_prepare_for_qora(tokenizer, output_dir: Path):
    start = time()
    log("Loading model config and weights…")
    logging.set_verbosity_warning()

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    log("Preparing model for QLoRA adapters…")
    model = prepare_model_for_kbit_training(model)

    assert os.path.exists(LORA_CONFIG_PATH), "Missing LoRA config"
    log(f"Checking LoRA config at {LORA_CONFIG_PATH}…")
    lora_cfg = check_lora_modules(model, LORA_CONFIG_PATH)
    log("Applying LoRA adapters…")
    model = get_peft_model(model, lora_cfg)

    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    model.config.use_cache = False

    end = time()
    log(f"✅ Model & LoRA ready in {end - start:.2f}s")
    logging.set_verbosity_warning()
    return model
