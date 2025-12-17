import json
import logging as pylog
from pathlib import Path
from typing import Callable, Dict, List

import torch
from transformers import TrainingArguments, Trainer

from config.training_config import TRAINING_EPOCHS, TRAINING_EXTRA_EPOCHS
from src.helpers.loggers import log
from .callbacks import EvalCallback
from .constants import LOGGING_STEPS, INTERVAL_EVAL


class SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        loss_weight = inputs.pop("loss_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_loss = token_loss.view(shift_labels.shape)

        valid_mask = (shift_labels != -100).float()
        per_sample = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1e-6)

        if loss_weight is not None:
            loss_weight = loss_weight.to(per_sample.device).float().view(-1)
            loss = (per_sample * loss_weight).sum() / loss_weight.sum().clamp_min(1e-6)
        else:
            loss = per_sample.mean()

        return (loss, outputs) if return_outputs else loss


def create_data_collator(tokenizer) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:
    def pad_collator(features: List[Dict]):
        def flatten1d(x):
            if hasattr(x, "flatten"):
                x = x.flatten()
            if hasattr(x, "tolist"):
                x = x.tolist()
            if isinstance(x, list) and x and isinstance(x[0], list):
                x = [item for sublist in x for item in sublist]
            return x

        seq_keys = ["input_ids", "labels", "attention_mask"]
        max_len = max(len(flatten1d(f["input_ids"])) for f in features)
        batch = {k: [] for k in seq_keys}
        loss_weights: List[float] = []

        for f in features:
            for k in seq_keys:
                if k == "labels":
                    pad_value = -100
                elif k == "attention_mask":
                    pad_value = 0
                else:
                    pad_value = tokenizer.pad_token_id
                v = flatten1d(f[k])
                arr = v + [pad_value] * (max_len - len(v))
                batch[k].append(arr)
            loss_weights.append(float(f.get("loss_weight", 1.0)))

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "loss_weight": torch.tensor(loss_weights, dtype=torch.float32),
        }

    return pad_collator


def train_model(
    model,
    tokenizer,
    dataset,
    output_dir,
    canonical_assistant_ids,
    train_dataset,
    resume_from_checkpoint: Path | None = None,
    final_log_fh=None,
):
    log("Configuring training arguments...")
    pylog.getLogger("accelerate").setLevel(pylog.INFO)
    pylog.getLogger("peft").setLevel(pylog.INFO)

    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"❌ Parameter {name} is still on meta device!")

    torch.utils.checkpoint._use_reentrant = False
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=TRAINING_EPOCHS,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.3,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        group_by_length=True,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=create_data_collator(tokenizer),
        callbacks=[
            EvalCallback(
                tokenizer,
                canonical_assistant_ids,
                output_dir,
                interval=INTERVAL_EVAL,
                raw_dataset=train_dataset,
                final_log_fh=final_log_fh,
            )
        ],
    )

    if resume_from_checkpoint:
        try:
            ts = Path(resume_from_checkpoint) / "trainer_state.json"
            if ts.exists():
                st = json.load(open(ts, "r", encoding="utf-8"))
                current_epoch = float(st.get("epoch", 0.0) or 0.0)
            else:
                current_epoch = 0.0
        except Exception:
            current_epoch = 0.0

        try:
            original_epochs = float(training_args.num_train_epochs or TRAINING_EPOCHS)
        except Exception:
            original_epochs = TRAINING_EPOCHS

        new_target_epochs = current_epoch + TRAINING_EXTRA_EPOCHS
        if new_target_epochs <= original_epochs:
            pass

        log(
            f"Resuming from checkpoint. current_epoch={current_epoch:.2f}, setting num_train_epochs -> {new_target_epochs:.2f}"
        )
        training_args.num_train_epochs = new_target_epochs
        trainer.args = training_args
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    else:
        trainer.train()

    model.save_pretrained(output_dir)
