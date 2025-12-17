import json
import random
import re
from pathlib import Path
from typing import Any, Dict

from transformers import TrainerCallback

from config.training_config import SYSTEM_PROMPT
from src.helpers.build_messages import build_messages
from src.helpers.loggers import log
from .constants import DIFFICULTY_TO_EVAL_WEIGHT
from .generation import run_generation_and_print
from .utils import _cast_diff, _meta_block, _extract_between, is_structured_output


class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, canonical_assistant_ids, output_dir, interval, raw_dataset, final_log_fh=None):
        self.tokenizer = tokenizer
        self.canonical_assistant_ids = canonical_assistant_ids
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        self.raw_dataset = raw_dataset
        self.final_log_fh = final_log_fh

    def _pick_eval_sample(self, state):
        weights = []
        for ex in self.raw_dataset:
            d = _cast_diff(ex.get("difficulty", 3))
            weights.append(DIFFICULTY_TO_EVAL_WEIGHT.get(d, 1.0))
        idx = random.choices(range(len(self.raw_dataset)), weights=weights, k=1)[0]
        return self.raw_dataset[idx]

    def _score_output(self, pred_text: str, ref_text: str, diff: int):
        pred_out = _extract_between(pred_text, "<output>", "</output>")
        tok = lambda s: re.findall(r"[a-z0-9]+", s.casefold())
        p = tok(pred_out)
        r = tok(ref_text or "")
        if not p and not r:
            f1 = 1.0
        elif not p or not r:
            f1 = 0.0
        else:
            ps, rs = set(p), set(r)
            inter = len(ps & rs)
            prec = inter / max(len(ps), 1)
            rec = inter / max(len(rs), 1)
            f1 = (2 * prec * rec) / (prec + rec + 1e-9)

        struct = 1.0 if is_structured_output(pred_text) else 0.0
        base = 0.2 * struct + 0.8 * f1
        return base * DIFFICULTY_TO_EVAL_WEIGHT.get(diff, 1.0), {"f1": f1, "structured": struct}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval != 0:
            return

        ex = self._pick_eval_sample(state)
        diff = _cast_diff(ex.get("difficulty", 3))
        user_content = _meta_block(ex) + ex["question"]
        messages = build_messages(SYSTEM_PROMPT, user_content)

        mode = "force_think" if state.global_step < 100 else "auto"
        output_str = run_generation_and_print(
            kwargs["model"],
            self.tokenizer,
            messages,
            canonical_assistant_ids=self.canonical_assistant_ids,
            label=f"Eval @ step {state.global_step} (diff={diff})",
            mode=mode,
        )

        score, parts = self._score_output(output_str, ex.get("output", ""), diff)
        log_dict = dict(state.log_history[-1]) if state.log_history else {}
        log_dict.update(
            {
                "eval_difficulty": diff,
                "eval_score_weighted": round(float(score), 4),
                "eval_f1": round(float(parts["f1"]), 4),
                "eval_structured": float(parts["structured"]),
            }
        )
        metrics_str = f"Metrics: {json.dumps(log_dict, indent=2)}\n\n"
        log_file = self.logs_dir / f"callback-{state.global_step}.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(metrics_str)
            f.write(output_str)

        try:
            if self.final_log_fh:
                self.final_log_fh.write(metrics_str)
                self.final_log_fh.write(output_str)
                self.final_log_fh.flush()
        except Exception:
            pass
