import torch
from transformers import StoppingCriteriaList, StoppingCriteria

from src.helpers.loggers import log
from .tokenization import format_and_tokenize
from .utils import is_structured_output, build_bad_words_ids


class StopOnSubstring(StoppingCriteria):
    def __init__(self, tokenizer, substrings, start_len: int, window_tokens: int = 64):
        self.tokenizer = tokenizer
        self.substrings = substrings
        self.start_len = int(start_len)
        self.window = window_tokens

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        seq_len = len(seq)
        if seq_len <= self.start_len:
            return False
        tail_start = max(self.start_len, seq_len - self.window)
        tail = seq[tail_start:seq_len]
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        return any(sub in text for sub in self.substrings)


def run_generation_and_print(model, tokenizer, messages, canonical_assistant_ids=None, label: str = "Eval", mode: str = "auto"):
    formatted_text, inputs = format_and_tokenize(
        messages,
        tokenizer,
        return_tensors=True,
        add_generation_prompt=True,
        canonical_assistant_ids=canonical_assistant_ids,
    )

    model_device = next(model.parameters()).device
    inputs = {k: (v.to(model_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    bad_words_ids = build_bad_words_ids(tokenizer)
    prompt_len = inputs["input_ids"].shape[1]
    stop_subs = ["</output>", "<|im_end|>"]
    stopping_criteria = StoppingCriteriaList([StopOnSubstring(tokenizer, stop_subs, start_len=prompt_len)])

    prev_mode = model.training
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=256,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=4,
            repetition_penalty=1.05,
            bad_words_ids=bad_words_ids,
        )
    if prev_mode:
        model.train()

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(output[0][input_len:], skip_special_tokens=False)

    prompt_tail = formatted_text if len(formatted_text) <= 800 else formatted_text[-800:]
    header = f"\n🧪 {label}:\n" if label else "\n🧪 Generation:\n"
    out_str = (
        header
        + "📥 Prompt (tail):\n"
        + prompt_tail
        + "\n\n"
        + "📤 Output:\n"
        + decoded
        + "\n"
    )

    try:
        structured_flag = is_structured_output(out_str)
    except Exception:
        structured_flag = is_structured_output(decoded)
    log(f"Is structured output: {structured_flag}")
    return out_str
