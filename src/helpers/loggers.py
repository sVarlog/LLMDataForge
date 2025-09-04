import sys
import logging as pylog  # stdlib logging

from config.training_config import (
    FINAL_LOG_FH, 
    DEF_LOG_PREFIX, 
    TEE_ACTIVE, 
    DEBUG, 
    DEF_DBG_PREFIX,
    _ORIG_STDOUT,
    _ORIG_STDERR
)

def _write_sink(s: str):
    try:
        if FINAL_LOG_FH:
            FINAL_LOG_FH.write(s + "\n")
            FINAL_LOG_FH.flush()
    except Exception:
        pass

def log(msg):
    s = f"\n{DEF_LOG_PREFIX}{msg}\n{'=' * 60}"
    print(s)
    # avoid double-writing: when tee is active, print already goes to file
    if not TEE_ACTIVE:
        _write_sink(s)


def debug(msg):
    if DEBUG:
        s = f"\n{DEF_DBG_PREFIX}{msg}\n{'-' * 60}"
        print(s)
        if not TEE_ACTIVE:
            _write_sink(s)

def _detach_handlers_to(file_obj):
    for name in ("transformers", "peft", "accelerate"):
        lg = pylog.getLogger(name)
        for h in list(lg.handlers):
            if getattr(h, "stream", None) is file_obj:
                lg.removeHandler(h)
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass

def close_logs():
    try:
        # restore std streams first
        if _ORIG_STDOUT: sys.stdout = _ORIG_STDOUT
        if _ORIG_STDERR: sys.stderr = _ORIG_STDERR
    except Exception:
        pass
    try:
        if FINAL_LOG_FH:
            FINAL_LOG_FH.flush()

            _detach_handlers_to(FINAL_LOG_FH)

            FINAL_LOG_FH.close()
    except Exception:
        pass