"""Optional optimizer-boundary evidence for external training supervisors."""
import json
from pathlib import Path

from transformers import TrainerCallback


class KunlitaMetricsCallback(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)

    def write(self, state, record):
        if state.is_world_process_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=True) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.write(state, {"event": "start", "step": state.global_step,
                           "world_size": args.world_size, "max_steps": state.max_steps})

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Trainer emits loss here after optimizer.step and global_step increment.
        # Other callbacks may log per-microbatch values: do not count those.
        if logs and "loss" in logs:
            self.write(state, {"event": "update", "step": state.global_step,
                               "loss": logs["loss"], **({"grad_norm": logs["grad_norm"]}
                               if "grad_norm" in logs else {})})

    def on_train_end(self, args, state, control, **kwargs):
        self.write(state, {"event": "end", "step": state.global_step})
