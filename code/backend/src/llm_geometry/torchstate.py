"""One process-wide lock for work that mutates global torch state.

Two subsystems flip process-global torch state: arch tracing (module hooks, a
monkeypatched rope function, the attention-implementation flag) and geo
training/fine-tuning (``use_deterministic_algorithms`` + the global RNG seed, via
``deterministic_torch``). Running either concurrently with any other forward pass can
corrupt a trace or perturb/deny unrelated sampling, so they all serialize here.
An RLock so a holder may re-enter (e.g. a traced forward inside a locked section).
"""

from __future__ import annotations

import threading

TORCH_GLOBAL_LOCK = threading.RLock()
