"""
State schema for the LangGraph adversarial loop (harden_generator <->
harden_detector), see graph/graph.py and ARCHITECTURE.md.

Plain TypedDict, matching LangGraph's standard state-schema pattern (verified
against the real installed langgraph package, not assumed — see graph/graph.py
module docstring). Every node function takes a (partial or full) State and
returns a dict of the keys it's updating; LangGraph merges that into the
running state between node calls.
"""
from typing import TypedDict


class AdversarialLoopState(TypedDict):
    # --- round bookkeeping ---
    round: int              # 1-indexed; incremented after each harden_detector node run
    max_rounds: int         # hard cap — reaching this without convergence routes to human_review
    fooling_rate_threshold: float  # a round's post-hardening fooling_rate at or below this = converged

    # --- checkpoint paths, updated as rounds progress ---
    detector_checkpoint: str            # current detector checkpoint (round N's harden_detector output, or the original trained detector_best.pt for round 1)
    generator_checkpoint: "str | None"  # current fine-tuned generator transformer checkpoint (None = pretrained F5-TTS base, before round 1)

    # --- static config paths, passed through unchanged every round ---
    data_config: str
    model_config: str
    generator_config: str
    adversarial_config: str
    # Round-numbered checkpoint output paths are derived inside graph/nodes.py
    # from adversarial.yaml's `paths.generator_checkpoint_dir` and
    # model.yaml's `paths.checkpoint_dir` — not duplicated here, so there's
    # one source of truth for where checkpoints land (same one
    # training/finetune_generator.py and training/harden_detector.py already
    # read via --generator-config / --model-config).

    # --- metric history, one entry appended per completed round ---
    fooling_rate_history: list   # list[float], oldest first
    val_eer_history: list        # list[float], oldest first

    # --- loop status / human-in-the-loop ---
    status: str                # "running" | "converged" | "max_rounds_reached" | "stopped_by_human" | "awaiting_human_review"
    human_decision: "str | None"  # set by the caller resuming an interrupt() — see graph/nodes.py human_review_node

    # --- per-side conditional routing (configs/adversarial.yaml
    # `routing.conditional`, see graph/graph.py's decide_side_node) ---
    active_side: "str | None"       # this round's active side: "generator" | "detector" | None (None = routing.conditional is false, i.e. always-both)
    active_side_history: list       # list[str], oldest first — which side was actually trained each completed round ("generator" | "detector" | "both")
