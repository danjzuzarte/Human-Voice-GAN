"""
LangGraph node functions for the adversarial loop. Each node is a thin
subprocess wrapper around one of the existing standalone CLI training
scripts (training/finetune_generator.py, training/harden_detector.py) —
consistent with how every script in this project is built as its own
argparse entry point (see e.g. training/train_detector.py,
training/generate_samples.py). Running each round's heavy lifting as a
separate `python ...` subprocess, rather than importing and calling these
scripts' main() in-process, is deliberate: it gives each round a clean CUDA
memory state (no leftover generator/detector tensors from the previous
round competing for GPU memory) and means a crash mid-round doesn't take
the whole graph process down with it — LangGraph's checkpointer (see
graph/graph.py) can still resume from the last completed round even if a
round's subprocess dies.

See graph/state.py for the state schema and graph/graph.py for how these
nodes are wired into a StateGraph with conditional routing.
"""
import json
import os
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_script(script_rel_path: str, args: list) -> None:
    """Runs `python {script_rel_path} {args}` from the repo root, streaming
    stdout/stderr live (so Colab cells show real-time training progress,
    same as running the script directly), and raises with the full command
    on a non-zero exit rather than swallowing the failure."""
    cmd = [sys.executable, script_rel_path] + args
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"'{' '.join(cmd)}' (cwd={REPO_ROOT}) exited with code {result.returncode} — "
            "see the streamed output above for the real error."
        )


def _read_yaml(path: str) -> dict:
    with open(os.path.join(REPO_ROOT, path) if not os.path.isabs(path) else path) as f:
        return yaml.safe_load(f)


def decide_side_node(state: dict) -> dict:
    """Per-side conditional routing (configs/adversarial.yaml's
    `routing.conditional`, see graph/graph.py's module docstring for the
    full graph shape). Pure decision logic, no subprocess — runs at the top
    of every round, before either training node.

    When `routing.conditional` is false (the default), returns
    active_side=None: graph/graph.py's route_after_decide_side always sends
    None to "run_generator", so the graph falls straight through to
    harden_generator then harden_detector every round, byte-for-byte the
    original always-both behavior — this node is a genuine no-op on that
    path, not a different codepath that could drift from it.

    When true, decides which single side gets trained this round:
      - Round 1 (no fooling_rate_history yet): always "generator" — mirrors
        the original loop's own round-1 ordering, since there's no signal
        yet to route on.
      - Otherwise: "detector" if last round's fooling_rate was at/above
        `routing.generator_winning_threshold` (the generator is fooling the
        detector often enough that it's the one that needs work), else
        "generator" (the detector's already ahead — give the generator more
        room). This is the standard "train whichever side is currently
        losing" GAN-balancing heuristic, evaluated fresh each round off
        fooling_rate (the metric this project has established is
        trustworthy in-loop — val_eer is not used for this decision)."""
    adv_cfg = _read_yaml(state["adversarial_config"])
    routing_cfg = adv_cfg.get("routing", {})
    if not routing_cfg.get("conditional", False):
        return {"active_side": None}

    history = state["fooling_rate_history"]
    if not history:
        side = "generator"
    else:
        threshold = routing_cfg.get("generator_winning_threshold", 0.5)
        side = "detector" if history[-1] >= threshold else "generator"

    signal = f"last fooling_rate: {history[-1]:.1%}" if history else "no history yet, round 1"
    print(f"[decide-side] round {state['round']} | {signal} -> active side this round: {side}")
    return {"active_side": side}


def harden_generator_node(state: dict) -> dict:
    """Runs training/finetune_generator.py for the current round — adversarially
    fine-tunes the generator's transformer weights against state['detector_checkpoint'].
    Resumes from state['generator_checkpoint'] if this isn't round 1 (so each round
    builds on the previous round's fine-tuning rather than restarting from the
    pretrained base every time)."""
    adv_cfg = _read_yaml(state["adversarial_config"])
    out_dir = adv_cfg["paths"]["generator_checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gen_round{state['round']}.pt")

    args = [
        "--data-config", state["data_config"],
        "--model-config", state["model_config"],
        "--generator-config", state["generator_config"],
        "--adversarial-config", state["adversarial_config"],
        "--detector-checkpoint", state["detector_checkpoint"],
        "--generator-checkpoint-out", out_path,
        "--round", str(state["round"]),
    ]
    if state.get("generator_checkpoint"):
        args += ["--generator-checkpoint-in", state["generator_checkpoint"]]

    _run_script("training/finetune_generator.py", args)
    return {"generator_checkpoint": out_path}


def harden_detector_node(state: dict) -> dict:
    """Runs training/harden_detector.py for the current round — generates
    fresh hard negatives from state['generator_checkpoint'] (this round's
    just-fine-tuned generator), hardens the detector's classifier head
    against them (warm-started from state['detector_checkpoint']), and
    reads back the round's fooling_rate / val_eer from the
    `.round_result.json` sidecar file harden_detector.py writes.

    Per-side conditional routing: if decide_side_node set
    state['active_side'] to "generator" this round (i.e. only the generator
    was trained; the detector sits this round out), this call passes
    `--eval-only` — the detector's weights stay unmodified, but a real
    fooling_rate/val_eer reading is still produced against the current
    detector (see training/harden_detector.py's `--eval-only` docstring),
    so route_after_harden_detector's convergence/stuck-rounds logic works
    identically regardless of which mode produced the round's numbers.
    active_side is None on the always-both baseline (routing.conditional:
    false) and "detector" on a round decide_side_node picked the detector
    for — neither of those passes --eval-only, so the detector always
    trains for real in both of those cases.

    Advances state['round'] by 1 — by the time this node returns, 'round'
    means "the next round to run", and len(fooling_rate_history) means
    "rounds completed so far". See graph/graph.py's routing function for how
    these are used."""
    model_cfg = _read_yaml(state["model_config"])
    out_dir = model_cfg["paths"]["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"detector_round{state['round']}.pt")

    active_side = state.get("active_side")  # None (baseline) | "generator" | "detector"
    args = [
        "--data-config", state["data_config"],
        "--model-config", state["model_config"],
        "--generator-config", state["generator_config"],
        "--adversarial-config", state["adversarial_config"],
        "--generator-checkpoint", state["generator_checkpoint"],
        "--detector-checkpoint-in", state["detector_checkpoint"],
        "--detector-checkpoint-out", out_path,
        "--round", str(state["round"]),
    ]
    if active_side == "generator":
        args.append("--eval-only")
    _run_script("training/harden_detector.py", args)

    result_path = out_path + ".round_result.json"
    with open(result_path) as f:
        result = json.load(f)

    return {
        "detector_checkpoint": out_path,
        "fooling_rate_history": state["fooling_rate_history"] + [result["fooling_rate"]],
        "val_eer_history": state["val_eer_history"] + [result["val_eer"]],
        "active_side_history": state.get("active_side_history", []) + [active_side or "both"],
        "round": state["round"] + 1,
        "status": "running",
    }


def human_review_node(state: dict) -> dict:
    """Pauses the graph via LangGraph's dynamic interrupt() (see graph/graph.py
    module docstring for why this specific mechanism was chosen, verified
    against the real installed langgraph package) and hands the caller the
    round history to make a call on. The caller resumes with
    `app.invoke(Command(resume=<decision string>), config=...)` — see
    graph/run_loop.py for the reference driver.

    Expected decision strings (case-insensitive, matched by prefix so
    "stop, this isn't converging" also works):
      "continue"                 — keep looping, but ONLY has anywhere to go if
                                    max_rounds hasn't actually been exhausted yet
                                    (e.g. this escalation was triggered by the
                                    stuck-rounds check, not the round budget) —
                                    see the no-op guard below
      "continue, extend to N"    — also raises max_rounds to N first, so there's
                                    real room to keep going
      "stop"                     — end the run here, keep the current checkpoints
    Anything not starting with "stop" is treated as "continue" — see
    graph/graph.py's route_after_human_review."""
    from langgraph.types import interrupt

    decision = interrupt({
        "reason": "max_rounds reached without convergence, or fooling_rate isn't improving across rounds",
        "round": state["round"],
        "max_rounds": state["max_rounds"],
        "fooling_rate_threshold": state["fooling_rate_threshold"],
        "fooling_rate_history": state["fooling_rate_history"],
        "val_eer_history": state["val_eer_history"],
        "detector_checkpoint": state["detector_checkpoint"],
        "generator_checkpoint": state["generator_checkpoint"],
    })

    decision_str = str(decision).strip()
    new_max_rounds = state["max_rounds"]
    if "extend to " in decision_str.lower():
        try:
            new_max_rounds = int(decision_str.lower().split("extend to ", 1)[1].split()[0])
        except (ValueError, IndexError):
            pass  # malformed "extend to" clause — keep max_rounds unchanged rather than crash the run

    is_stop = decision_str.lower().startswith("stop")
    if not is_stop and state["round"] > new_max_rounds:
        print(
            f"[human_review] decision {decision_str!r} did not raise max_rounds past the "
            f"current round ({state['round']}) — the round budget is still exhausted, so "
            "continuing would just re-hit this same review next round. Treating this as "
            "'stop' instead of looping forever. Use \"continue, extend to N\" (with N >= "
            f"{state['round']}) to actually keep going."
        )
        is_stop = True

    status = "stopped_by_human" if is_stop else "running"
    return {"human_decision": decision_str, "max_rounds": new_max_rounds, "status": status}
