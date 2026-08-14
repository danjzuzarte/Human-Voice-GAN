"""
Reference driver for the LangGraph adversarial loop (graph/graph.py). Runs
harden_generator <-> harden_detector rounds until convergence,
max_rounds, or a human stops the run — resumable across process
restarts/Colab-session drops via the SQLite checkpointer (same `thread_id`
+ `--sqlite-path` resumes an in-progress run instead of starting over).

Usage (fresh run):
    python graph/run_loop.py --data-config configs/data.yaml --model-config configs/model.yaml \\
        --generator-config configs/generator.yaml --adversarial-config configs/adversarial.yaml \\
        --detector-checkpoint /content/drive/MyDrive/voice-gan/checkpoints/detector/detector_best.pt \\
        --max-rounds 5 --sqlite-path /content/drive/MyDrive/voice-gan/checkpoints/adversarial_loop.sqlite

Usage (resume an interrupted/dropped run — same --sqlite-path and --thread-id,
--detector-checkpoint is ignored once a thread already has state):
    python graph/run_loop.py --sqlite-path /content/drive/MyDrive/voice-gan/checkpoints/adversarial_loop.sqlite

Non-interactive (Colab "run all cells", or scheduled runs) — auto-answers
any human_review interrupt with a fixed decision instead of blocking on
input():
    python graph/run_loop.py ... --auto-decision "continue"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.types import Command

from graph.graph import build_app, compile_with_saver, initial_state


def _get_human_decision(interrupt_payload: dict, auto_decision: "str | None") -> str:
    print("\n[run-loop] === human_review requested ===")
    print(json.dumps(interrupt_payload, indent=2))
    if auto_decision is not None:
        print(f"[run-loop] --auto-decision set, answering automatically: {auto_decision!r}")
        return auto_decision
    return input(
        "[run-loop] Decision? ('continue', 'continue, extend to N', or 'stop'): "
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--generator-config", default="configs/generator.yaml")
    parser.add_argument("--adversarial-config", default="configs/adversarial.yaml")
    parser.add_argument("--detector-checkpoint", default=None, help="the original trained detector checkpoint, or a previous round's — required to START a fresh run, ignored when resuming an existing thread")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--fooling-rate-threshold", type=float, default=None, help="defaults to adversarial.yaml's detector_hardening.fooling_rate_threshold")
    parser.add_argument("--sqlite-path", default="adversarial_loop.sqlite")
    parser.add_argument("--thread-id", default="adversarial-loop")
    parser.add_argument("--auto-decision", default=None, help="answer any human_review interrupt with this fixed string instead of prompting via input() — for non-interactive runs")
    args = parser.parse_args()

    fooling_rate_threshold = args.fooling_rate_threshold
    if fooling_rate_threshold is None:
        import yaml
        with open(args.adversarial_config) as f:
            fooling_rate_threshold = yaml.safe_load(f)["detector_hardening"]["fooling_rate_threshold"]

    config = {"configurable": {"thread_id": args.thread_id}}

    with build_app(args.sqlite_path) as saver:
        app = compile_with_saver(saver)

        existing = app.get_state(config)
        if existing.values:
            print(f"[run-loop] resuming thread '{args.thread_id}' from round {existing.values.get('round')} "
                  f"({len(existing.values.get('fooling_rate_history', []))} rounds already completed)")
            graph_input = None  # None: continue from the checkpointer's stored state, no new input to merge
        else:
            if not args.detector_checkpoint:
                raise ValueError(
                    "--detector-checkpoint is required to start a fresh run "
                    f"(no existing state found for thread '{args.thread_id}' at {args.sqlite_path})"
                )
            print(f"[run-loop] starting fresh thread '{args.thread_id}'")
            graph_input = initial_state(
                args.data_config, args.model_config, args.generator_config, args.adversarial_config,
                args.detector_checkpoint, args.max_rounds, fooling_rate_threshold,
            )

        while True:
            result = app.invoke(graph_input, config=config)

            if "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                decision = _get_human_decision(payload, args.auto_decision)
                graph_input = Command(resume=decision)
                continue

            # No interrupt -> the graph ran to END this call.
            final_state = app.get_state(config).values
            print(f"\n[run-loop] === run finished: status={final_state['status']} "
                  f"after {len(final_state['fooling_rate_history'])} round(s) ===")
            for i, (fr, eer) in enumerate(zip(final_state["fooling_rate_history"], final_state["val_eer_history"]), 1):
                print(f"[run-loop]   round {i}: fooling_rate={fr:.1%}  val_eer={eer:.2%}")
            print(f"[run-loop] final detector checkpoint: {final_state['detector_checkpoint']}")
            print(f"[run-loop] final generator checkpoint: {final_state['generator_checkpoint']}")
            break


if __name__ == "__main__":
    main()
