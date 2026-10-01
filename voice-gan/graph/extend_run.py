"""
Revives a run that ended with status=stopped_by_human because
route_after_harden_detector's round > max_rounds check fired (see
graph/nodes.py's human_review_node) — the common case being: you want to
raise the round ceiling and keep going instead of stopping there.

Why you can't just bump --max-rounds and re-run graph/run_loop.py: two
separate reasons, both real.
  1. `--max-rounds` is a FRESH-THREAD-ONLY argument. run_loop.py's resume
     path passes `graph_input=None` (not a new initial_state()), so the CLI
     flag is silently ignored once a thread already has checkpointed state
     — the persisted state["max_rounds"] only ever changes via a
     "continue, extend to N" human_review decision, never by re-parsing the
     CLI on resume.
  2. Once a run reaches END (status=stopped_by_human, routed there via
     route_after_human_review's "stop" branch), the graph is genuinely
     finished — there's no pending interrupt() left to resume into.
     Re-invoking with graph_input=None just replays the frozen final state;
     no node runs, regardless of --auto-decision.

This script uses LangGraph's `update_state(..., as_node="human_review")` to
manually re-open that branch: it rewrites the persisted status back to
"running" (and, optionally, max_rounds/fooling_rate_threshold) "as if"
human_review had just returned those values — so the graph's own
route_after_human_review conditional edge re-evaluates from there and
routes onward to decide_side and then to a real training node, instead of
staying stuck at END.
Everything else in the checkpointed state (round, both checkpoints,
fooling_rate_history, val_eer_history) is left untouched — update_state only
overwrites the keys you actually pass it.

After running this, re-run graph/run_loop.py exactly as before (same
--sqlite-path/--thread-id) — it'll pick up the reopened state and continue
from the next round.

Usage:
    python graph/extend_run.py --sqlite-path <path> --thread-id <id> --max-rounds 30
    python graph/extend_run.py --sqlite-path <path> --thread-id <id> --max-rounds 30 --fooling-rate-threshold 0.1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.graph import build_app, compile_with_saver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-path", required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--max-rounds", type=int, default=None, help="new round ceiling — omit to leave the persisted value unchanged")
    parser.add_argument("--fooling-rate-threshold", type=float, default=None, help="optionally also loosen/tighten the convergence threshold")
    args = parser.parse_args()

    config = {"configurable": {"thread_id": args.thread_id}}
    with build_app(args.sqlite_path) as saver:
        app = compile_with_saver(saver)
        before = app.get_state(config)
        if not before.values:
            raise ValueError(f"No existing state found for thread '{args.thread_id}' at {args.sqlite_path} — nothing to extend.")
        if before.values.get("status") != "stopped_by_human":
            print(
                f"[extend-run] warning: current status is {before.values.get('status')!r}, not 'stopped_by_human' — "
                "this script is meant for reviving a run that stopped there. Proceeding anyway."
            )

        values = {"status": "running"}
        if args.max_rounds is not None:
            values["max_rounds"] = args.max_rounds
        if args.fooling_rate_threshold is not None:
            values["fooling_rate_threshold"] = args.fooling_rate_threshold

        app.update_state(config, values, as_node="human_review")

        after = app.get_state(config).values
        print(f"[extend-run] thread '{args.thread_id}' reopened:")
        print(f"[extend-run]   round: {after['round']} (next round to run)")
        print(f"[extend-run]   max_rounds: {before.values['max_rounds']} -> {after['max_rounds']}")
        print(f"[extend-run]   fooling_rate_threshold: {before.values['fooling_rate_threshold']} -> {after['fooling_rate_threshold']}")
        print(f"[extend-run]   status: {before.values['status']} -> {after['status']}")
        print(f"[extend-run]   rounds completed so far: {len(after['fooling_rate_history'])}")
        print("[extend-run] now re-run graph/run_loop.py with the same --sqlite-path/--thread-id to continue.")


if __name__ == "__main__":
    main()
