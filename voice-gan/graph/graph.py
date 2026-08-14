"""
The LangGraph adversarial loop's StateGraph — wires graph/nodes.py's node
functions together with conditional routing, per ARCHITECTURE.md.

API usage below (StateGraph, add_conditional_edges, compile(checkpointer=...),
SqliteSaver, the dynamic interrupt()/Command(resume=...) pattern in
graph/nodes.py's human_review_node) was verified against the real installed
langgraph package (1.2.11 at verification time — checked live sources rather
than assuming API shape from training data, same approach used for the
F5-TTS research the generator code is built on), not assumed. In particular:
LangGraph moved from a static
`interrupt_before=[...]` compile-time list (older versions) to a dynamic
`interrupt()` call made from inside a node function, resumed via
`app.invoke(Command(resume=...), config=...)` — graph/run_loop.py is written
against this current API.

Loop shape (deliberately simple — see the design note below):

    START -> harden_generator -> harden_detector -> [route] -> ...
                  ^                                     |
                  |------------- "continue" -------------
                                                          |
                                            "converged" -> END
                                                          |
                                      "needs_review" -> human_review -> [route] -> ...
                                                                            |          |
                                                                     "continue"      "stop"
                                                                            |          |
                                                                     harden_generator  END

Design note (scope honesty): the original design sketch considered routing
each round to EITHER harden_generator OR harden_detector individually based
on the current fooling rate (skip whichever side is already winning). This
implementation always runs both every round (a fixed adversarial pair, like
a standard GAN step) and reserves the conditional routing for the
loop-level decision — keep looping, stop (converged), or escalate to a
human. That's a deliberate simplification: it's easier to reason about,
test, and resume, and still produces a real
adversarial arms race round over round. Skipping a side conditionally is a
plausible later refinement, not something this graph does today.
"""
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from graph.nodes import harden_detector_node, harden_generator_node, human_review_node
from graph.state import AdversarialLoopState

# How many consecutive non-improving rounds (fooling_rate flat or rising)
# before escalating to human_review instead of blindly continuing — cheap
# guard against silently burning the whole GPU-hours budget on a stuck run.
STUCK_ROUNDS_THRESHOLD = 3


def route_after_harden_detector(state: AdversarialLoopState) -> str:
    history = state["fooling_rate_history"]
    latest = history[-1]

    if latest <= state["fooling_rate_threshold"]:
        return "converged"

    if state["round"] > state["max_rounds"]:
        return "needs_review"

    if len(history) >= STUCK_ROUNDS_THRESHOLD:
        recent = history[-STUCK_ROUNDS_THRESHOLD:]
        if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
            return "needs_review"  # flat or rising fooling rate for STUCK_ROUNDS_THRESHOLD rounds straight

    return "continue"


def route_after_human_review(state: AdversarialLoopState) -> str:
    return "stop" if state["status"] == "stopped_by_human" else "continue"


def mark_converged(state: AdversarialLoopState) -> dict:
    return {"status": "converged"}


def build_graph() -> StateGraph:
    """Returns the uncompiled graph builder — call .compile(checkpointer=...)
    on the result (see build_app() below for the common case)."""
    graph = StateGraph(AdversarialLoopState)

    graph.add_node("harden_generator", harden_generator_node)
    graph.add_node("harden_detector", harden_detector_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("mark_converged", mark_converged)

    graph.add_edge(START, "harden_generator")
    graph.add_edge("harden_generator", "harden_detector")
    graph.add_conditional_edges(
        "harden_detector", route_after_harden_detector,
        {"continue": "harden_generator", "converged": "mark_converged", "needs_review": "human_review"},
    )
    graph.add_edge("mark_converged", END)
    graph.add_conditional_edges(
        "human_review", route_after_human_review,
        {"continue": "harden_generator", "stop": END},
    )

    return graph


def build_app(sqlite_path: str):
    """Context-manager helper: `with build_app(path) as app:` — yields a
    compiled, checkpointed app. SqliteSaver.from_conn_string is itself a
    context manager (confirmed against the real langgraph_checkpoint_sqlite
    package), so this just forwards to it rather than wrapping it in
    anything fancier."""
    return SqliteSaver.from_conn_string(sqlite_path)


def compile_with_saver(saver) -> "object":
    graph = build_graph()
    return graph.compile(checkpointer=saver)


def initial_state(
    data_config: str, model_config: str, generator_config: str, adversarial_config: str,
    detector_checkpoint: str, max_rounds: int, fooling_rate_threshold: float,
) -> AdversarialLoopState:
    """Builds round-1 state — pass the originally trained detector_best.pt
    as detector_checkpoint to start a fresh adversarial run, or a later round's
    detector checkpoint to continue hardening from where a previous run
    (possibly a previous graph/run_loop.py invocation) left off."""
    return {
        "round": 1,
        "max_rounds": max_rounds,
        "fooling_rate_threshold": fooling_rate_threshold,
        "detector_checkpoint": detector_checkpoint,
        "generator_checkpoint": None,
        "data_config": data_config,
        "model_config": model_config,
        "generator_config": generator_config,
        "adversarial_config": adversarial_config,
        "fooling_rate_history": [],
        "val_eer_history": [],
        "status": "running",
        "human_decision": None,
    }
