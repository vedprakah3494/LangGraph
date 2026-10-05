# =====================================================
# HARNESS ENGINEERING SUMMARY
# -----------------------------------------------------
# Purpose:
# 1. Permission Guardrail
#    -> Is this action allowed?
#
# 2. Observability Trace
#    -> Record everything that happened.
#
# Flow:
#
# START
#   ↓
# check_permission
#   ↓
# allow / block
#   ↓
# execute_action OR reject_action
#   ↓
# END
# =====================================================

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# =====================================================
# STEP 1: ALLOWLIST
# -----------------------------------------------------
# Only these actions may execute.
#
# lookup_order           ✅
# check_shipping_status  ✅
#
# issue_refund           ❌
# delete_order           ❌
# =====================================================

ALLOWED_ACTIONS = {
    "lookup_order",
    "check_shipping_status"
}


# =====================================================
# STEP 2: STATE
# -----------------------------------------------------
# requested_action -> user request
# trace            -> audit log
# allowed          -> permission result
# output           -> final response
# =====================================================

class State(TypedDict):
    requested_action: str
    trace: list
    allowed: bool
    output: str


# =====================================================
# STEP 3: PERMISSION GUARDRAIL
# -----------------------------------------------------
# Check:
# Is the action in ALLOWED_ACTIONS?
#
# Also write the decision into trace.
# =====================================================

def check_permission(state: State):

    allowed = (
        state["requested_action"]
        in ALLOWED_ACTIONS
    )

    if allowed:
        note = (
            f"check_permission: "
            f"'{state['requested_action']}' is allowed"
        )
    else:
        note = (
            f"check_permission: "
            f"'{state['requested_action']}' is blocked"
        )

    trace = state["trace"] + [note]

    return {
        "allowed": allowed,
        "trace": trace
    }


# =====================================================
# STEP 4: ROUTER
# -----------------------------------------------------
# If allowed:
#     execute_action
#
# Otherwise:
#     reject_action
# =====================================================

def route_after_permission(state: State):

    return (
        "execute"
        if state["allowed"]
        else "reject"
    )


# =====================================================
# STEP 5A: EXECUTE ACTION
# -----------------------------------------------------
# Runs only if permission granted.
#
# Adds execution info to trace.
# =====================================================

def execute_action(state: State):

    trace = state["trace"] + [
        f"execute_action: ran '{state['requested_action']}'"
    ]

    return {
        "trace": trace,
        "output":
            f"'{state['requested_action']}' completed."
    }


# =====================================================
# STEP 5B: REJECT ACTION
# -----------------------------------------------------
# Runs only if permission denied.
#
# Adds rejection info to trace.
# =====================================================

def reject_action(state: State):

    trace = state["trace"] + [
        f"reject_action: refused "
        f"'{state['requested_action']}'"
    ]

    return {
        "trace": trace,
        "output":
            f"'{state['requested_action']}' "
            f"was blocked by the harness."
    }


# =====================================================
# STEP 6: BUILD GRAPH
# -----------------------------------------------------
# START
#   ↓
# check_permission
#   ↓
# execute_action OR reject_action
#   ↓
# END
# =====================================================

builder = StateGraph(State)

builder.add_node(
    "check_permission",
    check_permission
)

builder.add_node(
    "execute_action",
    execute_action
)

builder.add_node(
    "reject_action",
    reject_action
)

builder.add_edge(
    START,
    "check_permission"
)

builder.add_conditional_edges(
    "check_permission",
    route_after_permission,
    {
        "execute": "execute_action",
        "reject": "reject_action"
    }
)

builder.add_edge(
    "execute_action",
    END
)

builder.add_edge(
    "reject_action",
    END
)

graph = builder.compile()


# =====================================================
# STEP 7: RUN TEST
# -----------------------------------------------------
# Shows:
# 1. Action requested
# 2. Trace log
# 3. Final output
# =====================================================

def run(action):

    result = graph.invoke({
        "requested_action": action,
        "trace": []
    })

    print(f"\nRequested Action: {action}")

    print("\nTRACE:")
    for item in result["trace"]:
        print("-", item)

    print("\nOUTPUT:")
    print(result["output"])


# =====================================================
# TEST 1
# -----------------------------------------------------
# Allowed Action
#
# Flow:
# check_permission
#       ↓
# allowed
#       ↓
# execute_action
# =====================================================

run("lookup_order")


# =====================================================
# TEST 2
# -----------------------------------------------------
# Blocked Action
#
# Flow:
# check_permission
#       ↓
# blocked
#       ↓
# reject_action
# =====================================================

run("issue_refund")


# =====================================================
# FINAL TAKEAWAY
# -----------------------------------------------------
# Harness Engineering =
#
# 1. Guardrails
#    -> What is allowed?
#
# 2. Observability
#    -> What happened?
#
# 3. Auditability
#    -> Why did it happen?
#
# Example:
#
# lookup_order    ✅ Allowed
# issue_refund    ❌ Blocked
#
# Trace provides a complete history of decisions.
# =====================================================