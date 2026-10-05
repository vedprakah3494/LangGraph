# ============================================================
# LOOP ENGINEERING SUMMARY
# ------------------------------------------------------------
# This graph demonstrates an Act → Verify → Retry loop.
#
# The loop can stop for THREE reasons:
#
# 1. Success     -> Verification passed
# 2. Cost Limit  -> Budget exhausted
# 3. Retry Limit -> Max attempts reached
#
# This is a Loop Engineering pattern because we control
# execution using multiple stop conditions.
# ============================================================


from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# STEP 1: STATE
# ------------------------------------------------------------
# Stores everything needed by the loop.
#
# attempt          -> current retry count
# cost             -> total cost spent
# cost_per_attempt -> cost of one iteration
# budget           -> maximum allowed cost
# max_attempts     -> retry limit
# threshold        -> success target
# result           -> output from act()
# passed           -> verification result
# ============================================================

class State(TypedDict):
    attempt: int
    cost: float
    cost_per_attempt: float
    budget: float
    max_attempts: int
    threshold: int
    result: int
    passed: bool


# ============================================================
# STEP 2: ACT
# ------------------------------------------------------------
# Simulates one unit of work.
#
# Every iteration:
# - increases attempt count
# - increases cost
# - produces a result
#
# Example:
# Attempt 1 -> Result 5
# Attempt 2 -> Result 10
# Attempt 3 -> Result 15
# ============================================================

def act(state: State):

    attempt = state["attempt"] + 1

    cost = (
        state["cost"]
        + state["cost_per_attempt"]
    )

    result = attempt * 5

    return {
        "attempt": attempt,
        "cost": cost,
        "result": result
    }


# ============================================================
# STEP 3: VERIFY
# ------------------------------------------------------------
# Checks whether current result satisfies threshold.
#
# Example:
# Result = 10
# Threshold = 10
#
# Passed = True
# ============================================================

def verify(state: State):

    return {
        "passed":
        state["result"] >= state["threshold"]
    }


# ============================================================
# STEP 4: LOOP CONTROLLER (LOOP ENGINEERING)
# ------------------------------------------------------------
# Decides whether to stop or continue.
#
# Stop Signal #1:
#   Verification passed
#
# Stop Signal #2:
#   Budget exhausted
#
# Stop Signal #3:
#   Maximum retries reached
#
# Otherwise:
#   Retry again
# ============================================================

def route_after_verify(state: State):

    if state["passed"\]:
        return "done"

    if state["cost"] >= state["budget"\]:
        return "done"

    if state["attempt"] >= state["max_attempts"\]:
        return "done"

    return "retry"


# ============================================================
# STEP 5: BUILD GRAPH
# ------------------------------------------------------------
# Flow:
#
# START
#   ↓
# act
#   ↓
# verify
#   ↓
# router
#   ↓
# retry → act
# done  → END
# ============================================================

builder = StateGraph(State)

builder.add_node("act", act)
builder.add_node("verify", verify)

builder.add_edge(START, "act")
builder.add_edge("act", "verify")

builder.add_conditional_edges(
    "verify",
    route_after_verify,
    {
        "retry": "act",
        "done": END
    }
)

graph = builder.compile()


# ============================================================
# STEP 6: STOP REASON
# ------------------------------------------------------------
# Human-readable explanation of why the loop ended.
# ============================================================

def stop_reason(result):

    if result["passed"\]:
        return "Verification Passed"

    if result["cost"] >= result["budget"\]:
        return "Budget Limit Reached"

    return "Maximum Attempts Reached"


# ============================================================
# STEP 7: EXECUTE
# ------------------------------------------------------------
# Runs a scenario and prints final outcome.
# ============================================================

def run(title, state):

    result = .invoke(state)

    print(f"\n{title}")
    print("-" * 50)

    print("Attempts :", result["attempt"])
    print("Cost     :", result["cost"])
    print("Result   :", result["result"])
    print("Reason   :", stop_reason(result))


# ============================================================
# SCENARIO A
# ------------------------------------------------------------
# Success Path
#
# Threshold reachable.
# Loop stops because verification passes.
# ============================================================

run(
    "RUN A - SUCCESS",
    {
        "attempt": 0,
        "cost": 0,
        "cost_per_attempt": 1,
        "budget": 100,
        "max_attempts": 5,
        "threshold": 10,
    }
)


# ============================================================
# SCENARIO B
# ------------------------------------------------------------
# Retry Limit Path
#
# Threshold impossible.
# Loop stops because max attempts reached.
# ============================================================

run(
    "RUN B - ATTEMPT LIMIT",
    {
        "attempt": 0,
        "cost": 0,
        "cost_per_attempt": 1,
        "budget": 100,
        "max_attempts": 3,
        "threshold": 1000,
    }
)


# ============================================================
# SCENARIO C
# ------------------------------------------------------------
# Cost Limit Path
#
# Threshold impossible.
# Budget exhausted before max attempts.
# ============================================================

run(
    "RUN C - COST LIMIT",
    {
        "attempt": 0,
        "cost": 0,
        "cost_per_attempt": 10,
        "budget": 25,
        "max_attempts": 10,
        "threshold": 1000,
    }
)

# ============================================================
# FINAL TAKEAWAY
# ------------------------------------------------------------
# Loop Engineering = Control execution.
#
# A loop should NOT stop only when successful.
#
# Production AI agents also monitor:
#
# ✓ Success Signal
# ✓ Cost Budget
# ✓ Retry Limit
# ✓ Token Budget
# ✓ Time Limit
#
# Multiple stop signals make agent systems safe,
# predictable, and cost-controlled.
# ============================================================