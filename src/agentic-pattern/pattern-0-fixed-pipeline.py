from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    request: str
    log: list[str]
    answer: str


# ----------------------------------
# Step 1: Receive Request
# ----------------------------------

def receive_request(state: State):
    return {
        "log": [f"Received: {state['request']}"]
    }


# ----------------------------------
# Step 2: Generate Response
# ----------------------------------

def generate_reply(state: State):

    answer = (
        f"Standard response for: "
        f"{state['request']}"
    )

    return {
        "answer": answer,
        "log": state["log"] + ["Generated response"]
    }


# ----------------------------------
# Step 3: Close Ticket
# ----------------------------------

def close_ticket(state: State):

    return {
        "log": state["log"] + ["Ticket closed"]
    }


# ----------------------------------
# Build Graph
# ----------------------------------

builder = StateGraph(State)

builder.add_node("receive", receive_request)
builder.add_node("reply", generate_reply)
builder.add_node("close", close_ticket)

builder.add_edge(START, "receive")
builder.add_edge("receive", "reply")
builder.add_edge("reply", "close")
builder.add_edge("close", END)

graph = builder.compile()

# ----------------------------------
# Execute
# ----------------------------------

result = graph.invoke({
    "request": "My VPN is disconnected"
})

print("\nWorkflow Log:")
for step in result["log"]:
    print(step)

print("\nAnswer:")
print(result["answer"]);