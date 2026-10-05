from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class MyState(TypedDict):
    price: float
    quantity: float
    total: float
    finalValue:float
    route: str

def use_tool(state: MyState):
    price = state.get('price', 0.0)
    quantity = state.get('quantity', 0.0)
    total =calculate(price, quantity)   
    state['total'] = total

def calculate(price: float, quantity: float) -> float:
    return price * quantity

# Node 1: Updates state with the chosen route
def choose_route(state: MyState):
    if state.get('total', 0.0) > 100:
       return "high_value"
    else:
       return "low_value"

def route_decision(state: MyState) -> str:
    return state["route"]

# Worker Nodes
def appply_discount(state: MyState):
    total = state.get('total', 0.0)
    discounted_total = total * 0.9  # Apply a 10% discount
    state['finalValue'] = discounted_total
    return state

def no_discount(state: MyState):
    total = state.get('total', 0.0)
    state['finalValue'] = total
    return state

# Build the Graph
workflow = StateGraph(MyState)

# 1. Add all functional nodes
workflow.add_node("choose_route", choose_route)
workflow.add_node("apply_discount", appply_discount)
workflow.add_node("no_discount", no_discount)

# 2. Define flow control
workflow.add_edge(START, "choose_route")

# The router function routes directly from choose_route to the destination nodes
workflow.add_conditional_edges(
    "choose_route",
    route_decision,
    {
        "high_value": "apply_discount",
        "low_value": "no_discount"
    }

)

# 3. All worker nodes lead to the End
workflow.add_edge("apply_discount", END)
workflow.add_edge("no_discount", END)

# Compile and Run
graph = workflow.compile()
result = graph.invoke({"request": "What is the weather like today?"})

print("Chosen Route:", result['route'])
print("Final Answer:", result['finalValue'])

result = graph.invoke({"request": "What is the latest news?"})

print("Chosen Route:", result['route'])
print("Final Answer:", result['finalValue'])
