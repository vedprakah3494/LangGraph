from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class MyState(TypedDict):
    request: str
    route: str
    answer: str

# Node 1: Updates state with the chosen route
def choose_route(state: MyState):
    request_text = state.get('request', '').lower()
    if "weather" in request_text:
        chosen = "weather"
    elif "news" in request_text:
        chosen = "news"
    else:
        chosen = "general"
    
    # Return ONLY what changes (LangGraph merges dictionary updates)
    return {"route": chosen}

# Router Function: Returns the string name of the next node
def route_decision(state: MyState) -> str:
    return state["route"]

# Worker Nodes
def weather(state: MyState):
    return {"answer": "The weather is sunny."}

def news(state: MyState):
    return {"answer": "Today's news is about the stock market."}

def general(state: MyState):
    return {"answer": "This is a general answer."}

# Build the Graph
workflow = StateGraph(MyState)

# 1. Add all functional nodes
workflow.add_node("choose_route", choose_route)
workflow.add_node("weather", weather)
workflow.add_node("news", news)
workflow.add_node("general", general)

# 2. Define flow control
workflow.add_edge(START, "choose_route")

# The router function routes directly from choose_route to the destination nodes
workflow.add_conditional_edges(
    "choose_route",
    route_decision,
      {
            "weather": "weather",
            "news": "news",
            "general": "general"
       }

)

# 3. All worker nodes lead to the End
workflow.add_edge("weather", END)
workflow.add_edge("news", END)
workflow.add_edge("general", END)

# Compile and Run
graph = workflow.compile()
result = graph.invoke({"request": "What is the weather like today?"})

print("Chosen Route:", result['route'])
print("Final Answer:", result['answer'])

result = graph.invoke({"request": "What is the latest news?"})

print("Chosen Route:", result['route'])
print("Final Answer:", result['answer'])
