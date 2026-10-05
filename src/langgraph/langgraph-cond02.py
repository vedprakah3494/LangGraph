from typing import TypedDict
from langgraph.graph import END, START, StateGraph
import ollama
import json

class MyState(TypedDict):
    request: str
    route: str
    answer: str


# Node 1: Converted from basic if/else into an LLM Router Agent
def choose_route(state: MyState):
    request_text = state.get('request', '')
    
    # Prompt instructing the LLM to choose a strict category string
    prompt = f"""You are a routing agent. Categorize the following user request into exactly one of these categories: 'weather', 'news', or 'general'.
    
    User Request: "{request_text}"
    
    Respond with ONLY the category word in lowercase. Do not include punctuation or extra sentences."""

    response = ollama.chat(
        model="gemma3",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    
    # Clean the output string to ensure it matches the keys in our mapping dictionary
    chosen = response['message']['content'].strip().lower()
    
    # Fallback safety guardrail: if the model hallucinates an invalid string, default to general
    if chosen not in ["weather", "news", "general"]:
        chosen = "general"
        
    return {"route": chosen}


# Router Function: Returns the string name of the next node
def route_decision(state: MyState) -> str:
    return state["route"]

# Worker Nodes
def weather(state: MyState):
    response = ollama.chat(
        model="gemma3",
        messages=[
            {"role": "system", "content": "Provide a detailed weather report in 100 words."},
            {"role": "user", "content": state['request']},
        ],
    )
    state['answer'] = response['message']['content']
    return state

def news(state: MyState):
    response = ollama.chat(
        model="gemma3",
        messages=[
            {"role": "system", "content": "Provide a detailed news report in 100 words."},
            {"role": "user", "content": state['request']},
        ],
    )
    state['answer'] = response['message']['content']
    return state

def general(state: MyState):
    response = ollama.chat(
        model="gemma3",
        messages=[
            {"role": "system", "content": "Provide a detailed general answer in 100 words."},
            {"role": "user", "content": state['request']},
        ],
    )
    state['answer'] = response['message']['content']
    return state

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
