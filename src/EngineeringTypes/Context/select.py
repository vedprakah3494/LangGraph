
from multiprocessing import context

import dotenv
from langgraph import graph
import ollama   
from langgraph.graph import StateGraph, START
from typing import TypedDict



MODEL = "gemma3"

class State(TypedDict):
    request: str
    rag_candidates: list[tuple[str, str]]
    rag_choice:str
    selected_rag:list[tuple[str, str]]
    context: str
    answer: str

dotenv.load_dotenv()

def rag_classifiyer(state: State):
    labels = [label for label, _ in state["rag_candidates"]]
    list_of_labels = "\n".join([f"{i+1}. {label}" for i, label in enumerate(labels)])
    prompt =  (
        "Which of these knowledge base passages, if any, is relevant to the question?\n"
        f"Return only one label from this list, or NONE: {', '.join(labels)}\n\n"
        f"Question: {state['request']}\n\n{list_of_labels}"
    )
    response = ollama.chat(MODEL, messages=[{"role": "user", "content": prompt}])
    choice = response["message"]["content"].strip().upper()
    valid = {label.upper() for label in labels} | {"NONE"}
    if choice not in valid:
        choice = "NONE"
    return {"rag_choice": choice}

def route_after_classification(state: State):
    if state["rag_choice"] == "NONE":
        return "skip_rag"
    else:
        return "include_rag"


def include_rag(state: State):
    for label, text in state["rag_candidates"]:
        if label.upper() == state["rag_choice"]:
            return {"selected_rag": [(label, text)]}
    return {"selected_rag": []}


def skip_rag(state: State):
    return {"selected_rag": []}


def build_context(state: State):
    lines = [f"TASK: {state['request']}"]
    for label, text in state["selected_rag"]:
        lines.append(f"RAG[{label}]: {text}")
    return {"context": "\n".join(lines)}


def ask_model(state: State):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": state['context']}],
    )
    return {"answer": response["message"]["content"]}


graph_builder = StateGraph(State)
graph_builder.add_node("rag_classifiyer", rag_classifiyer)
graph_builder.add_node("include_rag", include_rag)
graph_builder.add_node("skip_rag", skip_rag)
graph_builder.add_node("build_context", build_context)
graph_builder.add_node("ask_model", ask_model)


graph_builder.add_edge(START, "rag_classifiyer")
graph_builder.add_conditional_edges("rag_classifiyer", route_after_classification)
graph_builder.add_edge("include_rag", "build_context")
graph_builder.add_edge("skip_rag", "build_context")
graph_builder.add_edge("build_context", "ask_model")

graph=graph_builder.compile()

result = graph.invoke({
    "request": "What is the capital of France?",
    "rag_candidates": [
        ("RAG1", "The capital of France is Paris."),
        ("RAG2", "The capital of Germany is Berlin."),
        ("RAG3", "The capital of Italy is Rome."),
    ],
})
print(result["context"])
print(result["answer"])