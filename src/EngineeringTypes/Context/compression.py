from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import ollama

MODEL = "gemma3"
BUDGET = 180


# -----------------------------
# State
# -----------------------------
class State(TypedDict):
    sections: dict[str, str]
    budget: int
    cut_choice: str
    context: str
    dropped_section: str
    answer: str


# -----------------------------
# Build Context Sections
# -----------------------------
def assemble_sections(state: State):

    sections = {
        "TASK": "What is the status of ticket #4402?",

        "RAG": """
        Ticket #4402 was closed on September 28.
        Resolution: Network configuration issue fixed.
        Owner: Network Operations Team.
        """,

        "MEMORY": """
        Yesterday the user asked about VPN access.
        Earlier the user requested password reset instructions.
        """
    }

    return {
        "sections": sections,
        "budget": BUDGET,
        "dropped_section": "NONE"
    }


# -----------------------------
# Budget Compression Agent
# -----------------------------
def decide_cut(state: State):

    sections = state["sections"]

    total_size = sum(len(text) for text in sections.values())

    print(f"\nTotal Characters: {total_size}")
    print(f"Budget: {state['budget']}")

    if total_size <= state["budget"]:
        return {"cut_choice": "NONE"}

    listing = "\n".join(
        f"{label}: {text}"
        for label, text in sections.items()
        if label != "TASK"
    )

    prompt = f"""
        The context exceeds the budget.

        Current size: {total_size}
        Budget: {state['budget']}

        Choose ONE section to remove.

        Never choose TASK.

        Return only:
        RAG
        or
        MEMORY

        {listing}
        """

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    choice = response["message"]["content"].strip().upper()

    print("Agent Decision:", choice)

    # Safe fallback
    if choice not in ["RAG", "MEMORY"]:
        choice = "MEMORY"

    return {"cut_choice": choice}


# -----------------------------
# Routing
# -----------------------------
def route_after_cut(state: State):

    if state["cut_choice"] == "RAG":
        return "drop_rag"

    if state["cut_choice"] == "MEMORY":
        return "drop_memory"

    return "build_context"


# -----------------------------
# Drop RAG
# -----------------------------
def drop_rag(state: State):

    sections = dict(state["sections"])

    sections.pop("RAG", None)

    return {
        "sections": sections,
        "dropped_section": "RAG"
    }


# -----------------------------
# Drop MEMORY
# -----------------------------
def drop_memory(state: State):

    sections = dict(state["sections"])

    sections.pop("MEMORY", None)

    return {
        "sections": sections,
        "dropped_section": "MEMORY"
    }


# -----------------------------
# Build Final Context
# -----------------------------
def build_context(state: State):

    order = ["TASK", "RAG", "MEMORY"]

    context = "\n\n".join(
        f"{section}: {state['sections'][section]}"
        for section in order
        if section in state["sections"]
    )

    return {"context": context}


# -----------------------------
# Ask Model
# -----------------------------
def ask_model(state: State):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": state["context"]
            }
        ]
    )

    return {
        "answer": response["message"]["content"]
    }


# -----------------------------
# Graph
# -----------------------------
builder = StateGraph(State)

builder.add_node("assemble_sections", assemble_sections)
builder.add_node("decide_cut", decide_cut)
builder.add_node("drop_rag", drop_rag)
builder.add_node("drop_memory", drop_memory)
builder.add_node("build_context", build_context)
builder.add_node("ask_model", ask_model)

builder.add_edge(START, "assemble_sections")
builder.add_edge("assemble_sections", "decide_cut")

builder.add_conditional_edges(
    "decide_cut",
    route_after_cut
)

builder.add_edge("drop_rag", "build_context")
builder.add_edge("drop_memory", "build_context")

builder.add_edge("build_context", "ask_model")
builder.add_edge("ask_model", END)

graph = builder.compile()

# -----------------------------
# Run
# -----------------------------
result = graph.invoke({})

print("\nDropped Section:")
print(result["dropped_section"])

print("\nFinal Context:")
print(result["context"])

print("\nAnswer:")
print(result["answer"])
