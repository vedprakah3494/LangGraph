
import sqlite3

import ollama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing import TypedDict


class ChatState(TypedDict, total=False):
    user_message: str
    chat_history: list[dict]
    reply: str


def chatbot(state):
    history = state.get("chat_history", []) + [{"role": "user", "content": state["user_message"]}]
    response = ollama.chat(model="gemma3", messages=history)
    reply = response["message"]["content"]
    history = history + [{"role": "assistant", "content": reply}]
    return {"chat_history": history, "reply": reply}


graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Same db file as chatbot_memory.py -> both programs share saved history.
# Use check_same_thread=False to allow access from multiple threads
connection = sqlite3.connect("chatbot_memory.db", check_same_thread=False) 
checkpointer = SqliteSaver(connection)
graph = graph_builder.compile(checkpointer=checkpointer)