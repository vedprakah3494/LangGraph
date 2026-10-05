import sys
from langchain_protocol import TypedDict
import ollama
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import sqliteSaver

sys.stdout.reconfigure(encoding='utf-8')
 
class chatstate(TypedDict,total=False):
    user_message: str
    chat_history: list[str]
    reply: str

def chatbot_node(state: chatstate):
    history = state.get('chat_history', [])+[{"role": "user", "content": state.get('user_message', '')}]
    response = ollama.chat(
        model="gemma3",
        messages=[history,],
    )
    state['reply'] = response['message']['content']
    history.append({"role": "assistant", "content": state['reply']})
    state['chat_history'] = history
    return {"reply": state['reply'], "chat_history": state['chat_history']}


graph_workflow = StateGraph(chatstate)
graph_workflow.add_node("chatbot_node", chatbot_node)  


graph_workflow.add_edge(START, "chatbot_node")
graph_workflow.add_edge("chatbot_node", END)

with sqliteSaver("chatbot_state.db") as saver:
    graph = graph_workflow.compile(checkpointer=saver)
    settings = {"configurable":{"thread_id":"chatbot_no_state"} }
    continue_conversation = True
    while continue_conversation:
        user_input = input("User: ")
        result = graph.invoke({"user_message": user_input},config=settings)
        print("Assistant:", result['reply'])
        continue_conversation = input("Do you want to continue the conversation? (yes/no): ").strip().lower() == "yes"

