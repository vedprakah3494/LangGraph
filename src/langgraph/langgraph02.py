from typing import TypedDict
from urllib import response
from langgraph.graph import END,START,StateGraph
import ollama

class MyState(TypedDict):
    message:str


def Invokellm(state: MyState):
    response = ollama.chat(
        model="gemma3",
        messages=[
            {"role": "system", "content": "Classify the request in one word and also explain in detail in 100 word"},
            {"role": "user", "content": state['message']},
        ],
    )
    state['message'] = response['message']['content']
    return state

graph = StateGraph(MyState)
graph.add_node("invokellm", Invokellm)


graph.add_edge(START, "invokellm")  
graph.add_edge("invokellm", END)


graph = graph.compile()
result=graph.invoke({"message": "What is the weather like today?"})
print(result)
