

from typing import TypedDict

from langgraph.graph import END,START,StateGraph

class MyState(TypedDict):
    a:int
    b:int
    c:int

def Input(state:MyState):
    state['a']=10
    state['b']=20
    return state


def Calculate(state:MyState):
    state['c']=state['a']+state['b']
    return state

graph=StateGraph(MyState)
graph.add_node("input",Input)
graph.add_node("calculate",Calculate)

graph.add_edge(START,"input")
graph.add_edge("input","calculate")
graph.add_edge("calculate",END)

mygraph=graph.compile()
result=mygraph.invoke({"a":10,"b":20,"c":0})

print(result)



