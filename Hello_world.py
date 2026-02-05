from typing import Dict, TypedDict
 
from langgraph.graph import StateGraph


# create an agent state
class AgentState(TypedDict):
    message: str

# define a node
def greeting_node(state: AgentState) -> AgentState:
    """
        Simple node that adds a greeting message to the state.
    """
    state["message"] = "Hey " + state["message"] + ", how is your day going ?"
    return state

# create a stateGraphg
graph = StateGraph(AgentState)

graph.add_node("greeter", greeting_node) 

graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

app = graph.compile()

# display
from IPython.display import display, Image
display(Image(app.get_graph().draw_mermaid_png()))


# run
result = app.invoke({"message": "Bob"})

print(result)