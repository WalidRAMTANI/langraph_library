# React Agent with LangGraph and Gemini API

from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
# import necessary modules
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """ function to add two numbers."""
    return a + b

@tool
def multiply(a:int, b:int):
    """ function to multiply two numbers."""
    return a * b

@tool
def subtract(a:int, b:int):
    """ function to subtract two numbers."""
    return a - b

tools = [add, multiply, subtract]


# Create LLM class
llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-pro",
    temperature=1.0,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Bind tools to the model
model = llm.bind_tools(tools)


# Process node
# ----- Process node -----
def process(state: AgentState) -> AgentState:
    try:
        # Invoke the model
        response = model.invoke(state["messages"])
        # Append the LLM response
        state["messages"].append(response)
        return state

    except Exception as e:
        print("Error calling Gemini API:", e)
        return state

# Conditional function to decide whether to continue or end
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # Only AIMessage can have tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"


# Build the state graph
graph = StateGraph(AgentState)
graph.add_node("agent", process)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", 
    should_continue,
    {
     "continue": "tools", 
     "end": END
     }
)

graph.add_edge("tools", "agent")

agent = graph.compile()


# Initial message
inputs = {"messages": [("user", "add 2 + 3 then  multiply the result by 4. after that tell me a joke")]}

for state in agent.stream(inputs, stream_mode="values"):
    last_message = state["messages"][-1]
    
    # If the last message is an AIMessage with tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        state["messages"][-1].pretty_print()
    # Print the last message (AIMessage or ToolMessage)
    last_message = state["messages"][-1]
    if hasattr(last_message, "pretty_print"):
        last_message.pretty_print()
    else:
        print(last_message)
