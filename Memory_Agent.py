from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END


load_dotenv()

# Create a Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# State type
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Process node
def process(state: AgentState) -> AgentState:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[message.content for message in state["messages"]],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) # no extra thinking time
            )
        )
        reply_text = response.text
        print(f"FULL CONVERSATION :\n {state['messages']}")
        state["messages"].append(AIMessage(content=reply_text))
        print(f"FULL CONVERSATION :\n {state['messages']}")
        print(f"\nAI: {reply_text}")
        print("\n\n")
    except Exception as e:
        print("Error calling Gemini API:", e)
    return state

# Build the state graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Start conversation
conversation_history: List[Union[HumanMessage, AIMessage]] = []

user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]  # keep entire history

    user_input = input("Enter: ")


with open("memory_agent_conversation.txt", "w") as file:
    file.write("FULL CONVERSATION :\n\n\n")
    for message in conversation_history:
        role = "Human" if isinstance(message, HumanMessage) else "AI"
        file.write(f"{role}: {message.content}\n")

    file.write("\n\nFULL CONVERSATION --- IGNORE ---\n")

