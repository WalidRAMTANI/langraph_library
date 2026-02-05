from typing import Annotated, TypedDict, Sequence, List
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# this a globale variable ro store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """ function to update the document content."""
    global document_content
    document_content = content
    return f"Document content updated. Current content : {document_content}."

@tool
def save(filename: str) -> str:
    """ function to save the document content to a file.
    """
    global document_content
    if not filename.endswith(".txt"):
        filename += ".txt"
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        
        print(f"Document content saved to {filename}.")
        return f"Document content has been saved successfully to {filename}."
    except Exception as e:
        return f"Failed to save document content to {filename}. Error: {str(e)}"
    
tools = [update, save]

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
def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""You are a Drafter writing assistant that helps users to update and modify text documents.
        - if the user wants to udpate the document, use the 'update' tool with the complete updated version.
        - if the user wants to save the document, use the 'save' tool with the filename.
        - Make sure to always show the current content of the document after each update.
        """
    )
    if not state["messages"]:
        user_message = HumanMessage(content="I'm ready to help you draft your document. What would you like to create ?")
    else:
        user_input = input("\nWhat would you like to do with document ?")
        #print(f"\nUser input: {user_input}\n")
        user_message = HumanMessage(content=user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\nAI response: {response.content}\n")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS : {[tc['name'] for tc in response.tool_calls]}\n")
    
    return {"messages": list(state["messages"]) + [user_message, response]}

# question do we add 'document' in message.content --------------------------

def should_continue(state: AgentState):
    """"
        Conditional function to decide whether to continue or end
    """
    messages = state["messages"]
    if not state["messages"]:
        return "continue"

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"

    return "continue"


graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", 
    should_continue,
    {
     "continue": "agent", 
     "end": END
     }
)

app = graph.compile()


def print_messages(messages: List[BaseMessage]) -> None:
    """Prints the messages in the conversation."""
    for message in messages:
        if isinstance(message, HumanMessage):
            print(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool: {message.content}")
        else:
            print(f"System: {message}")


def run_document_agent():
    print("Welcome to Drafter Agent. You can update and save your document.")

    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if 'messages' in step:
            print_messages(step['messages'])



if __name__ == "__main__":
    run_document_agent()