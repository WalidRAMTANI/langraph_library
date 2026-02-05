from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, SystemMessage
from operator import add as add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-pro",
    temperature=0.0,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

pdf_path = "sqlselect_sa.pdf"

# Load and process the PDF document
if not os.path.exists(pdf_path):
    raise ValueError(f"PDF file not found at path: {pdf_path}")


pdf_loader = PyPDFLoader(pdf_path)


try:
    pages = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print(f"Loaded and split {len(pages)} pages from the PDF.")
except Exception as e:
    raise RuntimeError(f"Error loading or processing PDF: {str(e)}")

# pdf loaded in to pages

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# appyly yhe chunker on the pages
pages_split = text_splitter.split_documents(pages)

persistent_directory = "ramtani@MacBook-Air-de-RAMTANI/Documents/important\ apps/learning/computer\ science/lLangGraph"
collection_name = "sql_lecon"

if not os.path.exists(persistent_directory):
    os.makedirs(persistent_directory)


try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persistent_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # the amount of chunks to return
)


@tool 
def retriever_tool(query: str) -> str:
    """
    this tool searches and returns the information from the sqlselect_sa document
    """
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information on sqlselect_sa document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}: \n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]

model = llm.bind_tools(tools=tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state):
    """
        check if the last message contains tool calls
    """
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
    You are an intellegent AI assisntant who asnwers questions about sql lecon based on the document in sqlselect_sa.pdf.
    Use the retriever tool available to answer questions about the sql lecon data. You can make multiple calls if nedded.
    if you need to look up some information before asking a follow up question, you are allowed to that!
    Please always cite the specific parts of the documents you can use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLM agent
def call_llm(state=AgentState) -> AgentState:
    """
    function to call the LLM with the current state
    """
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}

# retriever agent
def take_action(state: AgentState) -> AgentState:
    """
        execute tool calls from LLM's response.
    """
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool : {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict: # check if valid tool
            print(f"\nTool : {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available Tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result lengh: {len(str(result))}")

        results.append(ToolMessage(content=str(result), tool_call_id=t['id'], name=t['name']))

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges("llm", 
    should_continue, 
    {True: "retriever_agent", False: END})

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()



def running_agent():
    print("\n== RAG AGENT ===")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n==== ANSWER ===")
        print(result["messages"][-1].content)

running_agent()