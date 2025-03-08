from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langchain_core.messages import ToolMessage
from typing_extensions import TypedDict
import json
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import os

load_dotenv()
app = FastAPI(title="AI Research Agentic System")
class QueryRequest(BaseModel):
    query: str

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# This is short term memory
memory = MemorySaver()

config = {"configurable": {"thread_id": "1"}}
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Initialize LLM with Groq
llm1 = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.1,
    max_tokens=2048, 
    api_key=os.getenv("GROQ_API_KEY")
)
llm2 = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.1,
    max_tokens=2048, 
    api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm2.bind_tools(tools)
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
# Define tools
graph_builder = StateGraph(State)


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
tool_node = BasicToolNode(tools=[tool])

def chatbot(state: State):
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):

    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, 
                              config
                              ):
        output = []
        for value in event.values():
            output.append({value["messages"][-1].content })
    return { "response": output[-1] } 


@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Research Agentic System API"}



@app.post("/query")
async def process_query(user_input: QueryRequest):
    try:
        response = stream_graph_updates(user_input=user_input.query)
        return response
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
