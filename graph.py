import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from tools import (
    shipping_rag_tool,
    returns_rag_tool,
    billing_rag_tool,
    account_rag_tool
)


load_dotenv()


class AgentState(TypedDict):
    user_message: str
    chat_history: list
    category: str
    response: str


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY")
)


def create_specialized_agent(rag_tool, system_prompt: str) -> AgentExecutor:
    """
    Creates a specialized LangChain agent with its own tool and prompt.

    Args:
        rag_tool: The RAG tool scoped to this agent's namespace.
        system_prompt: Department-specific instructions.

    Returns:
        An AgentExecutor ready to handle queries.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=[rag_tool],
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=[rag_tool],
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True
    )


SHIPPING_PROMPT = """You are a Shipping Support Specialist for ShopEasy.
You handle questions about delivery, tracking, shipping options, and lost packages.
Always search the knowledge base before answering.
Be empathetic if a package is missing or delayed.
Keep responses clear, professional, and concise."""

RETURNS_PROMPT = """You are a Returns and Refunds Specialist for ShopEasy.
You handle questions about returns, refunds, damaged items and exchanges.
Always search the knowledge base before answering.
Be understanding — customers requesting refunds are often frustrated.
Keep responses clear, professional and concise."""

BILLING_PROMPT = """You are a Billing Support Specialist for ShopEasy.
You handle questions about payments, invoices, charges and discount codes.
Always search the knowledge base before answering.
Never ask for full card numbers — only last 4 digits if needed.
Keep responses clear, professional, and concise."""

ACCOUNT_PROMPT = """You are an Account Support Specialist for ShopEasy.
You handle questions about account access, passwords, loyalty points, and settings.
Always search the knowledge base before answering.
Be careful with security-related issues.
Keep responses clear, professional and concise."""


shipping_agent = create_specialized_agent(shipping_rag_tool, SHIPPING_PROMPT)
returns_agent  = create_specialized_agent(returns_rag_tool,  RETURNS_PROMPT)
billing_agent  = create_specialized_agent(billing_rag_tool,  BILLING_PROMPT)
account_agent  = create_specialized_agent(account_rag_tool,  ACCOUNT_PROMPT)


def supervisor_node(state: AgentState) -> dict:
    """
    Classifies the user message into one of 4 categories.
    Does not answer — only routes.
    """
    classification_prompt = f"""You are a customer support routing system.

Classify the following message into EXACTLY one of these categories:
- shipping  -> delivery, tracking, shipment, lost package, delays
- returns   -> returns, refunds, damaged items, exchanges
- billing   -> payments, invoices, charges, discount codes
- account   -> login, password, loyalty points, profile, settings

Customer message: "{state['user_message']}"

Respond with ONLY the category word. Nothing else."""

    result = llm.invoke(classification_prompt)
    category = result.content.strip().lower()

    valid = ["shipping", "returns", "billing", "account"]
    if category not in valid:
        category = "shipping"

    print(f"[Supervisor] Routed to: {category}")
    return {"category": category}


def _run_agent_node(agent: AgentExecutor, state: AgentState) -> dict:
    """
    Helper that runs any specialized agent with the current state.
    Converts chat history dicts to LangChain message objects.
    """
    history = []
    for msg in state["chat_history"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    result = agent.invoke({
        "input": state["user_message"],
        "chat_history": history
    })

    return {"response": result["output"]}


def shipping_node(state: AgentState) -> dict:
    return _run_agent_node(shipping_agent, state)

def returns_node(state: AgentState) -> dict:
    return _run_agent_node(returns_agent, state)

def billing_node(state: AgentState) -> dict:
    return _run_agent_node(billing_agent, state)

def account_node(state: AgentState) -> dict:
    return _run_agent_node(account_agent, state)


def route_to_agent(state: AgentState) -> Literal["shipping", "returns", "billing", "account"]:
    return state["category"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("shipping",   shipping_node)
    graph.add_node("returns",    returns_node)
    graph.add_node("billing",    billing_node)
    graph.add_node("account",    account_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "shipping": "shipping",
            "returns":  "returns",
            "billing":  "billing",
            "account":  "account",
        }
    )

    graph.add_edge("shipping", END)
    graph.add_edge("returns",  END)
    graph.add_edge("billing",  END)
    graph.add_edge("account",  END)

    return graph.compile()

support_graph = build_graph()


def run_graph(user_message: str, chat_history: list) -> dict:
    """
    Runs the multi-agent graph and returns the final response.

    Args:
        user_message: The customer's latest message.
        chat_history: Previous conversation as list of dicts.

    Returns:
        Dict with 'response' and 'category'.
    """
    initial_state = AgentState(
        user_message=user_message,
        chat_history=chat_history,
        category="",
        response=""
    )

    try:
        final_state = support_graph.invoke(initial_state)
        return {
            "response": final_state["response"],
            "category": final_state["category"]
        }
    except Exception as e:
        return {
            "response": f"I encountered an error. Please try again. (Error: {str(e)})",
            "category": "unknown"
        }
