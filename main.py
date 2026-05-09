import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import psycopg2.extras
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from graph import (
        shipping_agent, returns_agent,
        billing_agent, account_agent
        )
from langchain_core.messages import HumanMessage, AIMessage
from database import (
    init_database,
    create_session,
    save_message,
    get_chat_history,
    session_exists
)

load_dotenv()

app = FastAPI(
    title="Hasnat's Chatbot API",
    description="Production-grade multi-agent support system with persistent memory",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_database()
    print("Server started. Database ready.")

frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
images_dir   = os.path.join(frontend_dir, "images")

if os.path.exists(images_dir):
    app.mount("/images", StaticFiles(directory=images_dir), name="images")


class StreamRequest(BaseModel):
    """Request body for the streaming chat endpoint."""
    session_id: str
    user_message: str


class SessionResponse(BaseModel):
    """Response when a new session is created."""
    session_id: str


class HistoryMessage(BaseModel):
    """A single message in the history response."""
    role: str
    content: str
    category: str = ""


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(frontend_dir, "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")


@app.post("/session", response_model=SessionResponse)
async def new_session():
    """Creates a new session and returns its UUID."""
    session_id = create_session()
    return SessionResponse(session_id=session_id)


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """
    Returns the full chat history for a session.
    Called on page load to restore previous conversation.
    """
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    conn = __import__('database').get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""
        SELECT role, content, category
        FROM messages
        WHERE session_id = %s
        ORDER BY created_at ASC
    """, (session_id,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return {"messages": [dict(row) for row in rows]}


@app.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    """
    Streams the agent response word by word using Server-Sent Events.
    """

    if not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    save_message(
        session_id=request.session_id,
        role="user",
        content=request.user_message
    )

    async def event_generator():
        """
        Async generator that yields SSE-formatted chunks.

        WHY ASYNC GENERATOR?
        asyncio allows the server to handle other requests while
        waiting for the LLM to generate the next token.
        yield sends each chunk immediately without waiting for all.
        """

        full_response = ""
        category      = "unknown"

        try:
            chat_history = get_chat_history(request.session_id)

            supervisor_state = AgentState(
                user_message=request.user_message,
                chat_history=chat_history,
                category="",
                response=""
            )

            supervisor_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )

            classification_prompt = f"""Classify this message into one of: shipping, returns, billing, account.
Customer message: "{request.user_message}"
Respond with ONLY the category word."""

            cat_result = supervisor_llm.invoke(classification_prompt)
            category   = cat_result.content.strip().lower()

            if category not in ["shipping", "returns", "billing", "account"]:
                category = "shipping"

            yield f"data: {json.dumps({'type': 'category', 'content': category})}\n\n"


            agent_map = {
                "shipping": shipping_agent,
                "returns":  returns_agent,
                "billing":  billing_agent,
                "account":  account_agent,
            }
            selected_agent = agent_map[category]

            lc_history = []
            for msg in chat_history:
                if msg["role"] == "user":
                    lc_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_history.append(AIMessage(content=msg["content"]))

   
            async for event in selected_agent.astream_events(
                {"input": request.user_message, "chat_history": lc_history},
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk_text = event["data"]["chunk"].content
                    if chunk_text:
                        full_response += chunk_text

                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_text})}\n\n"
                        await asyncio.sleep(0)

            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

            save_message(
                session_id=request.session_id,
                role="assistant",
                content=full_response,
                category=category
            )

        except Exception as e:
            error_msg = f"An error occurred. Please try again."
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            print(f"Streaming error: {str(e)}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
    
    
    @app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Hasnat's Chatbot API is running"}
