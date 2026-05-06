# Hasnat's Chatbot  Multi-Agent AI Support System

A production-grade customer support chatbot built with a multi-agent architecture. The system routes each customer query to a specialized agent based on the topic, with each agent searching its own isolated knowledge base to generate accurate responses.

Persistent conversation memory is stored in PostgreSQL, meaning history survives page refreshes and is maintained across sessions. Responses are streamed word by word to the frontend in real time.

---

## How It Works

Every incoming message goes through a supervisor agent that classifies the query and routes it to one of four specialized agents. Each agent has its own Pinecone namespace containing only its relevant documents, so a shipping question never touches billing data and vice versa.

The conversation history for each session is stored in PostgreSQL and passed to the agent on every request, giving it full memory of the conversation without relying on browser storage.

---

## Agents

**Supervisor**
Classifies the incoming query and routes it to the correct department. Does not answer questions directly.

**Shipping Agent**
Handles delivery timelines, order tracking, shipping options, lost packages and international shipping queries.

**Returns Agent**
Handles return eligibility, return initiation, damaged item reports and refund processing.

**Billing Agent**
Handles payment methods, failed transactions, invoices, duplicate charges and discount codes.

**Account Agent**
Handles password resets, account lockouts, two-factor authentication, loyalty points and profile settings.

---

## Tech Stack

- LangGraph  (multi-agent graph orchestration)
- LangChain  (agent and RAG framework)
- OpenAI GPT-4o-mini
- Pinecone  (vector database with namespace isolation per agent)
- PostgreSQL  (persistent session and message storage)
- FastAPI  (async backend API with streaming support)
- Server-Sent Events  (real-time word-by-word response streaming)
- LangSmith (agent observability and performance monitoring)

---

## Project Structure

```
project3/
├── database.py          # PostgreSQL session and message management
├── tools.py             # RAG search tools scoped per Pinecone namespace
├── graph.py             # LangGraph multi-agent workflow
├── main.py              # FastAPI backend with streaming endpoint
├── knowledge_base.py    # PDF ingestion pipeline into Pinecone
├── requirements.txt
├── .env.example
└── frontend/
    ├── index.html
    └── images/
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment variables**

Copy `.env.example` to `.env` and fill in your credentials:
```
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_CLOUD=
PINECONE_REGION=
DATABASE_URL=postgresql://username:password@localhost:5432/chatbot_db
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT= Customer-support-agent
```

**3. Create PostgreSQL database**

Create a database named `chatbot_db` using pgAdmin or psql. The tables are created automatically on first server start.

**4. Ingest knowledge base documents**
```bash
python knowledge_base.py
```

**5. Start the server**
```bash
uvicorn main:app --reload
```

Open `http://localhost:8000`

---

## Key Implementation Details

**Namespace isolation**  Each agent's knowledge base lives in a separate Pinecone namespace. The retriever is scoped at query time so agents never cross-contaminate each other's data.

**Persistent memory**  Each browser session gets a UUID stored in localStorage. On every request, the full message history is loaded from PostgreSQL and passed to the agent as context.

**Streaming** — The `/chat/stream` endpoint uses FastAPI's `StreamingResponse` with Server-Sent Events. The frontend processes the stream with a `ReadableStream` reader and renders tokens as they arrive.

**Observability**  LangSmith traces every run automatically when the environment variables are set. No additional instrumentation required.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index |
| `PINECONE_CLOUD` | Pinecone cloud provider |
| `PINECONE_REGION` | Pinecone region |
| `DATABASE_URL` | PostgreSQL connection string |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | LangSmith project name |
