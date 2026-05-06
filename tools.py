import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()


def get_retriever(namespace: str):
    """
    Creates a Pinecone retriever scoped to a specific namespace.

    Each specialized agent calls this with its own namespace
    so it only searches its own knowledge base.

    Args:
        namespace: One of 'shipping', 'returns', 'billing', 'account'

    Returns:
        A LangChain retriever searching only that namespace.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def create_rag_tool(namespace: str):
    """
    Factory function that creates a RAG search tool for a specific namespace.

    Why a factory?
    The @tool decorator requires each tool to have a unique function name.
    We cannot reuse the same function for 4 different namespaces.
    The factory creates a uniquely named function each time.

    Args:
        namespace: Pinecone namespace this tool searches.

    Returns:
        A LangChain tool function scoped to that namespace.
    """

    def search_fn(query: str) -> str:
        try:
            retriever = get_retriever(namespace)
            docs = retriever.invoke(query)

            if not docs:
                return f"No relevant information found in the {namespace} knowledge base."

            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "unknown")
                results.append(f"[Source {i}: {source}]\n{doc.page_content}")

            return "\n\n---\n\n".join(results)

        except Exception as e:
            return f"Error searching {namespace} knowledge base: {str(e)}"

    search_fn.__name__ = f"search_{namespace}_knowledge_base"
    search_fn.__doc__ = (
        f"Search the {namespace} knowledge base to find accurate answers "
        f"for customer questions about {namespace}. "
        f"Always use this tool before answering any {namespace} related question."
    )

    return tool(search_fn)


shipping_rag_tool = create_rag_tool("shipping")
returns_rag_tool  = create_rag_tool("returns")
billing_rag_tool  = create_rag_tool("billing")
account_rag_tool  = create_rag_tool("account")
