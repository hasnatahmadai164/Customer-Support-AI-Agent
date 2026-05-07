import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# PDF TO NAMESPACE MAPPING
PDF_NAMESPACE_MAP = {
    r"D:\New folder 2\docs\shipping_policy.pdf": "shipping",
    r"D:\New folder 2\docs\returns_policy.pdf":  "returns",
    r"D:\New folder 2\docs\billing_policy.pdf":  "billing",
    r"D:\New folder 2\docs\account_policy.pdf":  "account",
}

# TEXT SPLITTER CONFIGURATION
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,

    length_function=len,

    separators=["\n\n", "\n", ". ", " ", ""]
)


def load_and_chunk_pdf(pdf_path: str) -> list:
    """
    Loads a PDF file and splits it into chunks.

    HOW PyPDFLoader WORKS:
    PyPDFLoader reads the PDF page by page.
    Each page becomes a LangChain Document object with:
      - page_content: the extracted text of that page
      - metadata: {"source": "filepath", "page": page_number}

    Then RecursiveCharacterTextSplitter splits those Documents
    into smaller chunks, preserving the metadata from each page.

    Args:
        pdf_path: Path to the PDF file e.g. 'docs/shipping_policy.pdf'

    Returns:
        List of Document objects (chunks), each with text and metadata.
    """

    print(f"  Loading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"  Pages loaded: {len(pages)}")

    chunks = text_splitter.split_documents(pages)

    print(f"  Chunks created: {len(chunks)}")

    return chunks


def create_pinecone_index(pc: Pinecone) -> object:
    """
    Creates Pinecone index if it doesn't exist, then returns the Index object.
    Pinecone v8 pattern — same as Project 1.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not pc.has_index(index_name):
        print(f"Creating Pinecone index: '{index_name}'...")

        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD"),
                region=os.getenv("PINECONE_REGION")
            )
        )

        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print("Index is ready.")

    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")

    return pc.Index(index_name)


def ingest_pdfs():
    """
    Main ingestion function. Loads each PDF, chunks it,
    and uploads the chunks to the correct Pinecone namespace.

    FULL FLOW PER PDF:
    1. PyPDFLoader reads the PDF -> list of page Documents
    2. RecursiveCharacterTextSplitter splits pages -> list of chunk Documents
    3. Each chunk gets embedded by OpenAI -> 1536-dimensional vector
    4. Vector + original text + metadata uploaded to Pinecone namespace
    """

    print("=" * 55)
    print("ShopEasy Knowledge Base — PDF Ingestion")
    print("=" * 55)

    for pdf_path in PDF_NAMESPACE_MAP.keys():
        if not os.path.exists(pdf_path):
            print(f"ERROR: PDF not found: {pdf_path}")
            print("Please run: python create_pdfs.py first.")
            return

    # Initialize Pinecone client 
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = create_pinecone_index(pc)

    # Initialize OpenAI embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    total_chunks = 0

    for pdf_path, namespace in PDF_NAMESPACE_MAP.items():

        print(f"\nProcessing: {pdf_path} -> namespace: '{namespace}'")

        chunks = load_and_chunk_pdf(pdf_path)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=namespace
        )

        vectorstore.add_documents(documents=chunks)

        total_chunks += len(chunks)
        print(f"  Uploaded {len(chunks)} chunks to namespace '{namespace}'")

    print()
    print("=" * 55)
    print(f"Ingestion complete!")
    print(f"Total chunks uploaded: {total_chunks}")
    print(f"Namespaces: {list(PDF_NAMESPACE_MAP.values())}")
    print()
    print("Next step: uvicorn main:app --reload")
    print("=" * 55)


if __name__ == "__main__":
    ingest_pdfs()
