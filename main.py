"""## Import Required Libraries"""

import os
from dotenv import load_dotenv
load_dotenv()
from pprint import pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from composio import Composio
from composio_langchain import LangchainProvider
import asyncio
import argparse
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

"""## Load API's"""

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
COMPOSIO_USER_ID = os.getenv("COMPOSIO_USER_ID")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH")
USER_QUERY = os.getenv("USER_QUERY")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY")

if not COMPOSIO_API_KEY:
    raise ValueError("Missing COMPOSIO_API_KEY")

if not COMPOSIO_USER_ID:
    raise ValueError("Missing COMPOSIO_USER_ID")

if not EMBEDDING_MODEL:
    raise ValueError("Missing EMBEDDING_MODEL")

if not DOCUMENT_PATH:
    raise ValueError("Missing DOCUMENT_PATH")

if not USER_QUERY:
    raise ValueError("Missing USER_QUERY")

"""# Part I – Indexing

## Step 1: Load & Explore the Document
"""

loader = PyPDFLoader(DOCUMENT_PATH)
documents = loader.load()

print(f"Total Pages: {len(documents)}")

"""## Step 2: Split the Document into Chunks"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    add_start_index=True
)

all_splits = text_splitter.split_documents(documents)

print(f"Number of Chunks: {len(all_splits)}")

"""## Step 3: Initialize the Embedding Model"""

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

"""## Step 4: Create the Chroma Vector Database"""

vector_store = Chroma(
    collection_name="agentic_rag_research_collection",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db"
)
document_ids = vector_store.add_documents(all_splits)

"""### Verify Stored Embeddings"""

sample = vector_store.get(
    limit=1,
    include=["documents","embeddings"]
)

print(len(sample))

"""# Part II – Retrieval"""

def retrieve_context(query: str, k: int = 2):
  retrieved_docs = vector_store.similarity_search(query, k=k)

  docs_content = ""
  for doc in retrieved_docs:
    docs_content += f"Source: {doc.metadata}\n"
    docs_content += f"Content: {doc.page_content}\n\n"

  return docs_content, retrieved_docs

"""# PART III - Agent Tools

## Step 1: Retrieval Tool
"""

@tool
def retrieve_from_pdf(query:str)->str:
    """
    Retrieve information from the uploaded research paper.
    Use this tool for questions related to the paper.
    """

    context, docs = retrieve_context(query)

    return context

"""## Step 2 : Instialise Composio"""

composio = Composio(
    api_key=COMPOSIO_API_KEY,
    provider=LangchainProvider()
)

"""## Step 3 : Load MCP Tools"""

mcp_tools = composio.tools.get(
    user_id=COMPOSIO_USER_ID,
    toolkits=[
        "TAVILY"
    ]
)

all_tools = mcp_tools + [retrieve_from_pdf]

"""# PART IV - Agentic Reasoning Layer

## Step 1: Configure LLM
"""

model = init_chat_model(
    f"google_genai:{GEMINI_MODEL}",
    api_key=GEMINI_API_KEY,
    temperature=0.2
)

"""## Step 2: Define System Prompt"""

SYSTEM_PROMPT = """

You are an expert research assistant.

Available tools:

1. retrieve_from_pdf

Use this tool for:
- information inside the uploaded paper
- methodology
- architecture
- experiments
- contributions


2. TavilySearch

Use this tool for:
- recent research
- latest frameworks
- information after publication
- external comparisons


Decision rules:

- Paper-only questions:
  Use retrieve_from_pdf.

- Current information questions:
  Use TavilySearch.

- Comparison questions:
  Use both tools.

- Never hallucinate information.
- Clearly distinguish paper information and external information.

"""

"""## Step 3: Initialize the Memory"""

checkpointer = InMemorySaver()

MemoryConfig = {
    "configurable": {
        "thread_id": "1"
    }
}

"""## Step 3: Create the Agent"""

agent = create_agent(
    model=model,
    tools=all_tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    debug = False
)

"""# PART V — Application Layer

## Step 1: Agentic RAG Chat Function
"""

async def docu_chat(user_query):

    try:
        response = await agent.ainvoke(
            {
                "messages":[
                    {
                        "role":"user",
                        "content":user_query
                    }
                ]
            },config=MemoryConfig
        )

        return response

    except Exception as e:
        print("Error:",e)
        return None

"""## Helper Function for Clean Output"""

def get_final_answer(response):

    return response["messages"][-1].content[0]['text']

"""# Agentic RAG Test

## Test 1: Question only from PDF
"""

parser = argparse.ArgumentParser(description="Agentic RAG DocuChat")
parser.add_argument(
    "--query",
    default=USER_QUERY,
    help="Override USER_QUERY from .env"
)

args = parser.parse_args()
async def main():
    
    response = await docu_chat(args.query)

    if response is None:
        return

    print(f"\nUser Query:\n{args.query}\n")

    print("\nAssistant Response:\n")
    print(get_final_answer(response))


if __name__ == "__main__":
    asyncio.run(main())
