import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

PDF_PATH = "../docs/Leave-Policy.pdf"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3"
CHROMA_DB_PATH = ".chroma_db"

# -------------------------------------------------------
# Load PDF
# -------------------------------------------------------

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# -------------------------------------------------------
# Split Documents
# -------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
)

splits = text_splitter.split_documents(documents)

# -------------------------------------------------------
# Embedding Model
# -------------------------------------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# -------------------------------------------------------
# Create / Load Vector Store
# -------------------------------------------------------

if not os.path.exists(CHROMA_DB_PATH):
    print("Creating Chroma Vector Store...")

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
else:
    print("Loading Existing Chroma Vector Store...")

    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# -------------------------------------------------------
# LLM
# -------------------------------------------------------

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)

# -------------------------------------------------------
# Multi Query Prompt
# -------------------------------------------------------

multi_query_prompt = PromptTemplate.from_template(
    """
You are an expert search assistant.

Generate 4 different search queries for the user's question.

Each query should:
- Ask the same question differently
- Use synonyms where appropriate
- Explore different wording styles

Return ONLY the queries.
One query per line.
No numbering.
No explanation.

Question:
{question}
"""
)

# LCEL Chain
multi_query_chain = multi_query_prompt | llm

print("✅ Multi-Query RAG Ready")

# -------------------------------------------------------
# Chat Loop
# -------------------------------------------------------

while True:

    print("\n--------------------------------------")

    user_input = input(
        "\nAsk a question (type 'exit' to quit): "
    )

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # ---------------------------------------------------
    # STEP 1: Generate Multiple Queries
    # ---------------------------------------------------

    generated_queries = multi_query_chain.invoke(
        {"question": user_input}
    )

    queries_text = generated_queries.content.strip()

    print("\n===== GENERATED QUERIES =====")
    print(queries_text)

    query_list = [
        q.strip()
        for q in queries_text.split("\n")
        if q.strip()
    ]

    # ---------------------------------------------------
    # STEP 2: Retrieve Docs For Each Query
    # ---------------------------------------------------

    all_docs = []

    print("\n===== RETRIEVED CHUNKS =====")

    for index, query in enumerate(query_list, start=1):

        print(f"\nQuery {index}: {query}")

        docs = retriever.invoke(query)

        for doc in docs:
            print(f"→ {doc.page_content[:120]}...")
            all_docs.append(doc)

    # ---------------------------------------------------
    # STEP 3: Remove Duplicate Chunks
    # ---------------------------------------------------

    unique_docs = []
    seen = set()

    for doc in all_docs:

        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    print(
        f"\nUnique Chunks Retrieved: {len(unique_docs)}"
    )

    # ---------------------------------------------------
    # STEP 4: Build Context
    # ---------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in unique_docs
    )

    # ---------------------------------------------------
    # STEP 5: Final Answer Prompt
    # ---------------------------------------------------

    final_prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly with:

"I could not find that information in the document."

Context:
{context}

Question:
{user_input}
"""

    # ---------------------------------------------------
    # STEP 6: Generate Final Answer
    # ---------------------------------------------------

    response = llm.invoke(final_prompt)

    print("\n===== FINAL ANSWER =====")
    print(response.content)