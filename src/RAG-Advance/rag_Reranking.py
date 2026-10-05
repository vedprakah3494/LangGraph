import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

PDF_PATH = "../docs/Leave-Policy.pdf"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3"
CHROMA_PATH = ".chroma_db"

# ----------------------------------------------------
# Load PDF
# ----------------------------------------------------

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# ----------------------------------------------------
# Split Documents
# ----------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120
)

chunks = text_splitter.split_documents(documents)

# ----------------------------------------------------
# Embeddings
# ----------------------------------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# ----------------------------------------------------
# Create / Load Chroma
# ----------------------------------------------------

if not os.path.exists(CHROMA_PATH):

    print("Creating Chroma DB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

else:

    print("Loading Existing Chroma DB...")

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

# ----------------------------------------------------
# Retrieve Top 5 Candidates
# ----------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# ----------------------------------------------------
# LLM
# ----------------------------------------------------

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

# ----------------------------------------------------
# Cross Encoder Reranker
# ----------------------------------------------------

print("Loading CrossEncoder...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("✅ Reranking RAG Ready")

# ----------------------------------------------------
# Chat Loop
# ----------------------------------------------------

while True:

    question = input(
        "\nAsk a question (type 'exit' to quit): "
    )

    if question.lower() == "exit":
        break

    # ------------------------------------------------
    # STEP 1 - Initial Retrieval
    # ------------------------------------------------

    retrieved_docs = retriever.invoke(question)

    print("\n===== RETRIEVED CHUNKS =====")

    for idx, doc in enumerate(retrieved_docs, start=1):
        print(f"\nCandidate {idx}")
        print(doc.page_content[:200])

    # ------------------------------------------------
    # STEP 2 - Build Question/Chunk Pairs
    # ------------------------------------------------

    pairs = [
        [question, doc.page_content]
        for doc in retrieved_docs
    ]

    # ------------------------------------------------
    # STEP 3 - CrossEncoder Scoring
    # ------------------------------------------------

    scores = reranker.predict(pairs)

    print("\n===== RERANKING SCORES =====")

    for i, score in enumerate(scores):
        print(f"Chunk {i+1}: {score:.4f}")

    # ------------------------------------------------
    # STEP 4 - Combine Docs + Scores
    # ------------------------------------------------

    scored_docs = list(
        zip(retrieved_docs, scores)
    )

    # ------------------------------------------------
    # STEP 5 - Sort by Score Descending
    # ------------------------------------------------

    reranked_docs = sorted(
        scored_docs,
        key=lambda item: item[1],
        reverse=True
    )

    # ------------------------------------------------
    # STEP 6 - Keep Top 2
    # ------------------------------------------------

    top_docs = reranked_docs[:2]

    print("\n===== TOP CHUNKS AFTER RERANKING =====")

    for rank, (doc, score) in enumerate(top_docs, start=1):

        print(f"\nRank {rank}")
        print(f"Score: {score:.4f}")
        print(doc.page_content)

    # ------------------------------------------------
    # STEP 7 - Build Context
    # ------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc, score in top_docs
    )

    # ------------------------------------------------
    # STEP 8 - Final Prompt
    # ------------------------------------------------

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
reply exactly:

"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

    # ------------------------------------------------
    # STEP 9 - Generate Answer
    # ------------------------------------------------

    response = llm.invoke(prompt)

    print("\n===== FINAL ANSWER =====")
    print(response.content)