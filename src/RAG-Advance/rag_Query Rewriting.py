import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------
# STEP 1: Load LLM
# --------------------------------------------------

llm = ChatOllama(
    model="gemma3",
    temperature=0
)

# --------------------------------------------------
# STEP 2: Load PDF
# --------------------------------------------------

loader = PyPDFLoader("../docs/Leave-Policy.pdf")
documents = loader.load()

# --------------------------------------------------
# STEP 3: Split Documents into Chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120
)

splits = text_splitter.split_documents(documents)

# --------------------------------------------------
# STEP 4: Create Embeddings
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# --------------------------------------------------
# STEP 5: Create / Load Chroma DB
# --------------------------------------------------

if not os.path.exists(".chroma_db"):
    print("Creating vector database...")

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=".chroma_db"
    )
else:
    print("Loading existing vector database...")

    vector_store = Chroma(
        persist_directory=".chroma_db",
        embedding_function=embeddings
    )

# --------------------------------------------------
# STEP 6: Create Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------------------------------------
# STEP 7: Query Rewrite Prompt
# --------------------------------------------------

rewrite_prompt = PromptTemplate.from_template(
    """
You are a search expert.

Rewrite the user's question into a concise search query
that will retrieve the most relevant information.

Return ONLY the rewritten query.

Question:
{question}

Search Query:
"""
)

# Prompt -> LLM
rewrite_chain = rewrite_prompt | llm

print("✅ Query Rewriting RAG Ready")

# --------------------------------------------------
# STEP 8: Chat Loop
# --------------------------------------------------

while True:

    user_input = input(
        "\nAsk a question (type 'exit' to quit): "
    )

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    # ----------------------------------------------
    # STEP 8A: Rewrite User Question
    # ----------------------------------------------

    rewritten_response = rewrite_chain.invoke(
        {"question": user_input}
    )

    rewritten_query = rewritten_response.content.strip()

    print("\n===== REWRITTEN QUERY =====")
    print(rewritten_query)

    # ----------------------------------------------
    # STEP 8B: Retrieve Documents
    # ----------------------------------------------

    retrieved_docs = retriever.invoke(
        rewritten_query
    )

    print("\n===== RETRIEVED CHUNKS =====")

    for index, doc in enumerate(retrieved_docs, start=1):
        print(f"\nChunk {index}:")
        print(doc.page_content[:200])

    # ----------------------------------------------
    # STEP 8C: Build Context
    # ----------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # ----------------------------------------------
    # STEP 8D: Final Answer Prompt
    # ----------------------------------------------

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

    # ----------------------------------------------
    # STEP 8E: Generate Answer
    # ----------------------------------------------

    response = llm.invoke(final_prompt)

    print("\n===== FINAL ANSWER =====")
    print(response.content)