
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3"
PDF_FILE_PATH = "../docs/Leave-Policy.pdf"

# Load PDF
pdf_loader = PyPDFLoader(PDF_FILE_PATH)
documents = pdf_loader.load()

# Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120
)
splits = text_splitter.split_documents(documents)

# Create vector store
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

if not os.path.exists(".chroma_db"):
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=".chroma_db"
    )
else:
    vector_store = Chroma(
        persist_directory=".chroma_db",
        embedding_function=embeddings
    )
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

# HyDE Prompt
hyde_prompt = PromptTemplate.from_template("""
You are helping a retrieval system.

Generate a short hypothetical document paragraph that would
likely answer the user's question.

Write in the style of a formal policy document.

Question:
{question}

Hypothetical Document:
""")

hyde_chain = hyde_prompt | llm

print("✅ HyDE RAG Ready")

while True:
    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # Step 1: Generate hypothetical document
    hypothetical_doc = hyde_chain.invoke(
        {"question": query}
    ).content.strip()

    print("\n===== GENERATED HyDE DOCUMENT =====")
    print(hypothetical_doc)

    # Step 2: Retrieve relevant chunks using HyDE output
    retrieved_docs = retriever.invoke(hypothetical_doc)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Step 3: Final answer generation
    final_prompt = f"""
Answer ONLY using the provided context.

If the answer is not available in the context, reply:
'I could not find that information in the document.'

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(final_prompt)

    print("\n===== FINAL ANSWER =====")
    print(response.content)