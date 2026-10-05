import os
from typing import List, Literal
from typing_extensions import TypedDict

# Core LangChain & Chroma components (from your snippet)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph state machine components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ==========================================
# 1. DATA INGESTION (Your Provided Snippet)
# ==========================================
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3"
PDF_FILE_PATH = "../docs/Leave-Policy.pdf"

if not os.path.exists(PDF_FILE_PATH):
    # Safe fallback folder creation so the code doesn't crash on run
    os.makedirs(os.path.dirname(PDF_FILE_PATH), exist_ok=True)
    print(f"⚠️ '{PDF_FILE_PATH}' not found. Please place your actual PDF at that path.")
    # Creating a dummy text layout if file is missing for compilation survival
    raise FileNotFoundError(f"Please place your PDF file at: {os.path.abspath(PDF_FILE_PATH)}")

print(f"📄 Loading PDF document from: {PDF_FILE_PATH}...")
loader = PyPDFLoader(PDF_FILE_PATH)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
splits = text_splitter.split_documents(pages)
print(f"✂️ Extracted and split PDF into {len(splits)} text chunks.")

print("📦 Generating vector embeddings for PDF content...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("✅ Vector store initialization complete. Ready for LangGraph queries.\n")

# Initialize Local Chat LLM
llm = ChatOllama(model=LLM_MODEL, temperature=0)

# ==========================================
# 2. STATE DEFINITION & ROUTING LOGIC
# ==========================================
class AgentState(TypedDict):
    """The shared memory blueprint modified by graph nodes."""
    query: str
    context: str
    response: str

def intent_router(state: AgentState) -> Literal["retrieve_pdf_context", "direct_chat_respond"]:
    """Router Pattern: Categorizes incoming inputs to bypass DB overhead for basic talk."""
    query_text = state["query"]
    
    system_instruction = (
        "You are an intent routing classifier. Analyze the user message and respond with exactly one word:\n"
        "- If the user is asking about business rules, leaves, corporate policy, holiday queries, or document data, reply: RETRIEVE\n"
        "- If the user is greeting you, saying goodbye, or doing casual small talk, reply: CHAT\n"
        "Output ONLY the word 'RETRIEVE' or 'CHAT'."
    )
    
    decision = llm.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=query_text)
    ]).content.strip().upper()
    
    if "RETRIEVE" in decision:
        return "retrieve_pdf_context"
    return "direct_chat_respond"

# ==========================================
# 3. GRAPH NODES (EXECUTIVE ENGINE)
# ==========================================
def retrieve_pdf_context(state: AgentState):
    """Fetches matches from the loaded PDF vectors inside Chroma."""
    print("🤖 [Node: Retrieve] Scanning PDF vector space...")
    matched_docs = retriever.invoke(state["query"])
    # Combine matching chunks into a single reference block
    combined_context = "\n---\n".join([doc.page_content for doc in matched_docs])
    return {"context": combined_context}

def generate_grounded_answer(state: AgentState):
    """Generates an answer strictly bounded by the retrieved PDF snippets."""
    print("🤖 [Node: Generate Answer] Structuring grounded response...")
    prompt_envelope = (
        f"You are an HR compliance assistant. Answer the user's question using ONLY the PDF document excerpts below.\n\n"
        f"PDF Context:\n{state['context']}\n\n"
        f"Rules:\n"
        f"1. Base your answer only on the text above.\n"
        f"2. If the context does not contain the answer, say: 'I cannot find information regarding this request in the Leave Policy document.'\n"
        f"3. Do not make up facts."
    )
    msg = llm.invoke([SystemMessage(content=prompt_envelope), HumanMessage(content=state["query"])])
    return {"response": msg.content}

def direct_chat_respond(state: AgentState):
    """Replies directly to standard casual chit-chat queries."""
    print("🤖 [Node: Chat] Processing regular text exchange...")
    msg = llm.invoke([
        SystemMessage(content="You are a professional assistant. Provide a brief, polite response."),
        HumanMessage(content=state["query"])
    ])
    return {"response": msg.content}

# ==========================================
# 4. COMPILING THE LANGGRAPH STATE MACHINE
# ==========================================
builder = StateGraph(AgentState)

# Register workflow operational steps
builder.add_node("retrieve_pdf_context", retrieve_pdf_context)
builder.add_node("generate_grounded_answer", generate_grounded_answer)
builder.add_node("direct_chat_respond", direct_chat_respond)

# Set up routing edges from start
builder.add_conditional_edges(
    START,
    intent_router,
    {
        "retrieve_pdf_context": "retrieve_pdf_context",
        "direct_chat_respond": "direct_chat_respond"
    }
)

# Complete linear tracking links
builder.add_edge("retrieve_pdf_context", "generate_grounded_answer")
builder.add_edge("generate_grounded_answer", END)
builder.add_edge("direct_chat_respond", END)

# Initialize transactional persistence layer (SQLite thread-based memory)
memory_checkpointer = SqliteSaver.from_conn_string(":memory:")
rag_agent_app = builder.compile(checkpointer=memory_checkpointer)

# ==========================================
# 5. CONVERSATIONAL TESTING HARNESS
# ==========================================
def interact_with_agent(session_id: str, prompt: str):
    print(f"\n👤 User: {prompt}")
    config = {"configurable": {"thread_id": session_id}}
    
    # Run the compiled graph execution flow using streaming states
    for state_update in rag_agent_app.stream({"query": prompt, "context": "", "response": ""}, config=config):
        pass
        
    final_output = rag_agent_app.get_state(config).values.get("response")
    print(f"🤖 Agent Response:\n{final_output}")

# Simulation Runs
session_key = "user_employee_101"

# Test Interaction 1: Triggers direct casual chat channel
interact_with_agent(session_key, "Hi there, hope you're having a good day!")

# Test Interaction 2: Triggers PDF verification route
interact_with_agent(session_key, "How many days of casual leave am I allowed per year according to the policy?")
