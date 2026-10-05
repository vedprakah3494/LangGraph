import streamlit as st
from chatbot_streamlit import graph

st.title("My First LangGraph Chatbot with State Memory and Streamlit UI")

# Sidebar for managing individual threads
thread_id = st.sidebar.text_input("Thread ID", value="student-1")
settings = {"configurable": {"thread_id": thread_id}}

# Handle session termination state
if "stopped" not in st.session_state:
    st.session_state.stopped = False

# Capture user input (Disabled if the session is stopped)
user_message = st.chat_input(
    "Type a message ('bye' or 'quit' to stop)", 
    disabled=st.session_state.stopped
)

if user_message:
    if user_message.strip().lower() in ["bye", "quit"]:
        st.session_state.stopped = True
        # Instantly re-run so the input box disables right away
        st.rerun() 
    else:
        # 1. Update LangGraph state via database
        graph.invoke({"user_message": user_message}, config=settings)
        # 2. Force Streamlit to re-run and immediately fetch the freshly saved state
        st.rerun()

# Fetch the chronological history directly from the LangGraph SQLite backend
saved_state = graph.get_state(settings)
chat_history = saved_state.values.get("chat_history", [])

# Render past chat logs cleanly
for message in chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Final structural check for termination message
if st.session_state.stopped:
    with st.chat_message("assistant"):
        st.write("Goodbye!")
