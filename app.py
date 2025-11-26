import streamlit as st
import os
from vectorstore import process_and_store, process_all_pdfs_in_folder, get_vectorstore_stats
from rag_chain import chat_with_document
from config import Config

# Set up the app
st.set_page_config(page_title="Document Retrieval System", page_icon="📄", layout="wide")
st.title("📄 Document Retrieval System")
st.markdown("*Upload PDFs or place them in the `data/` folder - retrieval only (no LLM generation)*")

# Create data folder if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    
    # Process all existing PDFs in data folder on startup
    with st.spinner("Processing existing PDFs in data/ folder..."):
        results = process_all_pdfs_in_folder(Config.UPLOAD_FOLDER)
        if "error" not in results and results.get("total_files", 0) > 0:
            st.session_state.startup_results = results

# Show sidebar with vector store statistics
with st.sidebar:
    st.header("📊 Vector Store Stats")
    stats = get_vectorstore_stats()
    
    if stats["exists"]:
        st.success("✅ Vector store active")
        st.metric("Total Documents", stats["total_files"])
        st.metric("Total Chunks", stats["total_chunks"])
        
        with st.expander("📁 Processed Files"):
            for file in stats["files"]:
                st.text(f"• {file}")
    else:
        st.info("No documents indexed yet")
    
    # Show startup processing results if any
    if "startup_results" in st.session_state:
        st.divider()
        st.subheader("🔄 Startup Processing")
        results = st.session_state.startup_results
        if results["newly_processed"] > 0:
            st.success(f"✅ Processed {results['newly_processed']} new file(s)")
        if results["already_processed"] > 0:
            st.info(f"ℹ️ {results['already_processed']} file(s) already indexed")

# Upload the document
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    file_path = os.path.join(Config.UPLOAD_FOLDER, uploaded_file.name)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the document
    with st.spinner(f"Processing {uploaded_file.name}..."):
        success, message, chunks = process_and_store(file_path)
    
    if success:
        if "already processed" in message:
            st.info(f"ℹ️ {message}")
        else:
            st.success(f"✅ {message} ({chunks} chunks created)")
        # Force refresh of stats
        st.rerun()

# Initialize chat history if not set
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Search the document..."):
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Get retrieved chunks from document
    response = chat_with_document(query)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
