import streamlit as st
import os
from vectorstore import process_and_store, process_all_pdfs_in_folder, get_vectorstore_stats
from rag_chain import chat_with_document, retrieve_only
from config import Config

# Set up the app
st.set_page_config(page_title="Document RAG System", page_icon="🤖", layout="wide")
st.title("🤖 Document RAG System")
st.markdown("*Hybrid search with AI-powered answers and grounded citations*")

# Create data folder if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.last_chunks = []  # Store chunks from last retrieval
    
    # Process all existing PDFs in data folder on startup
    with st.spinner("Processing existing PDFs in data/ folder..."):
        results = process_all_pdfs_in_folder(Config.UPLOAD_FOLDER)
        if "error" not in results and results.get("total_files", 0) > 0:
            st.session_state.startup_results = results

# Show sidebar with vector store statistics and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode selector
    mode = st.radio(
        "Response Mode:",
        ["🤖 AI Generation (with citations)", "📄 Retrieval Only"],
        index=0,
        help="AI Generation uses Google Gemini to answer questions. Retrieval Only shows raw chunks."
    )
    
    st.divider()
    
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
    
    # Show retrieved chunks from last query
    if st.session_state.get("last_chunks"):
        st.divider()
        st.header("🔍 Retrieved Chunks")
        st.caption(f"Last query: {st.session_state.get('last_query', 'N/A')}")
        
        # Separate cited and non-cited chunks
        cited_chunks = [c for c in st.session_state.last_chunks if c.get('cited', False)]
        not_cited_chunks = [c for c in st.session_state.last_chunks if not c.get('cited', False)]
        
        # Show cited chunks first (if in AI generation mode)
        if cited_chunks:
            st.success(f"✅ {len(cited_chunks)} chunk(s) cited in answer")
            for chunk in cited_chunks:
                emoji = "✅"
                merged_info = f" (merged {chunk['merged_count']} chunks)" if chunk.get('merged_count', 1) > 1 else ""
                with st.expander(f"{emoji} Chunk {chunk['chunk_number']} • {chunk['score']:.0%}{merged_info}", expanded=False):
                    st.markdown(f"**Source:** {chunk['source']}")
                    st.markdown(f"**Page:** {chunk['page']}")
                    if chunk.get('chunk_title'):
                        st.markdown(f"**Section:** {chunk['chunk_title']}")
                    st.markdown(f"**Relevance:** {chunk['score']:.2%}")
                    if chunk.get('merged_count', 1) > 1:
                        st.info(f"🔀 Merged {chunk['merged_count']} chunks from same page")
                    st.success("✅ **Cited in answer**")
                    st.divider()
                    st.markdown(f"*{chunk['preview']}*")
                    
                    # Add link if available
                    if chunk.get('source_url') and chunk['page'] != 'N/A':
                        page_link = f"{chunk['source_url']}#page={chunk['page']}"
                        st.markdown(f"[📖 View in PDF]({page_link})")
        
        # Show not cited chunks
        if not_cited_chunks:
            with st.expander(f"📋 {len(not_cited_chunks)} additional chunk(s) retrieved (not cited)", expanded=False):
                for chunk in not_cited_chunks:
                    merged_info = f" (merged {chunk['merged_count']} chunks)" if chunk.get('merged_count', 1) > 1 else ""
                    st.markdown(f"**Chunk {chunk['chunk_number']}** • {chunk['score']:.0%}{merged_info}")
                    st.markdown(f"📄 {chunk['source']}, Page {chunk['page']}")
                    st.caption(chunk['preview'])
                    st.divider()

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

# Initialize chat history and chunks if not set
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Ask a question about your documents..."):
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Get response based on selected mode
    with st.spinner("🔍 Searching and generating answer..." if "AI Generation" in mode else "🔍 Retrieving relevant chunks..."):
        if "AI Generation" in mode:
            response, chunks_metadata = chat_with_document(query)
        else:
            response, chunks_metadata = retrieve_only(query)
    
    # Store chunks metadata and query for sidebar display
    st.session_state.last_chunks = chunks_metadata
    st.session_state.last_query = query

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    # Save response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update sidebar
    st.rerun()
