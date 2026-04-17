import streamlit as st
import os
import tempfile

from pipeline import (
    load_document_chunk,
    build_vector_store,
    get_vector_store,
    reset_vector_store,
    search_documents,
    get_answer,
    detect_language,
)

# ====================== CONFIG ======================
st.set_page_config(
    page_title="DocuMind",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM STYLING ======================
st.markdown("""
<style>
    /* Overall */
    .main {
        background-color: #ffffff;
        padding-top: 2rem;
    }
    
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }
    
    h1 {
        font-size: 2.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* Sidebar */
    .sidebar-content {
        background-color: #f8fafc;
    }
    
    /* Chat */
    .stChatMessage {
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    
    .stChatMessage.user {
        background-color: #f1f5f9;
    }
    
    .stChatMessage.assistant {
        background-color: #f8fafc;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        height: 46px;
        font-weight: 600;
    }
    
    /* Document cards */
    .doc-item {
        background: white;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 10px;
        transition: all 0.2s;
    }
    .doc-item:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown('<h1 style="text-align: center;">📘 DocuMind</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="text-align: center;">Ask intelligent questions about your documents • Powered by LLaMA 3.1</p>', 
            unsafe_allow_html=True)

st.divider()

# ====================== SESSION STATE ======================
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### 📂 Upload Documents")
    
    uploaded_files = st.file_uploader(
        label="",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns(2, gap="small")
    
    with col1:
        if st.button("Upload & Index", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Indexing documents..."):
                    try:
                        temp_paths = []
                        original_names = []

                        for file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                                tmp.write(file.getvalue())
                                temp_paths.append(tmp.name)
                                original_names.append(file.name)

                        chunks = load_document_chunk(temp_paths)

                        if chunks:
                            # Fix filenames
                            for chunk in chunks:
                                temp_src = chunk.metadata.get("source", "")
                                if temp_src in [os.path.basename(p) for p in temp_paths]:
                                    idx = [os.path.basename(p) for p in temp_paths].index(temp_src)
                                    chunk.metadata["source"] = original_names[idx]

                            if get_vector_store() is None:
                                build_vector_store(chunks)
                            else:
                                get_vector_store().add_documents(chunks)

                            for name in original_names:
                                if name not in st.session_state.indexed_files:
                                    st.session_state.indexed_files.append(name)

                            st.success(f"✅ {len(original_names)} file(s) indexed")
                        else:
                            st.error("Could not extract text from files.")

                        # Cleanup temp files
                        for p in temp_paths:
                            if os.path.exists(p):
                                os.unlink(p)

                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please upload files first.")

    with col2:
        if st.button("Reset All", use_container_width=True):
            if st.checkbox("Confirm reset?", key="confirm_reset"):
                reset_vector_store()
                st.session_state.indexed_files = []
                st.session_state.chat_history = []
                st.success("All documents cleared.")
                st.rerun()

    st.markdown("### 📑 Indexed Documents")
    if st.session_state.indexed_files:
        for file in st.session_state.indexed_files:
            st.markdown(f"""
                <div class="doc-item">
                    📄 {file}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No documents uploaded yet.")

    st.divider()
    st.caption("Built with LLaMA 3.1 • ChromaDB • Streamlit")

# ====================== CHAT INTERFACE ======================
st.subheader("💬 Ask any question")

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").markdown(message)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.indexed_files:
        st.error("Please upload and index at least one document first.")
    else:
        # Show user message
        st.session_state.chat_history.append(("user", prompt))
        st.chat_message("user").write(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking..."):
                try:
                    lang = detect_language(prompt)

                    context_texts, metadatas = search_documents(
                        query=prompt, 
                        k=6, 
                        fetch_k=18
                    )

                    if not context_texts:
                        answer = "I could not find the answer in the provided documents."
                    else:
                        answer = get_answer(
                            question=prompt,
                            context_texts=context_texts,
                            metadatas=metadatas,
                            language=lang
                        )

                    st.markdown(answer)

                    # Sources
                    if metadatas:
                        with st.expander("📌 Sources", expanded=False):
                            seen = set()
                            for m in metadatas:
                                src = m.get("source", "Unknown")
                                if src not in seen:
                                    seen.add(src)
                                    page = m.get("page", "N/A")
                                    st.write(f"**{src}** (Page: {page})")

                    st.session_state.chat_history.append(("assistant", answer))

                except Exception as e:
                    st.error(f"Failed to generate answer: {str(e)}")

# Footer
st.divider()
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:0.9rem;'>"
    "DocuMind — Private • Accurate • Multilingual"
    "</p>", 
    unsafe_allow_html=True
)