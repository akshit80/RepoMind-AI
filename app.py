import os
import time
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# Import Backend Pipeline Modules
from repo_cloner import RepoCloner, RepoCloneConfig
from code_parser import CodeParser
from vector_store import VectorStoreManager
from rag_chain import RepoMindRAGChain, QARequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. PAGE CONFIGURATION & DARK THEME STYLING
# ==========================================
st.set_page_config(
    page_title="RepoMind AI - GitHub Repository Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Developer Dark Glassmorphism CSS Injector
st.markdown("""
<style>
    /* Dark Theme Colors */
    .stApp {
        background-color: #0E1117;
    }
    .css-1d3 Wheeler, .stSidebar {
        background-color: #161B22 !important;
    }
    
    /* Custom Card Styling */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }
    
    /* Citation Header Badge */
    .citation-tag {
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 600;
        color: #38BDF8;
        background: #0F172A;
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid #1E293B;
    }
    
    .line-tag {
        font-family: monospace;
        font-size: 0.85rem;
        color: #A7F3D0;
        background: #064E3B;
        padding: 2px 6px;
        border-radius: 4px;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_indexed" not in st.session_state:
        st.session_state.is_indexed = False
    if "indexed_repo_id" not in st.session_state:
        st.session_state.indexed_repo_id = ""
    if "current_repo_url" not in st.session_state:
        st.session_state.current_repo_url = ""
    if "current_branch" not in st.session_state:
        st.session_state.current_branch = "main"
    if "indexing_stats" not in st.session_state:
        st.session_state.indexing_stats = {}
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = ""

init_session_state()

# ==========================================
# 3. BACKEND INDEXING ORCHESTRATION
# ==========================================
@st.cache_resource
def get_vector_store_manager():
    """Caches VectorStoreManager instance across sessions."""
    return VectorStoreManager(persist_directory="data/chroma_db")

vector_mgr = get_vector_store_manager()

def generate_repo_id(url: str, branch: str) -> str:
    clean_url = url.strip().rstrip("/").lower()
    clean_branch = branch.strip().lower() if branch else "main"
    return f"{clean_url}@{clean_branch}"

def run_indexing_pipeline(repo_url: str, branch: str, file_exts: List[str], token: str):
    """Executes backend cloning, AST parsing, chunking, and Chroma indexing."""
    start_time = time.time()
    repo_id = generate_repo_id(repo_url, branch)
    collection_name = f"repo_{hash(repo_id) & 0xffffffff}"

    with st.status("⚡ Indexing Repository...", expanded=True) as status:
        # Step 1: Clone Repository
        st.write("📥 **Step 1/4:** Shallow cloning Git repository...")
        cloner = RepoCloner()
        clone_config = RepoCloneConfig(
            repo_url=repo_url,
            branch=branch if branch else None,
            depth=1,
            token=token if token else None
        )
        repo_path = cloner.clone_repository(clone_config)
        
        try:
            # Step 2: Scan Files
            st.write(f"📂 **Step 2/4:** Scanning valid code files matching extensions: `{file_exts}`...")
            valid_files = cloner.get_valid_code_files(repo_path, allowed_extensions=file_exts)
            if not valid_files:
                raise ValueError("No matching code files found in repository for specified extensions.")

            # Step 3: Parse AST & Chunking
            st.write(f"🧩 **Step 3/4:** Parsing AST structures & chunking {len(valid_files)} code files...")
            parser = CodeParser(chunk_size=1000, chunk_overlap=150)
            all_chunks = []
            for file_path in valid_files:
                chunks = parser.parse_file(file_path, repo_path)
                all_chunks.extend(chunks)

            if not all_chunks:
                raise ValueError("No text chunks generated from scanned repository files.")

            # Step 4: Compute Embeddings & Add to Vector Store
            st.write(f"🧠 **Step 4/4:** Computing HuggingFace embeddings for {len(all_chunks)} chunks into Chroma DB...")
            total_added = vector_mgr.add_documents(collection_name, all_chunks)

            status.update(label="✅ Indexing Complete!", state="complete", expanded=False)

            duration = round(time.time() - start_time, 2)
            st.session_state.is_indexed = True
            st.session_state.indexed_repo_id = repo_id
            st.session_state.current_repo_url = repo_url
            st.session_state.current_branch = branch if branch else "main"
            st.session_state.collection_name = collection_name
            st.session_state.indexing_stats = {
                "chunk_count": len(all_chunks),
                "vector_count": total_added,
                "duration_sec": duration,
                "files_parsed": len(valid_files),
                "indexed_at": time.strftime("%H:%M:%S")
            }

        finally:
            # Clean up cloned temporary repository directory
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)

# ==========================================
# 4. SIDEBAR PANEL & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🧠 RepoMind AI</div>", unsafe_allow_html=True)
    st.caption("GitHub Repository Intelligence & RAG Chat System")
    st.divider()

    st.subheader("📦 Repository Setup")
    repo_url_input = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repository",
        help="Enter public repository HTTPS URL"
    )

    branch_input = st.text_input(
        "Branch Name",
        value="main",
        help="Target branch to clone (default: main)"
    )

    file_types = st.multiselect(
        "Indexed Extensions",
        options=[".py", ".ts", ".js", ".go", ".rs", ".java", ".cpp", ".c", ".h", ".md", ".yaml", ".json"],
        default=[".py", ".ts", ".js", ".md"],
        help="Select code file extensions to include in vector database"
    )

    github_token = st.text_input(
        "GitHub Token (Optional)",
        type="password",
        help="Optional Personal Access Token for private repositories"
    )

    # Primary Action Button
    index_btn = st.button("⚡ Index Repository", type="primary", use_container_width=True)

    if index_btn:
        if not repo_url_input.strip():
            st.error("Please provide a valid GitHub Repository URL.")
        else:
            # Auto-flush chat history if repo target changes
            new_repo_id = generate_repo_id(repo_url_input, branch_input)
            if st.session_state.indexed_repo_id and new_repo_id != st.session_state.indexed_repo_id:
                st.session_state.messages = []
                st.toast("Repository target updated! Chat history reset.", icon="🧹")

            try:
                run_indexing_pipeline(repo_url_input, branch_input, file_types, github_token)
                st.success("Repository indexed & ready for exploration!")
            except Exception as e:
                st.error(f"Indexing Failed: {str(e)}")

    st.divider()

    # Active Metrics Dashboard
    if st.session_state.is_indexed:
        st.subheader("📊 Active Index Metrics")
        stats = st.session_state.indexing_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats.get("chunk_count", 0))
            st.metric("Files Parsed", stats.get("files_parsed", 0))
        with col2:
            st.metric("Vectors", stats.get("vector_count", 0))
            st.metric("Time Taken", f"{stats.get('duration_sec', 0)}s")
            
        st.caption(f"Indexed at: {stats.get('indexed_at', 'N/A')}")
        st.divider()

    # Reset Chat History Control
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared.", icon="🧹")
        st.rerun()

# ==========================================
# 5. MAIN CHAT & EXPLORATION SCREEN
# ==========================================

# Active Context Banner
if st.session_state.is_indexed:
    st.info(
        f"📌 **Active Repository:** `{st.session_state.current_repo_url}` | "
        f"Branch: `{st.session_state.current_branch}` | "
        f"Status: **Indexed & Ready** 🟢"
    )
else:
    st.warning("👈 Enter a GitHub Repository URL in the sidebar and click **⚡ Index Repository** to start.")

# Quick Exploration Starter Prompts
quick_exploration_query = None
if st.session_state.is_indexed and not st.session_state.messages:
    st.markdown("### 💡 Quick Exploration Prompts")
    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        if st.button("🏗️ Explain High-Level Architecture", use_container_width=True):
            quick_exploration_query = "Explain the high-level architecture and component structure of this codebase."

    with p_col2:
        if st.button("🔑 Key Functions & Classes", use_container_width=True):
            quick_exploration_query = "What are the core classes and primary helper functions in this repository?"

    with p_col3:
        if st.button("⚡ Where are entrypoints defined?", use_container_width=True):
            quick_exploration_query = "Where are the main entrypoints, configuration files, or routes located?"

st.divider()

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render Source Code Citations Accordion
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📚 Cited Code Snippets ({len(msg['citations'])} sources)", expanded=False):
                for idx, cit in enumerate(msg["citations"], 1):
                    st.markdown(
                        f"<span class='citation-tag'>[{idx}] {cit['file_path']}</span> "
                        f"<span class='line-tag'>#L{cit['start_line']}-L{cit['end_line']}</span>",
                        unsafe_allow_html=True
                    )
                    st.code(cit["snippet"], language=cit.get("language", "python"))

# Determine user input (manual chat input or quick exploration click)
chat_input_val = st.chat_input("Ask a question about the repository codebase...", disabled=not st.session_state.is_indexed)

user_query = None
if quick_exploration_query:
    user_query = quick_exploration_query
elif chat_input_val:
    user_query = chat_input_val

# Process User Chat Input
if user_query:
    # Append & Display User Query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching codebase vector index & synthesizing response..."):
            try:
                rag_chain = RepoMindRAGChain(vector_mgr)
                request = QARequest(
                    query=user_query,
                    collection_name=st.session_state.collection_name,
                    top_k=5,
                    use_mmr=True
                )
                stream_gen, citations = rag_chain.stream_answer(request)
                
                # Stream the response
                placeholder = st.empty()
                full_resp = ""
                for chunk in stream_gen:
                    full_resp += chunk
                    placeholder.markdown(full_resp)

                if citations:
                    with st.expander(f"📚 Cited Code Snippets ({len(citations)} sources)", expanded=False):
                        for idx, cit in enumerate(citations, 1):
                            st.markdown(
                                f"<span class='citation-tag'>[{idx}] {cit['file_path']}</span> "
                                f"<span class='line-tag'>#L{cit['start_line']}-L{cit['end_line']}</span>",
                                unsafe_allow_html=True
                            )
                            st.code(cit["snippet"], language=cit.get("language", "python"))

                # Persist to Session History
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_resp,
                    "citations": citations
                })
                
                # Force rerun to clear quick exploration query from script rerun cycle
                st.rerun()

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
