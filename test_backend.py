import os
import shutil
import logging
from pathlib import Path

from repo_cloner import RepoCloner, RepoCloneConfig
from code_parser import CodeParser
from vector_store import VectorStoreManager
from rag_chain import RepoMindRAGChain, QARequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    logger.info("--- Starting RepoMind AI Pipeline Verification ---")
    
    # 1. URL Validation Test
    repo_url = "https://github.com/psf/requests"
    logger.info(f"Validating URL: {repo_url} -> {RepoCloner.validate_url(repo_url)}")
    assert RepoCloner.validate_url(repo_url), "URL validation failed"

    # 2. Shallow Clone Test
    cloner = RepoCloner()
    config = RepoCloneConfig(repo_url=repo_url, depth=1)
    repo_path = cloner.clone_repository(config)
    logger.info(f"Repository cloned to: {repo_path}")
    assert repo_path.exists(), "Clone directory does not exist"

    try:
        # 3. File Scanning Test
        valid_files = cloner.get_valid_code_files(repo_path, allowed_extensions=[".py", ".md"])
        logger.info(f"Found {len(valid_files)} matching code files.")
        assert len(valid_files) > 0, "No valid files found"

        # 4. Code Parser & AST Chunking Test
        parser = CodeParser(chunk_size=800, chunk_overlap=100)
        chunks = []
        for f in valid_files[:10]: # Test first 10 files
            c = parser.parse_file(f, repo_path)
            chunks.extend(c)
        logger.info(f"Generated {len(chunks)} Document chunks from sample files.")
        assert len(chunks) > 0, "No chunks generated"

        # 5. HuggingFace Embeddings & Chroma DB Ingestion Test
        vector_mgr = VectorStoreManager(persist_directory="data/test_chroma_db")
        collection_name = "test_requests_repo"
        added_count = vector_mgr.add_documents(collection_name, chunks)
        logger.info(f"Added {added_count} chunks to Chroma test collection.")
        assert added_count == len(chunks), "Chunk count mismatch in vector store"

        # 6. RAG Retrieval & QA Verification Test
        rag_chain = RepoMindRAGChain(vector_mgr)
        request = QARequest(
            query="How is session management or HTTPS handling structured?",
            collection_name=collection_name,
            top_k=3,
            use_mmr=True
        )
        response = rag_chain.answer_question(request)
        logger.info("--- RAG Answer Output ---")
        logger.info(response["answer"][:300] + "...")
        logger.info(f"Retrieved Chunks: {response['retrieved_chunks_count']}")
        logger.info(f"Citations Count: {len(response['citations'])}")
        assert response["retrieved_chunks_count"] > 0, "No chunks retrieved in RAG query"

        logger.info("✅ All pipeline components verified successfully!")

    finally:
        # Clean up temporary test repo directory & test vector store
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)
        test_db = Path("data/test_chroma_db")
        if test_db.exists():
            shutil.rmtree(test_db, ignore_errors=True)

if __name__ == "__main__":
    test_pipeline()
