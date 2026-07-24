import os
import hashlib
import logging
from typing import List, Optional
from pathlib import Path

# Try imports for huggingface embeddings across versions
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages HuggingFace embeddings and persistent Chroma database operations."""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = str(Path(persist_directory).resolve())
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing HuggingFaceEmbeddings model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )

    def _generate_chunk_id(self, doc: Document, index: int) -> str:
        """Generates a unique SHA-256 hash for document content, source path, and index."""
        content_key = f"{doc.page_content}::{doc.metadata.get('source', '')}::{doc.metadata.get('start_line', 1)}::{index}"
        return hashlib.sha256(content_key.encode("utf-8")).hexdigest()

    def get_or_create_store(self, collection_name: str) -> Chroma:
        """Loads or creates a persistent Chroma collection."""
        clean_name = collection_name.lower().replace("-", "_").replace(".", "_")
        # Ensure collection name complies with Chroma rules (3-63 chars, alphanumeric/underscores)
        clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")
        if len(clean_name) < 3:
            clean_name = f"collection_{clean_name}"
        clean_name = clean_name[:63]

        return Chroma(
            collection_name=clean_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, collection_name: str, documents: List[Document], batch_size: int = 100) -> int:
        """Inserts documents in batches with guaranteed unique IDs."""
        if not documents:
            return 0

        vector_store = self.get_or_create_store(collection_name)
        
        # Deduplicate and assign unique IDs
        unique_docs = []
        unique_ids = []
        seen_ids = set()

        for idx, doc in enumerate(documents):
            chunk_id = self._generate_chunk_id(doc, idx)
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_docs.append(doc)
                unique_ids.append(chunk_id)

        total_added = 0
        for i in range(0, len(unique_docs), batch_size):
            batch_docs = unique_docs[i : i + batch_size]
            batch_ids = unique_ids[i : i + batch_size]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            total_added += len(batch_docs)

        logger.info(f"Successfully added {total_added} document chunks to collection '{collection_name}'.")
        return total_added

    def delete_collection(self, collection_name: str) -> bool:
        """Deletes a collection from Chroma store."""
        try:
            vector_store = self.get_or_create_store(collection_name)
            vector_store.delete_collection()
            logger.info(f"Collection '{collection_name}' deleted.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
