import logging
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map file extensions to LangChain Language Enums
EXTENSION_LANGUAGE_MAP: Dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".md": Language.MARKDOWN,
    ".html": Language.HTML,
}


class CodeParser:
    """Parses code files using syntax-aware splitters and fallback text splitters."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_file(self, file_path: Path, repo_root: Path) -> List[Document]:
        """Parses a single file into semantic, metadata-rich Document chunks."""
        ext = file_path.suffix.lower()
        try:
            rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            rel_path = file_path.name

        language = EXTENSION_LANGUAGE_MAP.get(ext)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return []

            if language:
                # Syntax-aware splitting using Language-specific RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                docs = splitter.create_documents(
                    texts=[content],
                    metadatas=[{
                        "source": rel_path,
                        "language": language.value,
                        "repo_name": repo_root.name
                    }]
                )
            else:
                # Fallback plain character text splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                docs = splitter.create_documents(
                    texts=[content],
                    metadatas=[{
                        "source": rel_path,
                        "language": "text",
                        "repo_name": repo_root.name
                    }]
                )

            # Calculate start_line and end_line for each chunk for UI citations
            for doc in docs:
                chunk_text = doc.page_content
                # Simple line calculation relative to main file content
                start_line = content[:content.find(chunk_text[:50])].count("\n") + 1 if chunk_text[:50] in content else 1
                lines_in_chunk = chunk_text.count("\n")
                doc.metadata["start_line"] = start_line
                doc.metadata["end_line"] = start_line + lines_in_chunk

            return docs

        except Exception as e:
            logger.warning(f"Error parsing file {file_path}: {e}. Retrying basic text chunking.")
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                doc = Document(
                    page_content=content,
                    metadata={"source": rel_path, "language": "text", "repo_name": repo_root.name, "start_line": 1, "end_line": content.count("\n") + 1}
                )
                splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                return splitter.split_documents([doc])
            except Exception as sub_err:
                logger.error(f"Failed to read file {file_path}: {sub_err}")
                return []
