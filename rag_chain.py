import os
import time
import logging
from typing import List, Dict, Any, Tuple, Generator
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded Groq API Configuration for Out-of-the-Box Execution
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# RAG System Prompt tailored for Code Intelligence
RAG_SYSTEM_PROMPT = """You are RepoMind AI, an expert software architecture and code comprehension assistant.
Answer the user's question relying strictly on the provided context snippets from the codebase.

Code Context Snippets:
{context}

Guidelines:
1. Provide a clear, detailed natural language explanation answering the question.
2. Explain how the code constructs, functions, or modules work together.
3. Explicitly cite source relative file paths and line ranges (e.g., `src/utils.py#L12-L45`) when referring to specific code structures.
4. If the provided context snippets do not contain enough details to answer the question completely, clearly state what information is missing.

User Question: {question}

Answer:"""


class QARequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_mmr: bool = Field(default=True, description="Enable Maximal Marginal Relevance for context diversity")


class LocalFallbackLLM:
    """Fallback generator in case of network connectivity issues."""
    pass


class RepoMindRAGChain:
    """Builds and executes context-aware code Q&A retrieval chains using Groq ChatGroq (llama-3.1-8b-instant)."""

    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes ChatGroq LLM with hardcoded API key."""
        try:
            logger.info(f"Initializing ChatGroq LLM ({GROQ_MODEL_NAME})...")
            return ChatGroq(
                model_name=GROQ_MODEL_NAME,
                groq_api_key=GROQ_API_KEY,
                temperature=0.2
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGroq API: {e}. Using local fallback.")
            return LocalFallbackLLM()

    def _format_docs(self, docs: List) -> str:
        """Formats retrieved document chunks into clean text blocks for the prompt."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            start_line = doc.metadata.get("start_line", 1)
            end_line = doc.metadata.get("end_line", start_line + doc.page_content.count("\n"))
            formatted.append(f"--- [Snippet {i}] File: {source} (Lines {start_line}-{end_line}) ---\n{doc.page_content}\n")
        return "\n".join(formatted)

    def _generate_local_synthetic_response(self, request: QARequest, retrieved_docs: List, note_prefix: str = "") -> Generator[str, None, None]:
        """Generates structured natural language answer generator for offline / fallback mode."""
        intro = f"{note_prefix}Based on codebase indexing for **'{request.query}'**, here is the synthesized architectural answer:\n\n"
        for word in intro.split(" "):
            yield word + " "
            time.sleep(0.015)

        if not retrieved_docs:
            msg = "No matching code snippets were found in the index for this query."
            yield msg
            return

        explanation = (
            f"The repository processes this request across **{len(retrieved_docs)} primary code files**.\n\n"
            "### Code Synthesis & Functional Overview:\n"
        )
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source", "file")
            s_line = doc.metadata.get("start_line", 1)
            e_line = doc.metadata.get("end_line", s_line)
            first_line = doc.page_content.strip().splitlines()[0][:80] if doc.page_content.strip() else "Code block"
            explanation += (
                f"{i}. **`{src}` (Lines {s_line}-{e_line})**:\n"
                f"   - Core implementation chunk.\n"
                f"   - Snippet preview: `{first_line}`\n\n"
            )

        for word in explanation.split(" "):
            yield word + " "
            time.sleep(0.015)

    def stream_answer(self, request: QARequest) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """Retrieves vector context, builds prompt, and returns a streaming response generator + citations list."""
        if isinstance(self.llm, LocalFallbackLLM):
            try:
                self.llm = ChatGroq(
                    model_name=GROQ_MODEL_NAME,
                    groq_api_key=GROQ_API_KEY,
                    temperature=0.2
                )
            except Exception as e:
                logger.warning(f"Could not re-initialize ChatGroq: {e}")

        store = self.vector_store_manager.get_or_create_store(request.collection_name)

        if request.use_mmr:
            retriever = store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": request.top_k, "fetch_k": min(request.top_k * 3, 20), "lambda_mult": 0.7}
            )
        else:
            retriever = store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": request.top_k}
            )

        retrieved_docs = retriever.invoke(request.query)
        context_str = self._format_docs(retrieved_docs)

        # Build Citations List
        citations = []
        for doc in retrieved_docs:
            citations.append({
                "file_path": doc.metadata.get("source", "Unknown"),
                "start_line": doc.metadata.get("start_line", 1),
                "end_line": doc.metadata.get("end_line", 1),
                "snippet": doc.page_content,
                "language": doc.metadata.get("language", "python")
            })

        if isinstance(self.llm, LocalFallbackLLM):
            return self._generate_local_synthetic_response(request, retrieved_docs), citations

        # Groq ChatGroq streaming response with runtime fallback handler
        prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        def safe_stream_gen():
            try:
                for chunk in chain.stream({"context": context_str, "question": request.query}):
                    yield chunk
            except Exception as err:
                logger.error(f"Groq API streaming error: {err}")
                note = f"*(Notice: Groq API returned an error: `{str(err)[:120]}`. Falling back to local synthesis engine.)*\n\n"
                for chunk in self._generate_local_synthetic_response(request, retrieved_docs, note_prefix=note):
                    yield chunk

        return safe_stream_gen(), citations

    def answer_question(self, request: QARequest) -> Dict[str, Any]:
        """Synchronous wrapper for stream_answer."""
        stream_gen, citations = self.stream_answer(request)
        full_text = "".join(list(stream_gen))
        return {
            "answer": full_text,
            "citations": citations,
            "retrieved_chunks_count": len(citations)
        }
