# RepoMind AI
### Automated Codebase Intelligence Platform

RepoMind AI is an end-to-end **Retrieval-Augmented Generation (RAG)** platform that enables developers to analyze and query GitHub repositories using natural language. Instead of manually searching through large codebases, users can ask questions in plain English and receive context-aware answers backed by relevant source code references.

---

# Introduction

Large Language Models (LLMs) struggle to process entire software repositories because of:

- Limited context windows
- High token consumption
- Attention degradation over large inputs

RepoMind AI addresses these limitations by implementing a complete RAG pipeline that:

- Clones remote GitHub repositories
- Parses source code into semantic chunks
- Generates vector embeddings
- Retrieves only the most relevant code sections
- Produces accurate answers with source citations

This architecture enables efficient understanding of large codebases while significantly reducing inference cost and improving response quality.

---

# Features

- Clone any public GitHub repository
- Syntax-aware code parsing
- Automatic semantic chunk generation
- Local embedding generation
- High-speed vector similarity search
- Natural language codebase querying
- Source file citations for every response
- Conversation history management
- Secure environment variable support

---

# High-Level Architecture

```
GitHub Repository
        │
        ▼
Repository Cloning (GitPython)
        │
        ▼
Language-Aware Code Parsing
        │
        ▼
Semantic Chunk Generation
        │
        ▼
Embedding Model
(all-MiniLM-L6-v2)
        │
        ▼
Chroma Vector Database
        │
        ▼
Similarity Search
        │
        ▼
Groq + Llama 3.1
        │
        ▼
Natural Language Answer
with Source Citations
```

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| RAG Framework | LangChain (LCEL) |
| Repository Ingestion | GitPython |
| Vector Database | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 (HuggingFace) |
| Language Model | Meta Llama-3.1-8B-Instant |
| Inference Provider | Groq API |
| Environment Management | python-dotenv |
| Version Control | Git & GitHub |

---

# System Workflow

## Phase 1 — Repository Ingestion

The application accepts a public GitHub repository URL and performs a shallow clone (`depth=1`) using **GitPython** to minimize storage and improve cloning speed.

During ingestion, unnecessary files and directories are excluded, including:

- `.git`
- `__pycache__`
- Images
- Binary files
- Lock files
- Generated artifacts

This ensures that only meaningful source code is indexed.

---

## Phase 2 — Syntax-Aware Code Parsing

Unlike traditional text splitters that divide code purely by character count, RepoMind AI preserves programming language syntax using LangChain's **LanguageParser**.

The parser respects:

- Functions
- Classes
- Indentation
- Language-specific delimiters

Each repository is divided into:

- **Chunk Size:** 1000 tokens
- **Overlap:** 150 tokens

Every chunk retains metadata including:

- File path
- Relative directory
- Source information

This preserves contextual relationships between code segments.

---

## Phase 3 — Embedding Generation

Each semantic chunk is transformed into a **384-dimensional vector embedding** using the **all-MiniLM-L6-v2** transformer model.

Embedding generation is performed locally on the host machine.

The generated vectors are stored inside **ChromaDB**, creating a searchable vector index for semantic retrieval.

---

## Phase 4 — Retrieval-Augmented Generation (RAG)

When a user submits a query:

1. The question is converted into a vector embedding.
2. ChromaDB performs similarity search.
3. The top-k relevant code chunks are retrieved.
4. Retrieved context is injected into the LLM prompt.
5. The prompt is sent to the Groq inference engine.
6. Meta Llama-3.1 generates a context-aware response.
7. Source citations are returned alongside the generated answer.

---

# Core Engineering Highlights

## Deterministic RAG Pipeline

Built a complete Extract → Transform → Load (ETL) pipeline for software repositories before interacting with an external language model.

---

## Syntax-Aware Retrieval

Maintains programming language structure during parsing, significantly improving retrieval quality compared to traditional text splitting.

---

## Efficient Context Management

Reduces token usage by retrieving only the most relevant code snippets instead of processing the entire repository.

This improves:

- Accuracy
- Latency
- Cost efficiency

---

## High-Speed Inference

Utilizes **Groq's inference hardware** with **Meta Llama-3.1-8B-Instant** to deliver low-latency responses suitable for interactive applications.

---

## Persistent Semantic Search

Uses ChromaDB to persist embeddings locally, eliminating repeated embedding generation for previously indexed repositories.

---

## Secure Secret Management

Sensitive credentials are managed through environment variables using:

```python
os.getenv("GROQ_API_KEY")
```

instead of hardcoding API keys, ensuring secure deployment and preventing accidental exposure in version control.

---

# Project Structure

```
RepoMind-AI/
│
├── app.py
├── rag_chain.py
├── repo_cloner.py
├── code_parser.py
├── vector_store.py
├── requirements.txt
├── .gitignore
├── .env (ignored)
├── test_backend.py
└── data/
```

---

# Installation

### Clone the repository

```bash
git clone https://github.com/akshit80/RepoMind-AI.git
```

### Navigate to the project

```bash
cd RepoMind-AI
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
GROQ_API_KEY=your_api_key_here
```

### Run the application

```bash
streamlit run app.py
```

---

# Future Improvements

- Multi-language repository support
- Repository indexing cache
- Incremental repository updates
- Multi-repository querying
- Docker deployment
- User authentication
- Repository summarization
- Code dependency visualization
- Interactive architecture diagrams

---

# Author

**Akshit Dhiman**

- GitHub: https://github.com/akshit80
- LinkedIn: *(Add your LinkedIn profile here)*

---

# License

This project is licensed under the **MIT License**.
