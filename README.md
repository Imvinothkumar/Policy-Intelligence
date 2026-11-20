# PolicyRAG

Local document search and retrieval system for compliance and policy documents. It chunks PDFs, generates vector embeddings, stores them in FAISS, and answers questions with direct citations.

Everything runs locally on your machine with no external APIs or cloud services.

## Overview

PolicyRAG provides two modes to query internal documents:

- **Strict Retrieval**: Retrieves the most relevant chunks from policy files and displays quoted excerpts with exact page citations. No language model is used, ensuring zero hallucination.
- **RAG Synthesis**: Retrieves relevant chunks and passes them to a local Ollama model to generate a concise summary while maintaining citations.

It also supports multi-company workspaces, allowing different organizations to maintain separate document directories and vector indexes.

## Architecture

Documents -> Text Chunks -> Sentence-Transformers Embeddings -> FAISS Vector Index -> Similarity Search -> Strict Quote / Local Ollama Response -> Streamlit UI

1. Documents: PDFs placed in company folders under `data/companies/<Company Name>/`.
2. Chunking: Recursive text splitting (900 character chunks, 150 character overlap).
3. Embeddings: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors.
4. Storage: Local FAISS vector index with metadata for document name, page number, and chunk ID.
5. Query: Cosine similarity search filtered by a configurable relevance score threshold.
6. Answering: Either direct excerpt quotation or local synthesis via Ollama (`llama3.2:1b` or `llama3.2:latest`).

## Project Structure

```
policyrag/
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
├── data/
│   ├── mock_policies/         # Sample synthetic enterprise policies
│   └── companies/             # Company-specific document folders
├── indexes/                   # FAISS vector indexes (local, git-ignored)
├── src/
│   ├── answering.py           # Strict retrieval and RAG synthesis logic
│   ├── config.py              # Paths, models, and retrieval parameters
│   ├── indexing.py            # PDF loading, chunking, and FAISS indexing
│   ├── mock_docs.py           # Mock policy definitions
│   ├── ollama_client.py       # Ollama integration and token streaming
│   └── utils.py               # Text formatting utilities
└── scripts/
    ├── build_index.py         # Script to index all policies
    ├── create_mock_pdfs.py    # Script to generate sample policy PDFs
    ├── demo_queries.py        # Terminal query runner
    └── smoke_test.py          # Verification script
```

## Setup Instructions

### 1. Requirements

- Python 3.10 or higher
- Ollama (optional, only needed for RAG Synthesis mode)

### 2. Installation

Clone the repository and enter the directory:

```bash
git clone https://github.com/Imvinothkumar/Policy-Intelligence.git
cd Policy-Intelligence
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Pull a Local Model (Optional)

If you plan to use RAG Synthesis mode, install Ollama and pull a model:

```bash
ollama pull llama3.2:1b
```

Strict Retrieval mode works without Ollama.

### 4. Build the Sample Index

Generate the sample policy PDFs and build the FAISS index:

```bash
python scripts/create_mock_pdfs.py
python scripts/build_index.py
```

### 5. Run the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

## Features

- **User and Developer Modes**: A clean interface for asking questions and viewing citations, with a toggle to Developer mode for inspecting chunk distances, similarity scores, and index metrics.
- **Multiple Organizations**: Support for creating isolated document sets per company or department.
- **Direct PDF Upload**: Upload new policy documents from the UI and rebuild the index with one click.
- **Token Streaming**: Real-time answer streaming from Ollama for low perceived latency.
- **Auditability**: Every answer includes the source document, page number, and chunk reference.

## Sample Documents

The default dataset includes 6 synthetic policies:

- `corporate_kyc_onboarding_policy.pdf`
- `client_data_handling_standard.pdf`
- `ai_assistant_usage_policy.pdf`
- `model_risk_review_guide.pdf`
- `third_party_vendor_risk_policy.pdf`
- `incident_escalation_runbook.pdf`

The AI assistant policy contains a test adversarial phrase to verify that the prompt instructions prevent retrieved content from hijacking model behavior.

## Verification

Run the test suite to verify indexing and retrieval:

```bash
python scripts/smoke_test.py
```

## License

MIT
