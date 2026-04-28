# RAG with LangChain and Gemma (Local LLM)

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain that answers questions using content retrieved from a web page, powered entirely by local models via Ollama.

## What it does

1. **Loads** the Wikipedia page on Large Language Models using `WebBaseLoader`
2. **Splits** the content into overlapping chunks (1000 chars, 200 overlap) with `RecursiveCharacterTextSplitter`
3. **Embeds** the chunks using `nomic-embed-text` via Ollama and stores them in a **FAISS** vector store
4. **Retrieves** relevant chunks via similarity search for a given query
5. **Generates** an answer using the `gemma4:e4b` model (local, via Ollama) with a strict "answer only from context" prompt

## Two RAG chain approaches

### Method 1 — Manual LCEL chain
Builds the chain explicitly using LangChain Expression Language (LCEL):
```
retriever | format_docs → prompt → llm → StrOutputParser
```

### Method 2 — LangChain Classic chains
Uses `create_stuff_documents_chain` + `create_retrieval_chain` from `langchain_classic` for a more structured, higher-level API. Returns a dict with `answer` and source documents.

## Requirements

- [Ollama](https://ollama.com/) running locally with:
  - `ollama pull nomic-embed-text`
  - `ollama pull gemma4:e4b`
- Python dependencies:
  ```
  langchain
  langchain-community
  langchain-ollama
  langchain-classic
  langchain-text-splitters
  faiss-cpu  # or faiss-gpu
  ```

## Usage

Open and run `main.ipynb` cell by cell. Requires an active internet connection for the initial Wikipedia page load.
