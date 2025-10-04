# CodeLens

An AI-powered code analysis tool that provides intelligent code search and natural language querying for Python codebases using semantic embeddings and LLM-based analysis.

## Features

- **Direct Analysis**: Small files (<30KB) analyzed directly by LLM for maximum accuracy
- **Smart Chunking**: Large files automatically chunked into semantic units (functions, classes, assignments)
- **Query Classification**: Automatically routes comprehensive vs. specific queries to appropriate retrieval strategies
- **Metadata Filtering**: Direct filtering for comprehensive queries ("what are all the agents?")
- **Hybrid Search**: Combines keyword and semantic search for specific queries
- **Vector Storage**: ChromaDB with persistent storage for code embeddings
- **AI-Powered Explanations**: CodeLlama generates contextual answers to natural language questions

## Architecture

### Components

- **Direct Analyzer**: Sends entire file to LLM for files <30KB
- **Chunked Analyzer**: For larger files, uses intelligent retrieval pipeline:
  1. AST Parser → extracts code structures
  2. Chunk Builder → creates semantic chunks with metadata
  3. Query Classifier → determines query type
  4. Metadata Filter → filters by type for comprehensive queries
  5. Hybrid Search → keyword + semantic search for specific queries
  6. LLM Generation → synthesizes answer from retrieved chunks

### Technologies

- **Django + DRF**: REST API framework
- **ChromaDB**: Persistent vector database
- **Sentence Transformers**: all-MiniLM-L6-v2 for embeddings
- **Ollama + CodeLlama**: Local LLM for code analysis
- **Python AST**: Code parsing and structure extraction

## Project Structure

```
CodeLens/
├── CodeLens/              # Project settings
│   ├── settings.py
│   └── urls.py
├── guide/                 # Main application
│   ├── views.py          # API endpoints
│   ├── urls.py           # URL routing
│   ├── analyzers/        # Analysis strategies
│   │   ├── direct_analyzer.py
│   │   └── chunked_analyzer.py
│   ├── chunking/         # Code parsing
│   │   ├── ast_parser.py
│   │   └── chunk_builder.py
│   ├── retrieval/        # Search strategies
│   │   ├── query_classifier.py
│   │   ├── metadata_filter.py
│   │   └── hybrid_search.py
│   └── storage/          # Data persistence
│       ├── session_store.py
│       └── chroma_manager.py
└── manage.py

## API Endpoints

### Upload File
POST /api/upload/
- **Body**: `multipart/form-data` with `file` field
- **Returns**: Upload status, file size, analysis method (direct/chunked)

### Query Code
POST /api/query/
- **Body**: `{"query": "What are the different agents in this file?"}`
- **Returns**: 
```json
  {
    "status": "success",
    "query": "...",
    "answer": "...",
    "method": "direct_analysis|chunked_analysis",
    "chunks_analyzed": 10
  }