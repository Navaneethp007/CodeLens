# CodeLens

An AI-powered code analysis tool that provides intelligent code search and natural language querying for Python codebases using semantic embeddings and LLM-based analysis.

## Features

- **Single File & Codebase Analysis**: Upload individual files or entire folders for cross-file queries
- **Direct Analysis**: Small files (<3KB) analyzed directly by LLM for maximum accuracy
- **Smart Chunking**: Large files automatically chunked into semantic units (functions, classes, assignments)
- **Query Classification**: Automatically routes comprehensive vs. specific queries to appropriate retrieval strategies
- **Metadata Filtering**: Direct filtering for comprehensive queries ("what are all the agents?")
- **Hybrid Search**: Combines keyword and semantic search for specific queries
- **Cross-Encoder Reranking**: Scores and filters chunks by true relevance (0-1 scale)
- **Vector Storage**: ChromaDB with persistent storage for code embeddings
- **AI-Powered Explanations**: CodeLlama generates contextual answers to natural language questions

## Architecture

### Components

- **Direct Analyzer**: Sends entire file to LLM for files <3KB
- **Chunked Analyzer**: For larger files, uses intelligent retrieval pipeline:
  1. AST Parser → extracts code structures
  2. Chunk Builder → creates semantic chunks with metadata
  3. ChromaDB Storage → stores embeddings persistently
  4. Query Classifier → determines query type
  5. Metadata Filter → filters by type for comprehensive queries
  6. Hybrid Search → keyword + semantic search for specific queries
  7. Cross-Encoder Reranker → scores relevance and filters chunks
  8. LLM Generation → synthesizes answer from reranked chunks

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
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   └── storage/          # Data persistence
│       ├── session_store.py
│       └── chroma_manager.py
├── chroma_db/            # Vector database storage
└── manage.py
```

