# CodeLens

A Django-based web application that provides intelligent code search and indexing capabilities using embeddings and AI-powered code analysis.

## Features

- **Code Indexing**: Upload and index Python code files with intelligent chunking of classes, functions, and code blocks
- **Semantic Search**: Search through indexed code using natural language queries
- **AI-Powered Analysis**: Leverages CodeLlama to analyze and explain code relevance to search queries
- **Smart Chunking**: Automatically breaks down code into meaningful chunks while preserving context
- **Embeddings**: Uses Sentence Transformers to generate semantic embeddings for accurate code search
- **REST API**: Built with Django REST framework for easy integration
- **Vector Database**: Uses ChromaDB for efficient storage and retrieval of code embeddings

## Architecture

- **Backend**: Django + Django REST Framework application featuring:
  - Intelligent code parsing with Python's AST
  - Semantic code search using embeddings
  - Vector storage with ChromaDB
  - AI-powered code analysis with CodeLlama
  - Smart code chunking and indexing

- **Key Technologies**:
  - `Django`: Web framework for building the API
  - `Django REST Framework`: For RESTful API endpoints
  - `ChromaDB`: Vector database for storing code embeddings
  - `Sentence Transformers`: For generating code embeddings
  - `Ollama`: Interface with CodeLlama for code analysis
  - `ast`: Python Abstract Syntax Tree for code parsing

## Currently Working

- ✅ Code file upload and indexing endpoint (`/guide/upload/`)
- ✅ Semantic code search endpoint (`/guide/search/`)
- ✅ Smart code chunking and analysis
- ✅ Integration with ChromaDB for vector storage
- ✅ AI-powered code relevance explanations

## Project Structure

```
CodeLens/
├── CodeLens/          # Main project directory
│   ├── settings.py    # Project settings
│   └── urls.py       # Main URL configuration
├── guide/            # Main application
│   ├── views.py      # Core logic and API endpoints
│   └── urls.py       # App URL configuration
├── manage.py         # Django management script
└── requirements.txt  # Project dependencies
```

## API Endpoints

- `POST /guide/upload/`: Upload and index code files
  - Accepts code files with form data
  - Returns indexing statistics and status

- `GET /guide/search/`: Search indexed code
  - Parameters: 
    - `query`: Natural language search query
    - `top_k`: Number of results to return (default: 1)
  - Returns relevant code snippets with explanations
