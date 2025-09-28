#!/usr/bin/env python3
"""
CodeLens - Restructured Direct Approach
"""

import ast
import json
from typing import Dict, Any, List, Optional, Set
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import ollama
import chromadb
from dataclasses import dataclass
import networkx as nx

# Initialize clients
ollama_client = ollama.Client()
chroma_client = chromadb.PersistentClient(path="code_knowledge_base")

@dataclass
class CodeNode:
    """Represents a code element with its relationships"""
    node_id: str
    node_type: str
    name: str
    content: str
    docstring: Optional[str] = None
    parents: Set[str] = None
    children: Set[str] = None
    calls: Set[str] = None
    assigns_to: Set[str] = None
    assigned_by: Set[str] = None
    imports: Set[str] = None

    def __post_init__(self):
        self.parents = self.parents or set()
        self.children = self.children or set()
        self.calls = self.calls or set()
        self.assigns_to = self.assigns_to or set()
        self.assigned_by = self.assigned_by or set()
        self.imports = self.imports or set()

class CodeGraph:
    """Graph representation of code structure"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CodeNode] = {}
current_file_content = ""
current_filename = ""

class CodeAnalyzer:
    """Advanced code analysis using graph-based representation"""
    
    def __init__(self):
        self.collection = chroma_client.get_or_create_collection(
            name="code_documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.code_graph = CodeGraph()
        
    def analyze_code(self, file_content: str, filename: str) -> Dict[str, Any]:
        """Analyze code and build knowledge graph"""
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()
            
            # First pass: collect all definitions
            self._collect_definitions(tree, lines, filename)
            
            # Second pass: analyze relationships
            self._analyze_relationships(tree, filename)
            
            # Store in vector database with relationships
            self._store_in_vectordb(filename)
            
            return {
                'status': 'success',
                'nodes': len(self.code_graph.nodes),
                'relationships': self.code_graph.graph.number_of_edges()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
            
    def _collect_definitions(self, tree: ast.AST, lines: List[str], filename: str):
        """Collect all code definitions"""
        for node in ast.walk(tree):
            node_id = f"{filename}_{id(node)}"
            
            if isinstance(node, ast.ClassDef):
                content = self._get_node_content(node, lines)
                docstring = ast.get_docstring(node)
                self.code_graph.nodes[node_id] = CodeNode(
                    node_id=node_id,
                    node_type='class',
                    name=node.name,
                    content=content,
                    docstring=docstring
                )
                
            elif isinstance(node, ast.FunctionDef):
                content = self._get_node_content(node, lines)
                docstring = ast.get_docstring(node)
                self.code_graph.nodes[node_id] = CodeNode(
                    node_id=node_id,
                    node_type='function',
                    name=node.name,
                    content=content,
                    docstring=docstring
                )
                
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        content = self._get_node_content(node, lines)
                        self.code_graph.nodes[node_id] = CodeNode(
                            node_id=node_id,
                            node_type='assignment',
                            name=target.id,
                            content=content
                        )
                        
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                content = self._get_node_content(node, lines)
                names = []
                if isinstance(node, ast.Import):
                    names = [n.name for n in node.names]
                else:
                    module = node.module or ''
                    names = [f"{module}.{n.name}" for n in node.names]
                
                self.code_graph.nodes[node_id] = CodeNode(
                    node_id=node_id,
                    node_type='import',
                    name=','.join(names),
                    content=content,
                    imports=set(names)
                )
    
    def _analyze_relationships(self, tree: ast.AST, filename: str):
        """Analyze relationships between code elements"""
        
        def find_node_by_name(name: str) -> Optional[str]:
            for node_id, node in self.code_graph.nodes.items():
                if node.name == name:
                    return node_id
            return None
        
        for node in ast.walk(tree):
            node_id = f"{filename}_{id(node)}"
            
            if isinstance(node, ast.ClassDef):
                # Handle inheritance
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_id = find_node_by_name(base.id)
                        if base_id:
                            self.code_graph.nodes[node_id].parents.add(base_id)
                            self.code_graph.nodes[base_id].children.add(node_id)
                            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    caller_id = node_id
                    callee_name = node.func.id
                    callee_id = find_node_by_name(callee_name)
                    if callee_id:
                        self.code_graph.nodes[caller_id].calls.add(callee_id)
                        
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name):
                                caller_id = node_id
                                callee_name = node.value.func.id
                                callee_id = find_node_by_name(callee_name)
                                if callee_id:
                                    self.code_graph.nodes[caller_id].assigns_to.add(callee_id)
                                    self.code_graph.nodes[callee_id].assigned_by.add(caller_id)
    
    def _store_in_vectordb(self, filename: str):
        """Store code nodes in vector database with relationship context"""
        documents = []
        metadatas = []
        ids = []
        
        for node_id, node in self.code_graph.nodes.items():
            # Build rich context including relationships
            context_parts = [node.content]
            
            if node.docstring:
                context_parts.append(f"Documentation: {node.docstring}")
                
            if node.parents:
                parent_names = [self.code_graph.nodes[pid].name for pid in node.parents]
                context_parts.append(f"Inherits from: {', '.join(parent_names)}")
                
            if node.calls:
                called_names = [self.code_graph.nodes[cid].name for cid in node.calls]
                context_parts.append(f"Calls: {', '.join(called_names)}")
                
            if node.assigns_to:
                assign_names = [self.code_graph.nodes[aid].name for aid in node.assigns_to]
                context_parts.append(f"Creates/Configures: {', '.join(assign_names)}")
                
            if node.imports:
                context_parts.append(f"Imports: {', '.join(node.imports)}")
            
            documents.append('\n'.join(context_parts))
            metadatas.append({
                'filename': filename,
                'type': node.node_type,
                'name': node.name,
                'has_relationships': bool(node.parents or node.children or node.calls or 
                                       node.assigns_to or node.assigned_by or node.imports)
            })
            ids.append(node_id)
        
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
    def _get_node_content(self, node: ast.AST, lines: List[str]) -> str:
        """Get node content with context"""
        start = node.lineno - 1
        end = getattr(node, 'end_lineno', node.lineno)
        
        # Include comments above the node
        while start > 0 and lines[start - 1].strip().startswith('#'):
            start -= 1
            
        return '\n'.join(lines[start:end])

    def index_code(self, file_content: str, filename: str) -> Dict[str, Any]:
        try:
            # Clear existing documents for this file
            self.collection.delete(where={"filename": filename})
            
            # Parse and chunk the code
            chunks = self._chunk_code(file_content)
            
            # Prepare documents for indexing
            ids = [f"{filename}_{i}" for i in range(len(chunks))]
            documents = [chunk['content'] for chunk in chunks]
            metadatas = [{
                'filename': filename,
                'type': chunk['type'],
                'name': chunk['name']
            } for chunk in chunks]
            
            # Add to vector store
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            return {
                'status': 'success',
                'chunks_indexed': len(chunks),
                'filename': filename
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _chunk_code(self, file_content: str) -> List[Dict[str, Any]]:
        """Split code into meaningful chunks using AST"""
        chunks = []
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()
            
            # Add imports as a chunk
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    start = node.lineno - 1
                    end = getattr(node, 'end_lineno', node.lineno)
                    imports.extend(lines[start:end])
            if imports:
                chunks.append({
                    'content': '\n'.join(imports),
                    'type': 'imports',
                    'name': 'imports'
                })
            
            # Process classes, functions and significant assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    start = node.lineno - 1
                    end = getattr(node, 'end_lineno', node.lineno)
                    chunks.append({
                        'content': '\n'.join(lines[start:end]),
                        'type': 'class',
                        'name': node.name
                    })
                elif isinstance(node, ast.FunctionDef):
                    # Skip methods inside classes
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', node.lineno)
                        chunks.append({
                            'content': '\n'.join(lines[start:end]),
                            'type': 'function',
                            'name': node.name
                        })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            start = node.lineno - 1
                            end = getattr(node, 'end_lineno', node.lineno)
                            # Get a few lines of context around the assignment
                            context_start = max(0, start - 2)
                            context_end = min(len(lines), end + 2)
                            chunks.append({
                                'content': '\n'.join(lines[context_start:context_end]),
                                'type': 'assignment',
                                'name': target.id
                            })
            
            # If no chunks were created, create one chunk with all content
            if not chunks:
                chunks.append({
                    'content': file_content,
                    'type': 'module',
                    'name': 'main'
                })
                
        except SyntaxError:
            # If parsing fails, use the entire content as one chunk
            chunks.append({
                'content': file_content,
                'type': 'unknown',
                'name': 'unparseable'
            })
            
        return chunks
    
    def query_code(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the code knowledge base with graph-based analysis"""
        try:
            # Analyze query to understand search focus
            query_analysis = f"""Analyze this code-related query: {query}
            Specify:
            1. Primary code elements to focus on (classes, functions, variables, etc.)
            2. Relationship types to consider (inheritance, calls, configurations, etc.)
            3. Context requirements (implementation details, dependencies, etc.)
            4. Type of analysis needed (structural, behavioral, architectural, etc.)
            """
            
            analysis_response = ollama_client.generate(
                model='codellama',
                prompt=query_analysis,
                options={'temperature': 0.1}
            )

            # Search initial nodes
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['metadatas', 'distances']
            )

            if not results['ids'][0]:
                return {
                    'status': 'error',
                    'error': 'No indexed code found. Please upload a file first.'
                }

            # Collect relevant nodes and their relationships
            relevant_nodes = set()
            node_info = {}

            for i, node_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                node = self.code_graph.nodes.get(node_id)
                if not node:
                    continue

                relevant_nodes.add(node_id)
                node_info[node_id] = {
                    'type': node.node_type,
                    'name': node.name,
                    'content': node.content,
                    'docstring': node.docstring,
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity score
                    'relationships': {}
                }

                # Add connected nodes
                if node.parents:
                    node_info[node_id]['relationships']['inherits_from'] = [
                        self.code_graph.nodes[pid].name for pid in node.parents
                    ]
                    relevant_nodes.update(node.parents)

                if node.children:
                    node_info[node_id]['relationships']['inherited_by'] = [
                        self.code_graph.nodes[cid].name for cid in node.children
                    ]
                    relevant_nodes.update(node.children)

                if node.calls:
                    node_info[node_id]['relationships']['calls'] = [
                        self.code_graph.nodes[cid].name for cid in node.calls
                    ]
                    relevant_nodes.update(node.calls)

                if node.assigns_to:
                    node_info[node_id]['relationships']['configures'] = [
                        self.code_graph.nodes[aid].name for aid in node.assigns_to
                    ]
                    relevant_nodes.update(node.assigns_to)

                if node.imports:
                    node_info[node_id]['relationships']['imports'] = list(node.imports)

            # Build rich context for LLM analysis
            context = []
            for node_id in relevant_nodes:
                node = node_info.get(node_id) or self.code_graph.nodes[node_id]
                
                # Build context with relationships
                section = []
                section.append(f"# {node.node_type.upper()}: {node.name}")
                if node.docstring:
                    section.append(f"# Documentation: {node.docstring}")
                
                rels = node_info.get(node_id, {}).get('relationships', {})
                for rel_type, related in rels.items():
                    if related:
                        section.append(f"# {rel_type}: {', '.join(related)}")
                
                section.append(node.content)
                context.append("\n".join(section))
            
            context_str = "\n\n".join(context)
            
            # Generate response using Ollama with graph context
            prompt = f"""You are a code analysis expert. Analyze this Python code and answer the user's question.

Code Context with Relationships:
```python
{context_str}
```

User Question: {query}

Query Analysis: {analysis_response['response']}

Analysis Instructions:
1. Code Structure and Relationships:
   - Identify key components and their connections
   - Explain how elements interact and depend on each other
   - Highlight important design patterns or architectural choices
   - Use the relationship information to explain code flow

2. Detailed Analysis:
   - Class hierarchies and inheritance patterns
   - Function interactions and dependencies
   - Object lifecycle and configurations
   - Import relationships and external dependencies
   - Data and control flow between components

3. Response Guidelines:
   - Ground explanations in the actual code and relationships shown
   - Quote relevant code snippets when explaining
   - Describe both direct and indirect relationships
   - Highlight architectural insights from the graph structure
   - Note any missing context that would help understanding

Provide a comprehensive answer that leverages both the code content and relationship graph to explain the implementation.

Answer:"""

            response = ollama_client.generate(
                model='codellama',
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 500
                }
            )
            
            return {
                'status': 'success',
                'query': query,
                'answer': response['response'].strip(),
                'nodes_analyzed': len(relevant_nodes),
                'relationships_found': sum(len(info.get('relationships', {})) 
                                        for info in node_info.values()),
                'top_matches': [{
                    'name': node_info[nid]['name'],
                    'type': node_info[nid]['type'],
                    'score': node_info[nid].get('score', 0)
                } for nid in list(relevant_nodes)[:n_results]]
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _analyze_direct(self, file_content: str, filename: str, query: str) -> Dict[str, Any]:
        """Send entire file directly to LLM"""
        
        prompt = f"""You are a code analysis expert. Answer the user's question about this Python code.

Filename: {filename}

Code:
```python
{file_content}
```

User Question: {query}

Instructions:
- Analyze the entire code file
- Provide a complete and accurate answer
- Be specific and list all relevant items if asked for "all" or "different" items
- Focus only on what's actually in the code
- Give one comprehensive answer

Answer:"""

        try:
            response = ollama_client.generate(
                model='codellama',
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 500
                }
            )
            
            return {
                'status': 'success',
                'query': query,
                'answer': response['response'].strip(),
                'method': 'direct_analysis',
                'file_size': len(file_content)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': f"LLM generation failed: {str(e)}"
            }
    
    def _analyze_chunked(self, file_content: str, filename: str, query: str) -> Dict[str, Any]:
        """For large files, extract relevant sections first"""
        
        try:
            # Extract code structures
            structures = self._extract_code_structures(file_content, filename)
            
            # Build focused context based on query
            relevant_sections = self._get_relevant_sections(structures, query, file_content)
            
            prompt = f"""You are a code analysis expert. Answer the user's question about this Python code.

Filename: {filename}
File size: {len(file_content)} characters

Relevant Code Sections:
{relevant_sections}

User Question: {query}

Instructions:
- Analyze the provided code sections
- Provide a complete and accurate answer
- Be specific and list all relevant items
- If sections seem incomplete for the question, mention what might be missing

Answer:"""

            response = ollama_client.generate(
                model='codellama',
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 500
                }
            )
            
            return {
                'status': 'success',
                'query': query,
                'answer': response['response'].strip(),
                'method': 'chunked_analysis',
                'file_size': len(file_content),
                'sections_analyzed': len(relevant_sections.split('\n\n'))
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': f"Chunked analysis failed: {str(e)}"
            }
    
    def _extract_code_structures(self, file_content: str, filename: str) -> Dict[str, List[str]]:
        """Extract all code structures from file"""
        
        structures = {
            'imports': [],
            'classes': [],
            'functions': [],
            'assignments': [],
            'other': []
        }
        
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    code = self._extract_node_code(lines, node)
                    structures['imports'].append(code)
                    
                elif isinstance(node, ast.ClassDef):
                    code = self._extract_node_code(lines, node)
                    structures['classes'].append(f"# Class: {node.name}\n{code}")
                    
                elif isinstance(node, ast.FunctionDef):
                    # Skip methods inside classes
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                        code = self._extract_node_code(lines, node)
                        structures['functions'].append(f"# Function: {node.name}\n{code}")
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            code = self._extract_node_code(lines, node)
                            structures['assignments'].append(f"# Assignment: {target.id}\n{code}")
            
        except SyntaxError:
            # If parsing fails, treat as plain text
            structures['other'].append(file_content)
        
        return structures
    
    def _get_relevant_sections(self, structures: Dict[str, List[str]], query: str, file_content: str = "") -> str:
        """Get relevant code sections based on query keywords"""
        
        query_lower = query.lower()
        relevant = []
        
        # Always include imports
        if structures['imports']:
            relevant.append("# IMPORTS\n" + '\n'.join(structures['imports']))
        
        # Query-based selection
        if any(word in query_lower for word in ['agent', 'agents']):
            # Look for agent-related assignments and class definitions
            for assignment in structures['assignments']:
                if 'agent' in assignment.lower():
                    relevant.append(assignment)
            # Also include the imports since they might define agent-related classes
            if structures['imports']:
                relevant.append("# IMPORTS\n" + '\n'.join(structures['imports']))
            # For agent-related queries, include more assignments to catch all agent definitions
            relevant.extend(structures['assignments'])
        
        elif any(word in query_lower for word in ['class', 'classes']):
            relevant.extend(structures['classes'])
            
        elif any(word in query_lower for word in ['function', 'functions', 'def']):
            relevant.extend(structures['functions'])
            
        elif any(word in query_lower for word in ['task', 'tasks']):
            for assignment in structures['assignments']:
                if 'task(' in assignment.lower():
                    relevant.append(assignment)
                    
        else:
            # For general queries, include a mix
            relevant.extend(structures['classes'][:3])
            relevant.extend(structures['functions'][:3])
            relevant.extend(structures['assignments'][:5])
        
        # If nothing specific found, include top assignments
        if len(relevant) <= 1:
            relevant.extend(structures['assignments'][:8])
        
        return '\n\n'.join(relevant) if relevant else file_content[:5000]
    
    def _extract_node_code(self, lines: List[str], node) -> str:
        """Extract source code for AST node"""
        start = node.lineno - 1
        end = getattr(node, 'end_lineno', node.lineno) - 1
        return '\n'.join(lines[start:end + 1])

# Initialize vector store
code_analyzer = CodeAnalyzer()

@api_view(['POST'])
def upload_and_analyze_code(request):
    """Upload and index code file"""
    try:
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        filename = request.data.get('filename', uploaded_file.name)
        
        # Read file content
        file_content = uploaded_file.read().decode('utf-8')
        
        # Index the code in vector store
        result = code_analyzer.analyze_code(file_content, filename)
        
        if result['status'] == 'success':
            return Response({
                'status': 'success',
                'filename': filename,
                'chunks_indexed': result['chunks_indexed'],
                'message': f"File {filename} uploaded and indexed successfully"
            })
        else:
            return Response({'error': result['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    except UnicodeDecodeError:
        return Response({'error': 'File encoding not supported'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def analyze_code(request):
    """Query code knowledge base"""
    try:
        query = request.data.get('query')
        if not query:
            return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Query the knowledge base
        result = code_analyzer.query_code(query)
        
        if result['status'] == 'success':
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response({'error': result['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



