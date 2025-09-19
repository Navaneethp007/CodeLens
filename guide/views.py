import ast
import json
import uuid
from typing import List, Dict, Any
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Initialize global instances
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
ollama_client = ollama.Client()

class CodeIndexer:
    def __init__(self):
        self.collection = chroma_client.get_or_create_collection(name="codebase")
    
    def index_code_file(self, file_content: str, filename: str) -> Dict[str, Any]:
        try:
            chunks = self._parse_python_code(file_content, filename)
            indexed_count = 0
            
            for chunk in chunks:
                if self._store_chunk(chunk):
                    indexed_count += 1
            
            return {
                'status': 'success',
                'filename': filename,
                'chunks_indexed': indexed_count,
                'message': f"Indexed {indexed_count} chunks from {filename}"
            }
        except Exception as e:
            return {
                'status': 'error',
                'filename': filename,
                'error': str(e)
            }
    
    def _parse_python_code(self, file_content: str, filename: str) -> List[Dict[str, Any]]:
        chunks = []
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()
            
            # Extract classes, functions, and assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    code = self._extract_node_code(lines, node)
                    chunks.append({
                        'id': str(uuid.uuid4()),
                        'code': code,
                        'type': 'function',
                        'name': node.name,
                        'filename': filename,
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno)
                    })
                elif isinstance(node, ast.ClassDef):
                    code = self._extract_node_code(lines, node)
                    chunks.append({
                        'id': str(uuid.uuid4()),
                        'code': code,
                        'type': 'class',
                        'name': node.name,
                        'filename': filename,
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno)
                    })
                elif isinstance(node, ast.Assign):
                    # Capture variable assignments like agent definitions
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            code = self._extract_node_code(lines, node)
                            chunks.append({
                                'id': str(uuid.uuid4()),
                                'code': code,
                                'type': 'assignment',
                                'name': target.id,
                                'filename': filename,
                                'line_start': node.lineno,
                                'line_end': getattr(node, 'end_lineno', node.lineno)
                            })
            
            # Extract imports separately
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(self._extract_node_code(lines, node))
            
            if imports:
                chunks.append({
                    'id': str(uuid.uuid4()),
                    'code': '\n'.join(imports),
                    'type': 'imports',
                    'name': 'imports',
                    'filename': filename,
                    'line_start': 1,
                    'line_end': len(imports)
                })
            
            # Add full file
            chunks.append({
                'id': str(uuid.uuid4()),
                'code': file_content,
                'type': 'file',
                'name': filename,
                'filename': filename,
                'line_start': 1,
                'line_end': len(lines)
            })
            
        except SyntaxError:
            # Handle non-Python files
            chunks.append({
                'id': str(uuid.uuid4()),
                'code': file_content,
                'type': 'text',
                'name': filename,
                'filename': filename,
                'line_start': 1,
                'line_end': len(file_content.splitlines())
            })
        
        return chunks
    
    def _extract_node_code(self, lines: List[str], node) -> str:
        start = node.lineno - 1
        end = getattr(node, 'end_lineno', node.lineno) - 1
        return '\n'.join(lines[start:end + 1])
    
    def _store_chunk(self, chunk: Dict[str, Any]) -> bool:
        try:
            # Create enhanced searchable text with better context
            searchable_text = f"File: {chunk['filename']}\nType: {chunk['type']}\nName: {chunk['name']}\n"
            
            # Add context for different chunk types
            if chunk['type'] == 'assignment' and 'Agent(' in chunk['code']:
                searchable_text += "This defines an AI agent with role, goal, and capabilities.\n"
            elif chunk['type'] == 'assignment' and 'Task(' in chunk['code']:
                searchable_text += "This defines a task for an AI agent to execute.\n"
            elif chunk['type'] == 'class':
                searchable_text += "This defines a Python class with methods and attributes.\n"
            elif chunk['type'] == 'function':
                searchable_text += "This defines a Python function with specific functionality.\n"
            
            searchable_text += f"Code:\n{chunk['code']}"
            
            # Generate embedding
            embedding = embedding_model.encode([searchable_text])[0].tolist()
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[searchable_text],
                ids=[chunk['id']],
                metadatas=[{
                    'filename': chunk['filename'],
                    'type': chunk['type'],
                    'name': chunk['name'],
                    'line_start': chunk['line_start'],
                    'line_end': chunk['line_end'],
                    'code': chunk['code']
                }]
            )
            return True
        except Exception as e:
            print(f"Error storing chunk: {e}")
            return False

class CodeSearcher:
    def __init__(self):
        self.collection = chroma_client.get_collection("codebase")
    
    def search_and_explain(self, query: str, top_k: int = 5) -> Dict[str, Any]:  # Increased default
        try:
            # Search
            query_embedding = embedding_model.encode([query])[0].tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process and filter results
            processed = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Calculate similarity (handle potential negative distances)
                if distance < 0:
                    similarity = 1.0  # Perfect match if negative distance
                else:
                    similarity = max(0, 1 - distance)
                
                # Generate explanation
                explanation = self._explain_code(metadata['code'], query, metadata)
                
                processed.append({
                    'code': metadata['code'],
                    'filename': metadata['filename'],
                    'type': metadata['type'],
                    'name': metadata['name'],
                    'line_start': metadata['line_start'],
                    'line_end': metadata['line_end'],
                    'similarity_score': round(similarity, 3),
                    'explanation': explanation
                })
            
            # Sort by similarity score (highest first)
            processed.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return {
                'status': 'success',
                'query': query,
                'results': processed,
                'total_results': len(processed)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e)
            }
    
    def _explain_code(self, code: str, query: str, metadata: Dict[str, Any]) -> str:
        try:
            # Create context-aware prompt
            context = f"User asked: {query}\n\n"
            context += f"This code defines: {metadata['name']} (type: {metadata['type']})\n"
            context += f"Location: {metadata['filename']} lines {metadata['line_start']}-{metadata['line_end']}\n\n"
            context += f"Code:\n{code}\n\n"
            context += f"Answer the user's question about this code specifically:"
            
            response = ollama_client.generate(
                model='codellama',
                prompt=context,
                options={'temperature': 0.2, 'num_predict': 100}
            )
            return response['response'].strip()
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"

# Initialize services
code_indexer = CodeIndexer()
code_searcher = CodeSearcher()

@api_view(['POST'])
def upload_and_index_code(request):
    """Upload and index code file"""
    try:
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        filename = request.data.get('filename', uploaded_file.name)
        
        # Read file content
        file_content = uploaded_file.read().decode('utf-8')
        
        # Index the file
        result = code_indexer.index_code_file(file_content, filename)
        
        if result['status'] == 'success':
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except UnicodeDecodeError:
        return Response({'error': 'File encoding not supported'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def search_code(request):
    """Search code with natural language query"""
    try:
        query = request.data.get('query')
        top_k = request.data.get('top_k', 3)
        
        if not query:
            return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        result = code_searcher.search_and_explain(query, top_k)
        
        if result['status'] == 'success':
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
