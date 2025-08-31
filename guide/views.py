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

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
ollama_client = ollama.Client()


class CodeIndexer:
    def __init__(self):
        # Clear any existing collection and create a new one
        try:
            chroma_client.delete_collection(name="codebase")
        except:
            pass
        self.collection = chroma_client.create_collection(name="codebase")

    def index_code_file(self, file_content: str, filename: str) -> Dict[str, Any]:
        try:
            chunks = self._parse_python_code(file_content, filename)
            indexed_count = 0

            for chunk in chunks:
                if self._store_chunk(chunk):
                    indexed_count += 1

            return {
                "status": "success",
                "filename": filename,
                "chunks_indexed": indexed_count,
                "message": f"Indexed {indexed_count} chunks from {filename}",
            }
        except Exception as e:
            return {"status": "error", "filename": filename, "error": str(e)}

    def _parse_python_code(
        self, file_content: str, filename: str
    ) -> List[Dict[str, Any]]:
        chunks = []
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()
            print(f"\nParsing {filename}...")

            # Track class context
            current_class = None
            
            # First, handle top-level code
            top_level_code = []
            current_block = []
            
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith(' ') and not line.startswith('#'):
                    current_block.append((i + 1, line))
                    if len(current_block) >= 5:  # Create chunk after accumulating enough lines
                        block_lines = [l for _, l in current_block]
                        start_line = current_block[0][0]
                        end_line = current_block[-1][0]
                        chunks.append({
                            "id": str(uuid.uuid4()),
                            "code": "\n".join(block_lines),
                            "type": "code_block",
                            "name": f"block_{start_line}_{end_line}",
                            "filename": filename,
                            "line_start": start_line,
                            "line_end": end_line,
                            "parent": "",
                        })
                        current_block = []
            
            # Add remaining block if any
            if current_block:
                block_lines = [l for _, l in current_block]
                start_line = current_block[0][0]
                end_line = current_block[-1][0]
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "code": "\n".join(block_lines),
                    "type": "code_block",
                    "name": f"block_{start_line}_{end_line}",
                    "filename": filename,
                    "line_start": start_line,
                    "line_end": end_line,
                    "parent": "",
                })

            # Handle classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    current_class = node
                    # Split class into logical chunks
                    class_lines = self._extract_node_code(lines, node).split('\n')
                    chunk_size = 10  # Adjust this value to control chunk size
                    
                    for i in range(0, len(class_lines), chunk_size):
                        chunk_code = '\n'.join(class_lines[i:i + chunk_size])
                        if chunk_code.strip():
                            print(f"Found class chunk: {node.name} (part {i//chunk_size + 1})")
                            chunks.append({
                                "id": str(uuid.uuid4()),
                                "code": chunk_code,
                                "type": "class",
                                "name": f"{node.name}_part_{i//chunk_size + 1}",
                                "filename": filename,
                                "line_start": node.lineno + i,
                                "line_end": min(node.lineno + i + chunk_size - 1, 
                                              getattr(node, "end_lineno", node.lineno)),
                                "parent": "",
                            })
                            
                elif isinstance(node, ast.FunctionDef):
                    # Split function into logical chunks
                    func_lines = self._extract_node_code(lines, node).split('\n')
                    chunk_size = 8  # Slightly smaller chunks for functions
                    
                    parent = current_class.name if (
                        current_class 
                        and node in current_class.body
                    ) else ""
                    
                    for i in range(0, len(func_lines), chunk_size):
                        chunk_code = '\n'.join(func_lines[i:i + chunk_size])
                        if chunk_code.strip():
                            print(f"Found {'method' if parent else 'function'} chunk: {node.name} (part {i//chunk_size + 1})")
                            chunks.append({
                                "id": str(uuid.uuid4()),
                                "code": chunk_code,
                                "type": "function",
                                "name": f"{node.name}_part_{i//chunk_size + 1}",
                                "filename": filename,
                                "line_start": node.lineno + i,
                                "line_end": min(node.lineno + i + chunk_size - 1,
                                              getattr(node, "end_lineno", node.lineno)),
                                "parent": parent,
                            })

        except SyntaxError as e:
            print(f"Error parsing {filename}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error parsing {filename}: {str(e)}")

        return chunks

    def _extract_node_code(self, lines: List[str], node) -> str:
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno) - 1
        return "\n".join(lines[start : end + 1])

    def _store_chunk(self, chunk: Dict[str, Any]) -> bool:
        try:
            # Build a rich description for better semantic search
            chunk_type = "method" if chunk.get('parent') else chunk['type']
            
            # Create a more detailed description based on the code content
            code_lines = chunk['code'].strip().split('\n')
            first_line = code_lines[0].strip()
            
            # Extract key information from the code
            context_parts = []
            
            if chunk_type == "code_block":
                context = f"This is a code block containing {len(code_lines)} lines"
                if 'import' in first_line:
                    context += " of import statements"
                elif 'def' in first_line:
                    context += " defining a function"
                elif 'class' in first_line:
                    context += " defining a class"
                else:
                    context += f" starting with: {first_line[:50]}"
            else:
                context = f"This {chunk_type} is named {chunk['name'].split('_part_')[0]}"
                if chunk.get('parent'):
                    context += f" and belongs to class {chunk['parent']}"
                if '_part_' in chunk['name']:
                    part_num = chunk['name'].split('_part_')[1]
                    context += f" (part {part_num})"
            
            # Add any significant keywords found in the code
            keywords = ['return', 'if', 'for', 'while', 'try', 'except', 'with', 'async']
            found_keywords = [kw for kw in keywords if any(kw in line for line in code_lines)]
            if found_keywords:
                context += f"\nContains: {', '.join(found_keywords)}"
            
            searchable_text = f"""Description: {context}
Location: {chunk['filename']} (lines {chunk['line_start']}-{chunk['line_end']})
Code:
{chunk['code']}"""

            print(f"Indexing {chunk_type} {chunk['name']}")
            
            embedding = embedding_model.encode([searchable_text])[0].tolist()
            
            # Ensure all metadata values are strings or numbers
            metadata = {
                "filename": str(chunk['filename']),
                "type": str(chunk['type']),
                "name": str(chunk['name']),
                "line_start": int(chunk['line_start']),
                "line_end": int(chunk['line_end']),
                "code": str(chunk['code']),
                "parent": str(chunk.get('parent', ''))  # Empty string if no parent
            }
            
            self.collection.add(
                embeddings=[embedding],
                documents=[searchable_text],
                ids=[chunk["id"]],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            print(f"Error storing chunk: {e}")
            return False


class CodeSearcher:
    def __init__(self):
        self.collection = chroma_client.get_collection("codebase")
        
    def _check_collection(self) -> bool:
        try:
            count = self.collection.count()
            print(f"\nCollection status:")
            print(f"Total documents indexed: {count}")
            return count > 0
        except Exception as e:
            print(f"Error checking collection: {e}")
            return False

    def search_and_explain(self, query: str, top_k: int = 1) -> Dict[str, Any]:
        try:
            print("\n=== Search Debug Info ===")
            print(f"Query: {query}")
            
            # Check if we have any documents
            count = self.collection.count()
            print(f"Documents in collection: {count}")
            if count == 0:
                return {
                    "status": "success",
                    "query": query,
                    "results": [],
                    "total_results": 0,
                    "debug_info": "No documents indexed"
                }
            
            # Create a direct and focused query embedding
            query_embedding = embedding_model.encode([query])[0].tolist()
            
            # Get enough results to allow for filtering
            initial_k = max(10, top_k * 2)
            print(f"Searching for most relevant {initial_k} matches...")
            
            # Get initial results
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_k,
                include=["documents", "metadatas", "distances"],
            )
            
            processed_results = []
            total_matches = len(results['documents'][0])
            print(f"\nProcessing {total_matches} initial matches...")
            
            for i in range(total_matches):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1 - distance
                
                # Get explanation
                explanation = self._explain_code(metadata["code"], query)
                print(f"Analyzing {metadata['type']} '{metadata['name']}' (similarity: {similarity:.3f})")
                
                # Skip irrelevant results
                if explanation.lower().startswith("this code is not relevant"):
                    print(f"Skipping - marked as not relevant")
                    continue
                    
                processed_results.append({
                    "code": metadata["code"],
                    "filename": metadata["filename"],
                    "type": metadata["type"],
                    "name": metadata["name"],
                    "line_start": metadata["line_start"],
                    "line_end": metadata["line_end"],
                    "parent": metadata["parent"],
                    "similarity_score": round(similarity, 3),
                    "explanation": explanation,
                })
            
            # Sort by similarity and take top results
            processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            final_results = processed_results[:top_k] if processed_results else []
            
            print(f"\nFound {len(final_results)} relevant results out of {len(processed_results)} matches")
            
            return {
                "status": "success",
                "query": query,
                "results": final_results,
                "total_results": len(final_results),
                "debug_info": {
                    "total_matches": len(processed_results),
                    "initial_results": total_matches,
                    "similarity_threshold": None  # No threshold, using explanation filtering
                }
            }
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return {"status": "error", "query": query, "error": str(e)}
                

                    

                        
                    # Add to results

            
            # Sort by similarity score and take top results
            all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            final_results = all_results[:top_k] if all_results else []
            
            print(f"\nFound {len(final_results)} relevant results out of {len(all_results)} total matches")
            
            return {
                "status": "success",
                "query": query,
                "results": final_results,
                "total_results": len(final_results),
                "debug_info": {
                    "total_matches": len(all_results),
                    "queries_tried": search_queries
                }
            }

        except Exception as e:
            return {"status": "error", "query": query, "error": str(e)}

    def _explain_code(self, code: str, query: str) -> str:
        try:
            prompt = f"""Given the user's question: "{query}"

Analyze this code snippet:
{code}

Task: Explain specifically how this code relates to the user's question.
- If this code answers the question: Explain exactly how it does so
- If this code is not relevant: Say "This code is not relevant to the question"
- Focus only on parts that directly answer the question
- Be concise and specific

Your explanation:"""

            response = ollama_client.generate(
                model="codellama",
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": 100},
            )
            return response["response"].strip()
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"


code_indexer = CodeIndexer()
code_searcher = CodeSearcher()


@api_view(["POST"])
def upload_and_index_code(request):
    try:
        if "file" not in request.FILES:
            return Response(
                {"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        uploaded_file = request.FILES["file"]
        filename = request.data.get("filename", uploaded_file.name)

        # Read file content
        file_content = uploaded_file.read().decode("utf-8")

        # Index the file
        result = code_indexer.index_code_file(file_content, filename)

        if result["status"] == "success":
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except UnicodeDecodeError:
        return Response(
            {"error": "File encoding not supported"}, status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def collection_status(request):
    """Debug endpoint to check collection status"""
    try:
        collection = chroma_client.get_collection("codebase")
        count = collection.count()
        peek = collection.peek() if count > 0 else {}
        return Response({
            "status": "success",
            "total_documents": count,
            "sample_document": peek,
        })
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["POST"])
def search_code(request):
    try:
        query = request.data.get("query")
        top_k = request.data.get("top_k", 1)

        if not query:
            return Response(
                {"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Check collection status first
        collection = chroma_client.get_collection("codebase")
        count = collection.count()
        if count == 0:
            return Response({
                "status": "success",
                "query": query,
                "results": [],
                "total_results": 0,
                "debug_info": "No documents indexed. Please index some files first."
            })

        result = code_searcher.search_and_explain(query, top_k)
        
        # Add debug info to result
        result["debug_info"] = {
            "total_documents": count,
            "query": query,
            "top_k": top_k
        }

        if result["status"] == "success":
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
