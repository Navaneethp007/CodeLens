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
        self.collection = chroma_client.get_or_create_collection(name="codebase")

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

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    code = self._extract_node_code(lines, node)
                    chunks.append(
                        {
                            "id": str(uuid.uuid4()),
                            "code": code,
                            "type": "function",
                            "name": node.name,
                            "filename": filename,
                            "line_start": node.lineno,
                            "line_end": getattr(node, "end_lineno", node.lineno),
                            "parent": getattr(node, "parent_name", None),
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    code = self._extract_node_code(lines, node)
                    # Store class name for its methods
                    class_name = node.name
                    chunks.append(
                        {
                            "id": str(uuid.uuid4()),
                            "code": code,
                            "type": "class",
                            "name": class_name,
                            "filename": filename,
                            "line_start": node.lineno,
                            "line_end": getattr(node, "end_lineno", node.lineno),
                            "parent": None,
                        }
                    )
                    # Set parent name for methods inside this class
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, ast.FunctionDef):
                            child.parent_name = class_name

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
            searchable_text = f"File: {chunk['filename']}\nType: {chunk['type']}\nName: {chunk['name']}\nCode:\n{chunk['code']}"

            embedding = embedding_model.encode([searchable_text])[0].tolist()
            self.collection.add(
                embeddings=[embedding],
                documents=[searchable_text],
                ids=[chunk["id"]],
                metadatas=[
                    {
                        "filename": chunk["filename"],
                        "type": chunk["type"],
                        "name": chunk["name"],
                        "line_start": chunk["line_start"],
                        "line_end": chunk["line_end"],
                        "code": chunk["code"],
                        "parent": chunk.get("parent")
                    }
                ],
            )
            return True
        except Exception as e:
            print(f"Error storing chunk: {e}")
            return False


class CodeSearcher:
    def __init__(self):
        self.collection = chroma_client.get_collection("codebase")

    def search_and_explain(self, query: str, top_k: int = 1) -> Dict[str, Any]:
        try:
            # Create a focused search context
            search_context = f"Question: {query}\nFind the most relevant code that answers this question."
            query_embedding = embedding_model.encode([search_context])[0].tolist()

            # Get more results initially to filter
            initial_k = top_k * 3
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_k,
                include=["documents", "metadatas", "distances"],
            )

            processed = []
            for i in range(len(results["documents"][0])):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                # Only process if similarity score is good enough
                similarity = 1 - distance
                if similarity < 0.5:  # Adjust threshold as needed
                    continue

                # Get a focused explanation
                explanation = self._explain_code(metadata["code"], query)

                # Skip if explanation indicates not relevant
                if (
                    "not relevant" in explanation.lower()
                    or "does not" in explanation.lower()
                ):
                    continue

                processed.append(
                    {
                        "code": metadata["code"],
                        "filename": metadata["filename"],
                        "type": metadata["type"],
                        "name": metadata["name"],
                        "line_start": metadata["line_start"],
                        "line_end": metadata["line_end"],
                        "parent": metadata.get("parent"),
                        "similarity_score": round(similarity, 3),
                        "explanation": explanation,
                    }
                )

                # Stop if we have enough good results
                if len(processed) >= top_k:
                    break

            return {
                "status": "success",
                "query": query,
                "results": processed,
                "total_results": len(processed),
            }

        except Exception as e:
            return {"status": "error", "query": query, "error": str(e)}

    def _explain_code(self, code: str, query: str) -> str:
        try:
            prompt = f"""Based on this specific question: {query}

Analyze this code:
{code}

Provide a focused answer about how this specific code snippet answers the question.
If this code is not relevant to the question, say "This code is not relevant to the question."
Be brief and specific to the question asked."""

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


@api_view(["POST"])
def search_code(request):
    try:
        query = request.data.get("query")
        top_k = request.data.get("top_k", 1)

        if not query:
            return Response(
                {"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        result = code_searcher.search_and_explain(query, top_k)

        if result["status"] == "success":
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
