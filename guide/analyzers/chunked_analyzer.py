from ..chunking.ast_parser import ASTParser
from ..chunking.chunk_builder import ChunkBuilder
from ..retrieval.query_classifier import QueryClassifier
from ..retrieval.metadata_filter import MetadataFilter
from ..retrieval.hybrid_search import HybridSearch
from ..storage.chroma_manager import ChromaManager
import ollama

class ChunkedAnalyzer:
    """Chunked analysis for large files and codebases"""
    
    def __init__(self):
        self.ollama_client = ollama.Client()
        self.ast_parser = ASTParser()
        self.chunk_builder = ChunkBuilder()
        self.query_classifier = QueryClassifier()
        self.metadata_filter = MetadataFilter()
        self.hybrid_search = HybridSearch()
        self.chroma_manager = ChromaManager()
    
    def analyze(self, file_content: str = None, filename: str = None, query: str = None, 
                codebase_mode: bool = False, chunks: list = None) -> dict:
        """
        Analyze code using chunking and smart retrieval
        
        Args:
            file_content: For single file mode
            filename: For single file mode
            query: User's question
            codebase_mode: If True, uses pre-indexed chunks
            chunks: Pre-indexed chunks (for codebase mode)
        """
        try:
            # Single file mode
            if not codebase_mode:
                parsed_data = self.ast_parser.parse(file_content, filename)
                chunks_list = self.chunk_builder.build_chunks(parsed_data, file_content, filename)
                source_info = filename
            
            # Codebase mode
            else:
                chunks_list = chunks
                # Count unique files in chunks
                files_in_codebase = set(c['file_path'] for c in chunks_list)
                source_info = f"{len(files_in_codebase)} files"
            
            # Store chunks in ChromaDB
            self.chroma_manager.store_chunks(chunks_list)
            
            # Classify query type
            query_type = self.query_classifier.classify(query)
            
            # Retrieve relevant chunks based on query type
            if query_type == 'comprehensive':
                relevant_chunks = self.metadata_filter.filter(query, chunks_list)
                print(f"DEBUG: Metadata filter returned {len(relevant_chunks)} chunks")
            else:
                relevant_chunks = self.hybrid_search.search(query, chunks_list)
                print(f"DEBUG: Hybrid search returned {len(relevant_chunks)} chunks")
            
            print(f"DEBUG: Relevant chunks: {[c['name'] for c in relevant_chunks]}")
            
            # Generate answer using LLM
            answer = self._generate_answer(query, relevant_chunks, source_info, codebase_mode)
            
            return {
                'status': 'success',
                'query': query,
                'answer': answer,
                'method': 'chunked_analysis',
                'query_type': query_type,
                'chunks_analyzed': len(relevant_chunks),
                'source': source_info,
                'mode': 'codebase' if codebase_mode else 'single_file'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e)
            }
    
    def _generate_answer(self, query: str, chunks: list, source_info: str, codebase_mode: bool) -> str:
        """Generate answer from relevant chunks"""
        try:
            # Build context from chunks
            context = f"Source: {source_info}\n\n"
            context += f"Relevant Code Sections ({len(chunks)} total, sorted by relevance):\n\n"
            
            # Show ALL reranked chunks
            for i, chunk in enumerate(chunks, 1):
                relevance = chunk.get('relevance_score', 0)
                file_info = f" [File: {chunk['file_path']}]" if codebase_mode else ""
                context += f"{i}. {chunk['name']} ({chunk['type']}) [Relevance: {relevance:.2f}]{file_info}:\n"
                context += f"{chunk['code']}\n\n"
            
            prompt = f"""{context}

User Question: {query}

Instructions:
- All code sections above are highly relevant (passed reranking)
- Answer comprehensively using these sections
- If asked about "all" or "different" items, list every relevant one
- Be specific and complete
- Include file information if multiple files are referenced

Answer:"""
            
            response = self.ollama_client.generate(
                model='codellama',
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 600
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
