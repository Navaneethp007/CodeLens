from ..chunking.ast_parser import ASTParser
from ..chunking.chunk_builder import ChunkBuilder
from ..retrieval.query_classifier import QueryClassifier
from ..retrieval.metadata_filter import MetadataFilter
from ..retrieval.hybrid_search import HybridSearch
from ..storage.chroma_manager import ChromaManager
import ollama

class ChunkedAnalyzer:
    """Chunked analysis for large files"""
    
    def __init__(self):
        self.ollama_client = ollama.Client()
        self.ast_parser = ASTParser()
        self.chunk_builder = ChunkBuilder()
        self.query_classifier = QueryClassifier()
        self.metadata_filter = MetadataFilter()
        self.hybrid_search = HybridSearch()
        self.chroma_manager = ChromaManager()
    
    def analyze(self, file_content: str, filename: str, query: str) -> dict:
        """Analyze large file using chunking and smart retrieval"""
        try:
            # Parse and chunk the file
            parsed_data = self.ast_parser.parse(file_content, filename)
            if 'error' in parsed_data:
              return {
                'status': 'error',
                'error': f"Failed to parse file: {parsed_data['error']}"
            }
            chunks = self.chunk_builder.build_chunks(parsed_data, file_content, filename)
            if not chunks or len(chunks) == 0:
              return {
                'status': 'error',
                'error': 'No code chunks could be extracted from file'
            }
        
            print(f"DEBUG: Created {len(chunks)} chunks")
            
            # Store chunks
            self.chroma_manager.store_chunks(chunks)
            
            # Classify query type
            query_type = self.query_classifier.classify(query)
            
            # Retrieve relevant chunks based on query type
            if query_type == 'comprehensive':
                relevant_chunks = self.metadata_filter.filter(query, chunks)
                print(f"DEBUG: Metadata filter returned {len(relevant_chunks)} chunks")
            else:
                relevant_chunks = self.hybrid_search.search(query, chunks)
                print(f"DEBUG: Hybrid search returned {len(relevant_chunks)} chunks")
            
            print(f"DEBUG: Relevant chunks: {[c['name'] for c in relevant_chunks]}")
            
            # Generate answer using LLM
            answer = self._generate_answer(query, relevant_chunks, filename)
            
            return {
                'status': 'success',
                'query': query,
                'answer': answer,
                'method': 'chunked_analysis',
                'query_type': query_type,
                'chunks_analyzed': len(relevant_chunks),
                'filename': filename
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e)
            }
    
    def _generate_answer(self, query: str, chunks: list, filename: str) -> str:
        try:
            context = f"Filename: {filename}\n\n"
            context += f"Relevant Code Sections ({len(chunks)} total, sorted by relevance):\n\n"
            
            # Show ALL reranked chunks (already filtered by relevance threshold)
            for i, chunk in enumerate(chunks, 1):
                relevance = chunk.get('relevance_score', 0)
                context += f"{i}. {chunk['name']} ({chunk['type']}) [Relevance: {relevance:.2f}]:\n"
                context += f"{chunk['code']}\n\n"
            
            prompt = f"""{context}
    
    User Question: {query}
    
    Instructions:
    - All code sections above are highly relevant (passed reranking)
    - Answer comprehensively using these sections
    - If asked about "all" or "different" items, list every relevant one
    - Be specific and complete
    
    Answer:"""
            
            response = self.ollama_client.generate(
                model='codellama',
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 800}
            )
            
            return response['response'].strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
