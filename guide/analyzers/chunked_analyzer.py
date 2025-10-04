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
            chunks = self.chunk_builder.build_chunks(parsed_data, file_content, filename)
            
            # Store chunks
            self.chroma_manager.store_chunks(chunks)
            
            # Classify query type
            query_type = self.query_classifier.classify(query)
            
            # Retrieve relevant chunks based on query type
            if query_type == 'comprehensive':
                relevant_chunks = self.metadata_filter.filter(query, chunks)
            else:
                relevant_chunks = self.hybrid_search.search(query, chunks)
            
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
            context += f"Relevant Code Sections ({len(chunks)} total):\n\n"
            
            # FIX: Show ALL chunks, not limited
            for i, chunk in enumerate(chunks, 1):
                context += f"{i}. {chunk['name']} ({chunk['type']}) [lines {chunk['line_start']}-{chunk['line_end']}]:\n"
                context += f"{chunk['code']}\n\n"
            
            prompt = f"""{context}
    
    User Question: {query}
    
    Instructions:
    - Analyze ALL {len(chunks)} code sections above
    - If asked about "all" or "different" items, list EVERY one you find
    - Be specific with names and details
    - Ignore irrelevant sections
    - Give one comprehensive answer
    
    Answer:"""
            
            response = self.ollama_client.generate(
                model='codellama',
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 600}  # Increased for longer answers
            )
            
            return response['response'].strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"