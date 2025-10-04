from typing import List, Dict, Any
from ..storage.chroma_manager import ChromaManager

class HybridSearch:
    """Hybrid search combining keyword and semantic search"""
    
    def __init__(self):
        self.chroma_manager = ChromaManager()
    
    def search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search"""
        
        # Keyword search
        keyword_results = self._keyword_search(query, chunks)
        
        # Semantic search via ChromaDB
        semantic_results = self.chroma_manager.search(query, top_k=top_k * 2)
        
        # Combine results
        combined = self._combine_results(keyword_results, semantic_results, chunks)
        
        return combined[:top_k]
    
    def _keyword_search(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple keyword-based search"""
        
        query_words = query.lower().split()
        scored_chunks = []
        
        for chunk in chunks:
            score = 0
            code_lower = chunk['code'].lower()
            name_lower = chunk['name'].lower()
            
            # Score based on keyword matches
            for word in query_words:
                if len(word) > 3:  # Skip short words
                    score += code_lower.count(word) * 2
                    score += name_lower.count(word) * 5  # Name matches are more important
            
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks[:10]]
    
    def _combine_results(self, keyword_results: List[Dict], semantic_results: List[Dict], all_chunks: List[Dict]) -> List[Dict]:
        """Combine and deduplicate results"""
        
        seen_ids = set()
        combined = []
        
        # Add keyword results first (higher priority)
        for chunk in keyword_results:
            if chunk['id'] not in seen_ids:
                seen_ids.add(chunk['id'])
                combined.append(chunk)
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result.get('id')
            if chunk_id and chunk_id not in seen_ids:
                # Find full chunk from all_chunks
                for chunk in all_chunks:
                    if chunk['id'] == chunk_id:
                        seen_ids.add(chunk_id)
                        combined.append(chunk)
                        break
        
        return combined