from typing import List, Dict, Any
from ..storage.chroma_manager import ChromaManager

class HybridSearch:
    def __init__(self):
        self.chroma_manager = ChromaManager()
    
    def search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search"""
        
        print(f"DEBUG: Starting hybrid search with {len(chunks)} chunks")
        
        # Keyword search
        keyword_results = self._keyword_search(query, chunks)
        print(f"DEBUG: Keyword search found {len(keyword_results)} results")
        
        # Semantic search via ChromaDB
        semantic_results = self.chroma_manager.search(query, top_k=top_k * 2)
        print(f"DEBUG: Semantic search found {len(semantic_results)} results")
        
        # Create chunk ID map for faster lookup
        chunk_map = {c['id']: c for c in chunks}
        
        # Combine results
        seen_ids = set()
        combined = []
        
        # Add keyword results first (higher priority)
        for chunk in keyword_results[:top_k]:
            if chunk['id'] not in seen_ids:
                seen_ids.add(chunk['id'])
                combined.append(chunk)
        
        # Add semantic results
        for result in semantic_results:
            result_id = result.get('id')
            if result_id and result_id not in seen_ids and result_id in chunk_map:
                seen_ids.add(result_id)
                combined.append(chunk_map[result_id])
        
        print(f"DEBUG: Combined search returned {len(combined)} chunks")
        return combined[:top_k]
    
    def _keyword_search(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple keyword-based search"""
    
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        print(f"DEBUG: Query words for keyword search: {query_words}")
        
        if not query_words:
            return chunks[:5]
        
        scored_chunks = []
        
        for chunk in chunks:
            score = 0
            code_lower = chunk['code'].lower()
            name_lower = chunk['name'].lower()
            
            for word in query_words:
                code_count = code_lower.count(word)
                name_count = name_lower.count(word)
                score += code_count * 2
                score += name_count * 5
            
            # DEBUG: Print first few chunks with scores
            if len(scored_chunks) < 3:
                print(f"DEBUG: Chunk '{chunk['name']}' ({chunk['type']}) scored {score}")
            
            # Include ALL chunks, even with score 0
            scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 even if score is 0
        result = [chunk for chunk, score in scored_chunks[:5]]
        print(f"DEBUG: Returning {len(result)} keyword results")
        return result
