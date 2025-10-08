from typing import List, Dict, Any
from ..storage.chroma_manager import ChromaManager
from .reranker import Reranker

class HybridSearch:
    def __init__(self):
        self.chroma_manager = ChromaManager()
        self.reranker = Reranker()
    
    def search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 15) -> List[Dict[str, Any]]:
        """Perform hybrid search with reranking"""
        
        print(f"DEBUG: Starting hybrid search with {len(chunks)} chunks")
        
        # Step 1: Initial retrieval (cast wider net)
        keyword_results = self._keyword_search(query, chunks)
        print(f"DEBUG: Keyword search found {len(keyword_results)} results")
        
        semantic_results = self.chroma_manager.search(query, top_k=top_k)
        print(f"DEBUG: Semantic search found {len(semantic_results)} results")
        
        # Step 2: Combine results
        chunk_map = {c['id']: c for c in chunks}
        seen_ids = set()
        combined = []
        
        for chunk in keyword_results[:top_k]:
            if chunk['id'] not in seen_ids:
                seen_ids.add(chunk['id'])
                combined.append(chunk)
        
        for result in semantic_results:
            result_id = result.get('id')
            if result_id and result_id not in seen_ids and result_id in chunk_map:
                seen_ids.add(result_id)
                combined.append(chunk_map[result_id])
        
        print(f"DEBUG: Combined search returned {len(combined)} chunks")
        
        # Step 3: RERANK using cross-encoder
        reranked_chunks = self.reranker.rerank(query, combined, top_k=None, min_score=0.4)
        
        print(f"DEBUG: After reranking: {len(reranked_chunks)} relevant chunks")
        return reranked_chunks
    
    def _keyword_search(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple keyword-based search"""
        
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        
        if not query_words:
            return chunks[:10]
        
        scored_chunks = []
        
        for chunk in chunks:
            score = 0
            code_lower = chunk['code'].lower()
            name_lower = chunk['name'].lower()
            
            for word in query_words:
                score += code_lower.count(word) * 2
                score += name_lower.count(word) * 5
            
            scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks]
