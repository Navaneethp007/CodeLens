from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import numpy as np

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Reranker model loaded")
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = None, min_score: float = None) -> List[Dict[str, Any]]:
        """
        Rerank chunks by relevance to query
        
        Args:
            query: User's question
            chunks: List of retrieved chunks
            top_k: Return top K chunks (if None, use min_score filtering)
            min_score: Minimum normalized score to keep chunk (0-1 scale)
        
        Returns:
            Filtered and sorted chunks by relevance
        """
        
        if not chunks:
            return []
        
        print(f"\n=== RERANKER DEBUG ===")
        print(f"Reranking {len(chunks)} chunks")
        
        # Prepare pairs for cross-encoder
        pairs = []
        for chunk in chunks:
            chunk_text = f"Type: {chunk['type']}, Name: {chunk['name']}\nCode: {chunk['code'][:500]}"  # Limit text length
            pairs.append([query, chunk_text])
        
        # Get raw scores
        raw_scores = self.model.predict(pairs)
        
        # Normalize scores to 0-1 range using min-max scaling
        min_score_val = float(np.min(raw_scores))
        max_score_val = float(np.max(raw_scores))
        score_range = max_score_val - min_score_val
        
        if score_range == 0:
            # All scores are the same, normalize to 0.5
            normalized_scores = [0.5] * len(raw_scores)
        else:
            normalized_scores = [(float(s) - min_score_val) / score_range for s in raw_scores]
        
        # Attach normalized scores to chunks
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk['relevance_score'] = normalized_scores[i]
            chunk['raw_score'] = float(raw_scores[i])
            
            if i < 5:
                print(f"  Chunk '{chunk['name']}' ({chunk['type']}): norm_score={normalized_scores[i]:.3f}, raw={raw_scores[i]:.2f}")
            
            scored_chunks.append(chunk)
        
        # Sort by relevance score
        scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Filter based on strategy
        if top_k is not None:
            # Return top K chunks
            result = scored_chunks[:top_k]
            print(f"Returning top {len(result)} chunks")
        elif min_score is not None:
            # Filter by minimum score
            result = [c for c in scored_chunks if c['relevance_score'] >= min_score]
            print(f"Kept {len(result)} chunks above threshold {min_score}")
        else:
            # Default: return top 50% of chunks
            mid_point = len(scored_chunks) // 2
            result = scored_chunks[:max(mid_point, 3)]  # At least 3 chunks
            print(f"Returning top 50%: {len(result)} chunks")
        
        print("=== END RERANKER DEBUG ===\n")
        return result
