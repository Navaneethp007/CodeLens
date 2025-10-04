from typing import List, Dict, Any

class MetadataFilter:
    """Generic metadata-based filtering for comprehensive queries"""
    
    def filter(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter chunks based on query intent"""
        
        query_lower = query.lower()
        
        # Detect what type of code elements user is asking about
        if "function" in query_lower or "def" in query_lower:
            return [c for c in chunks if c['type'] == 'function']
        
        elif "class" in query_lower or "classes" in query_lower:
            return [c for c in chunks if c['type'] == 'class']
        
        elif "import" in query_lower:
            return [c for c in chunks if c['type'] == 'imports']
        
        # For general comprehensive questions, return all assignments
        # LLM will determine what's relevant (agents, tasks, variables, etc.)
        elif any(word in query_lower for word in ["what are", "different", "list all", "how many"]):
            return [c for c in chunks if c['type'] == 'assignment']
        
        # Fallback: return top chunks by line count (more code = more important)
        else:
            sorted_chunks = sorted(chunks, key=lambda c: c['line_end'] - c['line_start'], reverse=True)
            return sorted_chunks[:10]