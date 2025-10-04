import re

class QueryClassifier:
    """Classify query type for appropriate retrieval strategy"""
    
    def __init__(self):
        self.comprehensive_patterns = [
            r"what are (all |the )?(different |various )?",
            r"list (all |the )?",
            r"show me (all |the )?",
            r"how many",
            r"tell me about (all |the )?(different |various )?",
            r"give me (all |the )?"
        ]
        
        self.specific_patterns = [
            r"how does .* work",
            r"what does .* do",
            r"explain .*",
            r"where is",
            r"find .*"
        ]
    
    def classify(self, query: str) -> str:
        """
        Classify query into:
        - 'comprehensive': needs broad coverage (list all, what are different, etc.)
        - 'specific': needs focused results (how does X work, explain Y, etc.)
        """
        query_lower = query.lower().strip()
        
        # Check comprehensive patterns
        for pattern in self.comprehensive_patterns:
            if re.search(pattern, query_lower):
                return 'comprehensive'
        
        # Check specific patterns
        for pattern in self.specific_patterns:
            if re.search(pattern, query_lower):
                return 'specific'
        
        # Default to specific for safety
        return 'specific'