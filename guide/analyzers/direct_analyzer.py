import ollama

class DirectAnalyzer:
    """Direct analysis for small files - send entire file to LLM"""
    
    def __init__(self):
        self.ollama_client = ollama.Client()
    
    def analyze(self, file_content: str, filename: str, query: str) -> dict:
        """Analyze code by sending full file to LLM"""
        try:
            prompt = self._build_prompt(file_content, filename, query)
            
            response = self.ollama_client.generate(
                model='codellama',
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 500
                }
            )
            
            return {
                'status': 'success',
                'query': query,
                'answer': response['response'].strip(),
                'method': 'direct_analysis',
                'filename': filename
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e)
            }
    
    def _build_prompt(self, file_content: str, filename: str, query: str) -> str:
        """Build prompt for LLM"""
        return f"""You are a code analysis expert. Answer the user's question about this Python code.

Filename: {filename}

Code:
```python
{file_content}
Question: {query}   
Instructions:
- Analyze the entire code file carefully
- Provide a complete and accurate answer
- If asked about "all" or "different" items, list EVERY one you find
- Be specific with names, line numbers, and details
- Only include information that exists in the code
- Give one comprehensive answer

Answer:"""
