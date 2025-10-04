class SessionStore:
    """Simple in-memory storage for current session"""
    
    def __init__(self):
        self._file_content = ""
        self._filename = ""
    
    def set_file(self, file_content: str, filename: str):
        """Store file in session"""
        self._file_content = file_content
        self._filename = filename
    
    def get_file_content(self) -> str:
        """Get stored file content"""
        return self._file_content
    
    def get_filename(self) -> str:
        """Get stored filename"""
        return self._filename
    
    def has_file(self) -> bool:
        """Check if file exists in session"""
        return bool(self._file_content)
    
    def clear(self):
        """Clear session"""
        self._file_content = ""
        self._filename = ""