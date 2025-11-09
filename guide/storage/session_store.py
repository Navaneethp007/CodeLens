class SessionStore:
    """Session storage for single file and codebase modes"""
    
    def __init__(self):
        # Single file mode
        self._file_content = ""
        self._filename = ""
        
        # Codebase mode
        self._folder_path = ""
        self._python_files = []
        self._codebase_chunks = []
        self._indexed_files_count = 0
        self._total_chunks_count = 0
        self._indexing_errors = []
        self._is_indexed = False
    
    # ===== SINGLE FILE MODE =====
    
    def set_file(self, file_content: str, filename: str):
        """Store single file in session"""
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
    
    # ===== CODEBASE MODE =====
    
    def set_codebase(self, folder_path: str, python_files: list):
        """Store codebase path and files"""
        self._folder_path = folder_path
        self._python_files = python_files
        self._is_indexed = False
    
    def get_codebase(self) -> tuple:
        """Get codebase path and files list"""
        return self._folder_path, self._python_files
    
    def has_codebase(self) -> bool:
        """Check if codebase exists in session"""
        return bool(self._folder_path)
    
    def set_codebase_indexed(self, chunks: list, indexed_files: int, total_chunks: int, errors: list):
        """Store indexed codebase data"""
        self._codebase_chunks = chunks
        self._indexed_files_count = indexed_files
        self._total_chunks_count = total_chunks
        self._indexing_errors = errors
        self._is_indexed = True
    
    def has_codebase_indexed(self) -> bool:
        """Check if codebase is indexed"""
        return self._is_indexed
    
    def get_codebase_chunks(self) -> list:
        """Get all indexed chunks from codebase"""
        return self._codebase_chunks
    
    def get_codebase_status(self) -> dict:
        """Get codebase status info"""
        return {
            'folder_path': self._folder_path,
            'files_found': len(self._python_files),
            'files_indexed': self._indexed_files_count,
            'total_chunks': self._total_chunks_count,
            'indexing_errors': self._indexing_errors if self._indexing_errors else None
        }
    
    # ===== CLEAR SESSION =====
    
    def clear(self):
        """Clear entire session"""
        # Single file
        self._file_content = ""
        self._filename = ""
        
        # Codebase
        self._folder_path = ""
        self._python_files = []
        self._codebase_chunks = []
        self._indexed_files_count = 0
        self._total_chunks_count = 0
        self._indexing_errors = []
        self._is_indexed = False
