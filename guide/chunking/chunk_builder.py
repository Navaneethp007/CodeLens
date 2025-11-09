import uuid
from typing import List, Dict, Any

class ChunkBuilder:
    """Build chunks with metadata from parsed data"""
    
    def build_chunks(self, parsed_data: Dict[str, Any], file_content: str, filename: str) -> List[Dict[str, Any]]:
        """Create chunks from parsed data (single file)"""
        return self._build_chunks_internal(parsed_data, file_content, filename)
    
    def build_chunks_from_codebase(self, all_parsed_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks from multiple parsed files (codebase)"""
        all_chunks = []
        
        for file_path, parsed_data in all_parsed_data.items():
            # Get file content if available
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except:
                file_content = ""
            
            chunks = self._build_chunks_internal(parsed_data, file_content, file_path)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _build_chunks_internal(self, parsed_data: Dict[str, Any], file_content: str, file_path: str) -> List[Dict[str, Any]]:
        """Internal method to build chunks"""
        chunks = []
        
        # Add imports as one chunk
        if parsed_data.get('imports'):
            import_code = '\n'.join([imp['code'] for imp in parsed_data['imports']])
            chunks.append({
                'id': str(uuid.uuid4()),
                'type': 'imports',
                'name': 'imports',
                'code': import_code,
                'filename': file_path.split('/')[-1],
                'file_path': file_path,
                'line_start': 1,
                'line_end': len(parsed_data['imports'])
            })
        
        # Add class chunks
        for cls in parsed_data.get('classes', []):
            chunks.append({
                'id': str(uuid.uuid4()),
                'type': 'class',
                'name': cls['name'],
                'code': cls['code'],
                'filename': file_path.split('/')[-1],
                'file_path': file_path,
                'line_start': cls['line_start'],
                'line_end': cls['line_end'],
                'docstring': cls['docstring'],
                'methods': cls['methods']
            })
        
        # Add function chunks
        for func in parsed_data.get('functions', []):
            chunks.append({
                'id': str(uuid.uuid4()),
                'type': 'function',
                'name': func['name'],
                'code': func['code'],
                'filename': file_path.split('/')[-1],
                'file_path': file_path,
                'line_start': func['line_start'],
                'line_end': func['line_end'],
                'docstring': func['docstring']
            })
        
        # Add assignment chunks
        for assign in parsed_data.get('assignments', []):
            chunks.append({
                'id': str(uuid.uuid4()),
                'type': 'assignment',
                'name': assign['name'],
                'code': assign['code'],
                'filename': file_path.split('/')[-1],
                'file_path': file_path,
                'line_start': assign['line_start'],
                'line_end': assign['line_end']
            })
        
        # Add full file chunk
        chunks.append({
            'id': str(uuid.uuid4()),
            'type': 'file',
            'name': file_path.split('/')[-1],
            'code': file_content,
            'filename': file_path.split('/')[-1],
            'file_path': file_path,
            'line_start': 1,
            'line_end': len(file_content.splitlines())
        })
        
        return chunks
