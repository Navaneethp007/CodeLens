import uuid
from typing import List, Dict, Any

class ChunkBuilder:
    """Build chunks with metadata from parsed data"""
    
    def build_chunks(self, parsed_data: Dict[str, Any], file_content: str, filename: str) -> List[Dict[str, Any]]:
        """Create chunks from parsed data"""
        
        chunks = []
        
        # Add imports as one chunk
        if parsed_data['imports']:
            import_code = '\n'.join([imp['code'] for imp in parsed_data['imports']])
            chunks.append({
                'id': str(uuid.uuid4()),
                'type': 'imports',
                'name': 'imports',
                'code': import_code,
                'filename': filename,
                'line_start': 1,
                'line_end': len(parsed_data['imports'])
            })
        
        # Add class chunks
        for cls in parsed_data['classes']:
            chunks.append({
                'id': str(uuid.uuid4()),
                'type': 'class',
                'name': cls['name'],
                'code': cls['code'],
                'filename': filename,
                'line_start': cls['line_start'],
                'line_end': cls['line_end'],
                'docstring': cls['docstring'],
                'methods': cls['methods']
            })
        
        # Add function chunks
        for func in parsed_data['functions']:
            chunks.append({
                'id': str(uuid.uuid4()),
                'type': 'function',
                'name': func['name'],
                'code': func['code'],
                'filename': filename,
                'line_start': func['line_start'],
                'line_end': func['line_end'],
                'docstring': func['docstring']
            })
        
        # Add assignment chunks
        for assign in parsed_data['assignments']:
            chunks.append({
                'id': str(uuid.uuid4()),
                'type': 'assignment',
                'name': assign['name'],
                'code': assign['code'],
                'filename': filename,
                'line_start': assign['line_start'],
                'line_end': assign['line_end']
            })
        
        # Add full file chunk
        chunks.append({
            'id': str(uuid.uuid4()),
            'type': 'file',
            'name': filename,
            'code': file_content,
            'filename': filename,
            'line_start': 1,
            'line_end': len(file_content.splitlines())
        })
        
        return chunks