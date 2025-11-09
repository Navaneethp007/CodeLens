from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .analyzers.direct_analyzer import DirectAnalyzer
from .analyzers.chunked_analyzer import ChunkedAnalyzer
from .storage.session_store import SessionStore
from .chunking.ast_parser import ASTParser
from .chunking.chunk_builder import ChunkBuilder
import os

session_store = SessionStore()
direct_analyzer = DirectAnalyzer()
chunked_analyzer = ChunkedAnalyzer()
ast_parser = ASTParser()
chunk_builder = ChunkBuilder()

FILE_SIZE_THRESHOLD = 3000  # 3KB

@api_view(['POST'])
def upload_file(request):
    """Upload a single Python file for analysis"""
    try:
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        filename = request.data.get('filename', uploaded_file.name)
        file_content = uploaded_file.read().decode('utf-8')
        
        session_store.set_file(file_content, filename)
        
        return Response({
            'status': 'success',
            'filename': filename,
            'file_size': len(file_content),
            'lines': len(file_content.splitlines()),
            'mode': 'single_file',
            'message': f'File uploaded successfully'
        }, status=status.HTTP_200_OK)
        
    except UnicodeDecodeError:
        return Response({'error': 'File encoding not supported'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def upload_codebase(request):
    """Upload a folder path for codebase analysis"""
    try:
        folder_path = request.data.get('path')
        
        if not folder_path:
            return Response({'error': 'No folder path provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not os.path.isdir(folder_path):
            return Response({'error': f'Path is not a directory: {folder_path}'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Count Python files
        python_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        if not python_files:
            return Response({'error': f'No Python files found in {folder_path}'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Store codebase path in session
        session_store.set_codebase(folder_path, python_files)
        
        return Response({
            'status': 'success',
            'folder_path': folder_path,
            'python_files_found': len(python_files),
            'message': f'Found {len(python_files)} Python files. Run /index-codebase/ to start indexing.'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def index_codebase(request):
    """Index all Python files in the uploaded codebase"""
    try:
        if not session_store.has_codebase():
            return Response({'error': 'No codebase uploaded. Upload a codebase first.'}, status=status.HTTP_400_BAD_REQUEST)
        
        folder_path, python_files = session_store.get_codebase()
        
        total_chunks = 0
        indexed_files = 0
        errors = []
        
        # Parse all files
        all_parsed_data = {}
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                parsed_data = ast_parser.parse(file_content, file_path)
                all_parsed_data[file_path] = parsed_data
                indexed_files += 1
                
            except Exception as e:
                errors.append({'file': file_path, 'error': str(e)})
        
        # Build chunks from all files
        all_chunks = chunk_builder.build_chunks_from_codebase(all_parsed_data)
        total_chunks = len(all_chunks)
        
        # Store codebase info
        session_store.set_codebase_indexed(all_chunks, indexed_files, total_chunks, errors)
        
        return Response({
            'status': 'success',
            'folder_path': folder_path,
            'files_indexed': indexed_files,
            'total_chunks': total_chunks,
            'errors': errors if errors else None,
            'message': f'Indexed {indexed_files} files with {total_chunks} code chunks'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def query_code(request):
    """Query the uploaded code (single file or codebase)"""
    try:
        query = request.data.get('query')
        if not query:
            return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Check what mode we're in
        if session_store.has_codebase_indexed():
            # Codebase mode
            all_chunks = session_store.get_codebase_chunks()
            result = chunked_analyzer.analyze(None, None, query, codebase_mode=True, chunks=all_chunks)
            return Response(result, status=status.HTTP_200_OK)
        
        elif session_store.has_file():
            # Single file mode
            file_content = session_store.get_file_content()
            filename = session_store.get_filename()
            
            if len(file_content) < FILE_SIZE_THRESHOLD:
                result = direct_analyzer.analyze(file_content, filename, query)
            else:
                result = chunked_analyzer.analyze(file_content, filename, query)
            
            return Response(result, status=status.HTTP_200_OK)
        
        else:
            return Response({'error': 'No file or codebase uploaded. Upload first.'}, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def file_status(request):
    """Get current session status"""
    if session_store.has_codebase_indexed():
        codebase_info = session_store.get_codebase_status()
        return Response({
            'mode': 'codebase',
            'status': 'ready',
            **codebase_info
        }, status=status.HTTP_200_OK)
    
    elif session_store.has_file():
        return Response({
            'mode': 'single_file',
            'status': 'ready',
            'filename': session_store.get_filename(),
            'file_size': len(session_store.get_file_content()),
            'lines': len(session_store.get_file_content().splitlines())
        }, status=status.HTTP_200_OK)
    
    else:
        return Response({
            'status': 'no_content',
            'message': 'No file or codebase uploaded'
        }, status=status.HTTP_200_OK)

@api_view(['POST'])
def clear_session(request):
    """Clear the current session"""
    try:
        session_store.clear()
        return Response({
            'status': 'success',
            'message': 'Session cleared successfully'
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
