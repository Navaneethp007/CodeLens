from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .analyzers.direct_analyzer import DirectAnalyzer
from .analyzers.chunked_analyzer import ChunkedAnalyzer
from .storage.session_store import SessionStore

# Initialize components
session_store = SessionStore()
direct_analyzer = DirectAnalyzer()
chunked_analyzer = ChunkedAnalyzer()

FILE_SIZE_THRESHOLD = 3000

@api_view(['POST'])
def upload_file(request):
    """Upload a Python file for analysis"""
    try:
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        filename = request.data.get('filename', uploaded_file.name)
        
        # Read file content
        file_content = uploaded_file.read().decode('utf-8')
        
        # Store in session
        session_store.set_file(file_content, filename)
        
        return Response({
            'status': 'success',
            'filename': filename,
            'file_size': len(file_content),
            'lines': len(file_content.splitlines()),
            'analysis_method': 'direct' if len(file_content) < FILE_SIZE_THRESHOLD else 'chunked',
            'message': f'File uploaded successfully'
        }, status=status.HTTP_200_OK)
        
    except UnicodeDecodeError:
        return Response({'error': 'File encoding not supported'}, status=status.HTTP_400_BAD_REQUEST)
        return Response({'error': 'File encoding not supported'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def query_code(request):
    """Query the uploaded code file"""
    try:
        # Check if file exists
        if not session_store.has_file():
            return Response({'error': 'No file uploaded. Upload a file first.'}, status=status.HTTP_400_BAD_REQUEST)
        
        query = request.data.get('query')
        if not query:
            return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        file_content = session_store.get_file_content()
        filename = session_store.get_filename()
        
        # Choose analyzer based on file size
        if len(file_content) < FILE_SIZE_THRESHOLD:
            result = direct_analyzer.analyze(file_content, filename, query)
        else:
            result = chunked_analyzer.analyze(file_content, filename, query)
        
        return Response(result, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

@api_view(['GET'])
def file_status(request):
    """Get current file status"""
    if session_store.has_file():
        return Response({
            'status': 'ready',
            'filename': session_store.get_filename(),
            'file_size': len(session_store.get_file_content()),
            'lines': len(session_store.get_file_content().splitlines())
        }, status=status.HTTP_200_OK)
    else:
        return Response({
            'status': 'no_file',
            'message': 'No file currently uploaded'

        }, status=status.HTTP_200_OK)
