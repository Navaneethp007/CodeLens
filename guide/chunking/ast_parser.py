import ast
from typing import Dict, List, Any

class ASTParser:
    def parse(self, file_content: str, filename: str) -> Dict[str, Any]:
        parsed_data = {
            'filename': filename,
            'imports': [],
            'classes': [],
            'functions': [],
            'assignments': [],
            'file_content': file_content
        }
        
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()
            
            # FIX: Use iter_child_nodes instead of walk to avoid duplicates
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = {
                        'code': self._extract_code(lines, node),
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno)
                    }
                    parsed_data['imports'].append(import_info)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'code': self._extract_code(lines, node),
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'docstring': ast.get_docstring(node),
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    }
                    parsed_data['classes'].append(class_info)
                
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'code': self._extract_code(lines, node),
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'docstring': ast.get_docstring(node)
                    }
                    parsed_data['functions'].append(func_info)
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assign_info = {
                                'name': target.id,
                                'code': self._extract_code(lines, node),
                                'line_start': node.lineno,
                                'line_end': getattr(node, 'end_lineno', node.lineno)
                            }
                            parsed_data['assignments'].append(assign_info)
            
        except SyntaxError as e:
            parsed_data['error'] = f"Syntax error: {str(e)}"
        
        return parsed_data
    
    def _extract_code(self, lines: List[str], node) -> str:
        start = node.lineno - 1
        end = getattr(node, 'end_lineno', node.lineno) - 1
        return '\n'.join(lines[start:end + 1])