import os
from pathlib import Path
import ast
from typing import Dict, List, Any
import json

class CodeIndexer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.index = {
            'files': {},
            'functions': {},
            'classes': {},
            'imports': {}
        }
        
    def index_file(self, file_path: Path) -> Dict[str, Any]:
        """Index a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            file_info = {
                'path': str(file_path.relative_to(self.root_dir)),
                'functions': [],
                'classes': [],
                'imports': []
            }
            
            # Index all nodes in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line_no': node.lineno,
                        'end_line_no': node.end_lineno,
                        'args': [arg.arg for arg in node.args.args]
                    }
                    file_info['functions'].append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line_no': node.lineno,
                        'end_line_no': node.end_lineno,
                        'bases': [base.id for base in node.bases if isinstance(base, ast.Name)]
                    }
                    file_info['classes'].append(class_info)
                    
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        file_info['imports'].append({
                            'name': name.name,
                            'line_no': node.lineno
                        })
                        
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        file_info['imports'].append({
                            'name': f"{node.module}.{name.name}",
                            'line_no': node.lineno
                        })
            
            return file_info
            
    def index_directory(self) -> None:
        """Index all Python files in the project."""
        for file_path in self.root_dir.rglob('*.py'):
            # Skip virtual environment directories
            if any(part.startswith('.venv') or part == 'venv' for part in file_path.parts):
                continue
                
            try:
                file_info = self.index_file(file_path)
                self.index['files'][str(file_path.relative_to(self.root_dir))] = file_info
                
                # Add to global indexes
                for func in file_info['functions']:
                    self.index['functions'][f"{file_path.stem}:{func['name']}"] = {
                        'file': str(file_path.relative_to(self.root_dir)),
                        **func
                    }
                    
                for cls in file_info['classes']:
                    self.index['classes'][f"{file_path.stem}:{cls['name']}"] = {
                        'file': str(file_path.relative_to(self.root_dir)),
                        **cls
                    }
                    
                for imp in file_info['imports']:
                    if imp['name'] not in self.index['imports']:
                        self.index['imports'][imp['name']] = []
                    self.index['imports'][imp['name']].append({
                        'file': str(file_path.relative_to(self.root_dir)),
                        'line_no': imp['line_no']
                    })
                    
            except Exception as e:
                print(f"Error indexing {file_path}: {str(e)}")
                
    def save_index(self, output_file: str = 'code_index.json') -> None:
        """Save the index to a JSON file."""
        output_path = self.root_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)
            
    def load_index(self, input_file: str = 'code_index.json') -> None:
        """Load an existing index from a JSON file."""
        input_path = self.root_dir / input_file
        if input_path.exists():
            with open(input_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
                
    def search_code(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Search the index for code elements matching the query."""
        results = {
            'functions': [],
            'classes': [],
            'files': []
        }
        
        query = query.lower()
        
        # Search functions
        for func_id, func_info in self.index['functions'].items():
            if query in func_id.lower():
                results['functions'].append(func_info)
                
        # Search classes
        for class_id, class_info in self.index['classes'].items():
            if query in class_id.lower():
                results['classes'].append(class_info)
                
        # Search files
        for file_path, file_info in self.index['files'].items():
            if query in file_path.lower():
                results['files'].append({
                    'path': file_path,
                    **file_info
                })
                
        return results

def create_code_index(project_root: str = None) -> None:
    """Create a code index for the project."""
    if project_root is None:
        project_root = Path(__file__).parent.parent.absolute()
        
    indexer = CodeIndexer(project_root)
    indexer.index_directory()
    indexer.save_index()
    print(f"Code index created at: {project_root}/code_index.json")

if __name__ == '__main__':
    create_code_index() 