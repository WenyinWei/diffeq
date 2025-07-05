#!/usr/bin/env python3
"""
Header Documentation Generator for DiffEq Library

This script automatically generates documentation for header files using
a template-based approach. It parses C++ header files and extracts
documentation comments, class definitions, function signatures, and examples.
"""

import os
import re
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

class HeaderParser:
    """Parser for C++ header files to extract documentation elements."""
    
    def __init__(self, header_path: str):
        self.header_path = header_path
        self.content = self._read_file()
        self.lines = self.content.split('\n')
        
    def _read_file(self) -> str:
        """Read the header file content."""
        with open(self.header_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_brief_description(self) -> str:
        """Extract brief description from file header comments."""
        # Look for @brief or similar patterns
        brief_pattern = r'@brief\s+(.+?)(?:\n|$)'
        match = re.search(brief_pattern, self.content)
        if match:
            return match.group(1).strip()
        
        # Look for first comment block
        comment_pattern = r'/\*\*?\s*(.+?)\s*\*/'
        match = re.search(comment_pattern, self.content)
        if match:
            return match.group(1).strip()
        
        return "C++ header file"
    
    def extract_classes(self) -> List[Dict]:
        """Extract class definitions and their documentation."""
        classes = []
        
        # Pattern to match class definitions with documentation
        class_pattern = r'/\*\*?\s*(.+?)\s*\*/\s*template\s*<[^>]*>\s*class\s+(\w+)'
        matches = re.finditer(class_pattern, self.content, re.DOTALL)
        
        for match in matches:
            doc = match.group(1).strip()
            class_name = match.group(2)
            
            # Extract template parameters
            template_match = re.search(r'template\s*<([^>]*)>', match.group(0))
            template_params = []
            if template_match:
                params_str = template_match.group(1)
                # Parse template parameters
                for param in params_str.split(','):
                    param = param.strip()
                    if 'typename' in param or 'class' in param:
                        param_name = param.split()[-1]
                        template_params.append(param_name)
            
            classes.append({
                'name': class_name,
                'documentation': doc,
                'template_params': template_params
            })
        
        return classes
    
    def extract_functions(self) -> List[Dict]:
        """Extract function declarations and their documentation."""
        functions = []
        
        # Pattern to match function declarations with documentation
        func_pattern = r'/\*\*?\s*(.+?)\s*\*/\s*([^;]+);'
        matches = re.finditer(func_pattern, self.content, re.DOTALL)
        
        for match in matches:
            doc = match.group(1).strip()
            signature = match.group(2).strip()
            
            # Extract function name
            name_match = re.search(r'(\w+)\s*\([^)]*\)', signature)
            if name_match:
                func_name = name_match.group(1)
                
                # Extract parameters
                params_match = re.search(r'\(([^)]*)\)', signature)
                params = []
                if params_match:
                    params_str = params_match.group(1)
                    for param in params_str.split(','):
                        param = param.strip()
                        if param and param != 'void':
                            param_name = param.split()[-1]
                            params.append(param_name)
                
                functions.append({
                    'name': func_name,
                    'documentation': doc,
                    'signature': signature,
                    'parameters': params
                })
        
        return functions
    
    def extract_concepts(self) -> List[Dict]:
        """Extract C++20 concept definitions."""
        concepts = []
        
        # Pattern to match concept definitions
        concept_pattern = r'/\*\*?\s*(.+?)\s*\*/\s*template\s*<[^>]*>\s*concept\s+(\w+)'
        matches = re.finditer(concept_pattern, self.content, re.DOTALL)
        
        for match in matches:
            doc = match.group(1).strip()
            concept_name = match.group(2)
            
            concepts.append({
                'name': concept_name,
                'documentation': doc
            })
        
        return concepts
    
    def extract_examples(self) -> List[str]:
        """Extract code examples from comments."""
        examples = []
        
        # Pattern to match code examples in comments
        example_pattern = r'```cpp\s*(.+?)\s*```'
        matches = re.finditer(example_pattern, self.content, re.DOTALL)
        
        for match in matches:
            examples.append(match.group(1).strip())
        
        return examples

class DocumentationGenerator:
    """Generator for header documentation using templates."""
    
    def __init__(self, template_path: str):
        self.template_path = template_path
        self.template = self._read_template()
    
    def _read_template(self) -> str:
        """Read the documentation template."""
        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_documentation(self, header_path: str, output_path: str):
        """Generate documentation for a header file."""
        parser = HeaderParser(header_path)
        
        # Extract information from header
        header_name = os.path.basename(header_path)
        brief_desc = parser.extract_brief_description()
        classes = parser.extract_classes()
        functions = parser.extract_functions()
        concepts = parser.extract_concepts()
        examples = parser.extract_examples()
        
        # Generate documentation content
        content = self.template
        
        # Replace template placeholders
        content = content.replace('{Header Name}', header_name)
        content = content.replace('{Brief Description}', brief_desc)
        content = content.replace('{header_path}', header_path)
        
        # Generate classes section
        classes_section = self._generate_classes_section(classes)
        content = content.replace('{Classes Section}', classes_section)
        
        # Generate functions section
        functions_section = self._generate_functions_section(functions)
        content = content.replace('{Functions Section}', functions_section)
        
        # Generate concepts section
        concepts_section = self._generate_concepts_section(concepts)
        content = content.replace('{Concepts Section}', concepts_section)
        
        # Generate examples section
        examples_section = self._generate_examples_section(examples)
        content = content.replace('{Examples Section}', examples_section)
        
        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Generated documentation: {output_path}")
    
    def _generate_classes_section(self, classes: List[Dict]) -> str:
        """Generate the classes documentation section."""
        if not classes:
            return "No classes defined in this header."
        
        section = ""
        for cls in classes:
            section += f"#### `{cls['name']}`\n\n"
            section += f"**Description**: {cls['documentation']}\n\n"
            
            if cls['template_params']:
                section += "**Template Parameters**:\n"
                for param in cls['template_params']:
                    section += f"- `{param}`: Description of the template parameter\n"
                section += "\n"
            
            section += "**Public Methods**:\n"
            section += "- `method_name(params)`: Description of the method\n"
            section += "- `method_name(params)`: Description of the method\n\n"
            
            section += "**Example**:\n"
            section += "```cpp\n"
            section += f"// Example usage of {cls['name']}\n"
            section += f"{cls['name']} obj;\n"
            section += "// ... usage example\n"
            section += "```\n\n"
        
        return section
    
    def _generate_functions_section(self, functions: List[Dict]) -> str:
        """Generate the functions documentation section."""
        if not functions:
            return "No functions defined in this header."
        
        section = ""
        for func in functions:
            section += f"#### `{func['name']}`\n\n"
            section += f"**Description**: {func['documentation']}\n\n"
            
            if func['parameters']:
                section += "**Parameters**:\n"
                for param in func['parameters']:
                    section += f"- `{param}`: Description of the parameter\n"
                section += "\n"
            
            section += "**Returns**: Description of the return value\n\n"
            
            section += "**Example**:\n"
            section += "```cpp\n"
            section += f"// Example usage of {func['name']}\n"
            section += f"{func['name']}();\n"
            section += "```\n\n"
        
        return section
    
    def _generate_concepts_section(self, concepts: List[Dict]) -> str:
        """Generate the concepts documentation section."""
        if not concepts:
            return "No concepts defined in this header."
        
        section = ""
        for concept in concepts:
            section += f"#### `{concept['name']}`\n\n"
            section += f"**Description**: {concept['documentation']}\n\n"
            
            section += "**Requirements**:\n"
            section += "- `requirement`: Description of the requirement\n"
            section += "- `requirement`: Description of the requirement\n\n"
            
            section += "**Example**:\n"
            section += "```cpp\n"
            section += f"// Example of a type that satisfies {concept['name']}\n"
            section += "template<typename T>\n"
            section += f"requires {concept['name']}<T>\n"
            section += "void function(T& obj) {\n"
            section += "    // ... implementation\n"
            section += "}\n"
            section += "```\n\n"
        
        return section
    
    def _generate_examples_section(self, examples: List[str]) -> str:
        """Generate the examples documentation section."""
        if not examples:
            return "No examples found in this header."
        
        section = "### Examples\n\n"
        
        for i, example in enumerate(examples, 1):
            section += f"#### Example {i}\n\n"
            section += "```cpp\n"
            section += example
            section += "\n```\n\n"
        
        return section

def main():
    """Main function for the documentation generator."""
    parser = argparse.ArgumentParser(description='Generate documentation for C++ header files')
    parser.add_argument('--input', '-i', required=True, help='Input header file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory for generated documentation')
    parser.add_argument('--template', '-t', default='docs/templates/header_template.md', 
                       help='Template file for documentation generation')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='Process directories recursively')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DocumentationGenerator(args.template)
    
    # Process input
    if os.path.isfile(args.input):
        # Single file
        output_path = os.path.join(args.output, os.path.basename(args.input).replace('.hpp', '.md'))
        generator.generate_documentation(args.input, output_path)
    elif os.path.isdir(args.input):
        # Directory
        pattern = '**/*.hpp' if args.recursive else '*.hpp'
        header_files = glob.glob(os.path.join(args.input, pattern), recursive=args.recursive)
        
        for header_file in header_files:
            # Calculate relative path for output
            rel_path = os.path.relpath(header_file, args.input)
            output_path = os.path.join(args.output, rel_path.replace('.hpp', '.md'))
            generator.generate_documentation(header_file, output_path)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    print("Documentation generation completed!")
    return 0

if __name__ == '__main__':
    exit(main()) 