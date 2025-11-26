"""
AST Utilities for C Code Structural Analysis
Tree-sitter based AST feature extraction
"""

import tree_sitter
from tree_sitter import Node
from typing import Dict, List, Tuple, Set
import re

class ASTFeatureExtractor:
    """Extract structural features from C AST"""
    
    def __init__(self):
        # C node types mapping
        self.node_type_map = {
            'translation_unit': 1, 'function_definition': 2, 'declaration': 3,
            'compound_statement': 4, 'if_statement': 5, 'while_statement': 6,
            'for_statement': 7, 'do_statement': 8, 'switch_statement': 9,
            'case_statement': 10, 'break_statement': 11, 'continue_statement': 12,
            'return_statement': 13, 'expression_statement': 14, 'assignment_expression': 15,
            'binary_expression': 16, 'unary_expression': 17, 'call_expression': 18,
            'identifier': 19, 'number_literal': 20, 'string_literal': 21,
            'character_literal': 22, 'primitive_type': 23, 'pointer_declarator': 24,
            'array_declarator': 25, 'parameter_list': 26, 'argument_list': 27,
            'field_declaration': 28, 'struct_specifier': 29, 'union_specifier': 30,
            'enum_specifier': 31, 'typedef_declaration': 32, 'include_directive': 33,
            'define_directive': 34, 'condition_clause': 35, 'update_expression': 36,
            'field_expression': 37, 'subscript_expression': 38, 'cast_expression': 39,
            'sizeof_expression': 40, 'conditional_expression': 41, 'parenthesized_expression': 42
        }
        
        # Control flow node types
        self.control_flow_map = {
            'if_statement': 1, 'while_statement': 2, 'for_statement': 3,
            'do_statement': 4, 'switch_statement': 5, 'case_statement': 6,
            'break_statement': 7, 'continue_statement': 8, 'return_statement': 9,
            'goto_statement': 10, 'label_statement': 11, 'function_definition': 12
        }
        
        # Data flow patterns
        self.data_flow_map = {
            'declaration': 1, 'assignment_expression': 2, 'call_expression': 3,
            'identifier': 4, 'pointer_declarator': 5, 'array_declarator': 6,
            'field_expression': 7, 'subscript_expression': 8, 'address_expression': 9,
            'dereference_expression': 10, 'parameter_declaration': 11
        }

def extract_structural_features(root_node: Node, source_code: str) -> Dict[str, List[int]]:
    """
    Extract structural features from C AST
    
    Args:
        root_node: Tree-sitter root node
        source_code: Original source code
        
    Returns:
        Dictionary with feature lists
    """
    extractor = ASTFeatureExtractor()
    
    # Initialize feature lists
    features = {
        'node_types': [],
        'depths': [],
        'siblings': [],
        'control_flow': [],
        'data_flow': []
    }
    
    # Track variables for data flow analysis
    variable_scopes = {}
    current_scope_depth = 0
    
    def traverse(node: Node, depth: int, sibling_index: int, scope_vars: Set[str]):
        """Recursively traverse AST and extract features"""
        nonlocal current_scope_depth, variable_scopes
        
        # Node type encoding
        node_type = extractor.node_type_map.get(node.type, 0)
        features['node_types'].append(node_type)
        
        # Depth in AST
        features['depths'].append(min(depth, 49))  # Max depth 49
        
        # Sibling position
        features['siblings'].append(min(sibling_index, 255))  # Max siblings 255
        
        # Control flow type
        control_type = extractor.control_flow_map.get(node.type, 0)
        features['control_flow'].append(control_type)
        
        # Data flow analysis
        data_flow_type = 0
        if node.type in extractor.data_flow_map:
            data_flow_type = extractor.data_flow_map[node.type]
            
            # Track variable declarations and uses
            if node.type == 'identifier':
                var_name = source_code[node.start_byte:node.end_byte]
                if var_name in scope_vars:
                    data_flow_type = 4  # Variable use
                else:
                    data_flow_type = 11  # Potential new variable
            elif node.type == 'declaration':
                # Extract declared variable names
                for child in node.children:
                    if child.type == 'init_declarator' or child.type == 'declarator':
                        var_node = find_identifier_in_declarator(child)
                        if var_node:
                            var_name = source_code[var_node.start_byte:var_node.end_byte]
                            scope_vars.add(var_name)
        
        features['data_flow'].append(data_flow_type)
        
        # Handle scope changes
        new_scope_vars = scope_vars.copy()
        if node.type in ['compound_statement', 'function_definition']:
            current_scope_depth += 1
        
        # Recursively process children
        for i, child in enumerate(node.children):
            traverse(child, depth + 1, i, new_scope_vars)
        
        if node.type in ['compound_statement', 'function_definition']:
            current_scope_depth -= 1
    
    # Start traversal
    traverse(root_node, 0, 0, set())
    
    return features

def find_identifier_in_declarator(node: Node) -> Node:
    """Find the identifier node in a declarator"""
    if node.type == 'identifier':
        return node
    
    for child in node.children:
        result = find_identifier_in_declarator(child)
        if result:
            return result
    
    return None

def analyze_memory_patterns(code: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Analyze memory-related patterns in C code
    
    Returns:
        Dictionary with locations of memory operations
    """
    patterns = {
        'malloc_calls': [],
        'free_calls': [],
        'pointer_arithmetic': [],
        'array_access': [],
        'null_checks': [],
        'buffer_operations': []
    }
    
    lines = code.split('\n')
    
    for line_num, line in enumerate(lines):
        # malloc/calloc/realloc patterns
        if re.search(r'\b(malloc|calloc|realloc)\s*\(', line):
            patterns['malloc_calls'].append((line_num, line.find('malloc')))
        
        # free patterns
        if re.search(r'\bfree\s*\(', line):
            patterns['free_calls'].append((line_num, line.find('free')))
        
        # Pointer arithmetic
        if re.search(r'\w+\s*[\+\-]\s*\d+', line) and '*' in line:
            patterns['pointer_arithmetic'].append((line_num, 0))
        
        # Array access
        if re.search(r'\w+\s*\[\s*\w*\s*\]', line):
            patterns['array_access'].append((line_num, 0))
        
        # NULL checks
        if re.search(r'\b(NULL|nullptr)\b', line):
            patterns['null_checks'].append((line_num, 0))
        
        # Buffer operations (strcpy, strcat, etc.)
        if re.search(r'\b(strcpy|strcat|sprintf|gets)\s*\(', line):
            patterns['buffer_operations'].append((line_num, 0))
    
    return patterns

def detect_undefined_behavior_patterns(code: str) -> List[Dict[str, any]]:
    """
    Detect potential undefined behavior patterns
    
    Returns:
        List of UB pattern detections
    """
    ub_patterns = []
    lines = code.split('\n')
    
    for line_num, line in enumerate(lines):
        # Buffer overflow potential
        if re.search(r'gets\s*\(', line):
            ub_patterns.append({
                'type': 'buffer_overflow',
                'line': line_num + 1,
                'severity': 'high',
                'description': 'gets() is unsafe and can cause buffer overflow'
            })
        
        # Uninitialized variable usage
        if re.search(r'\bint\s+\w+\s*;', line) and not re.search(r'=', line):
            ub_patterns.append({
                'type': 'uninitialized_variable',
                'line': line_num + 1,
                'severity': 'medium',
                'description': 'Variable declared but not initialized'
            })
        
        # Double free potential
        if line.count('free(') > 1:
            ub_patterns.append({
                'type': 'double_free',
                'line': line_num + 1,
                'severity': 'high',
                'description': 'Multiple free() calls in same line'
            })
        
        # Null pointer dereference
        if re.search(r'\*\w+', line) and 'if' not in line and 'NULL' not in line:
            ub_patterns.append({
                'type': 'potential_null_deref',
                'line': line_num + 1,
                'severity': 'medium',
                'description': 'Pointer dereferenced without NULL check'
            })
        
        # Array bounds
        if re.search(r'\w+\[\w*\]', line) and 'sizeof' not in line:
            ub_patterns.append({
                'type': 'potential_bounds_violation',
                'line': line_num + 1,
                'severity': 'medium',
                'description': 'Array access without bounds checking'
            })
    
    return ub_patterns

def analyze_control_flow_complexity(code: str) -> Dict[str, int]:
    """
    Analyze control flow complexity metrics
    
    Returns:
        Dictionary with complexity metrics
    """
    metrics = {
        'cyclomatic_complexity': 1,  # Start with 1
        'nesting_depth': 0,
        'function_count': 0,
        'loop_count': 0,
        'conditional_count': 0
    }
    
    lines = code.split('\n')
    current_depth = 0
    max_depth = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Count control structures
        if re.search(r'\b(if|while|for|switch)\s*\(', stripped):
            metrics['cyclomatic_complexity'] += 1
            if 'if' in stripped:
                metrics['conditional_count'] += 1
            elif any(keyword in stripped for keyword in ['while', 'for']):
                metrics['loop_count'] += 1
        
        # Count functions
        if re.search(r'\w+\s+\w+\s*\([^)]*\)\s*{', stripped):
            metrics['function_count'] += 1
        
        # Track nesting depth
        current_depth += stripped.count('{')
        max_depth = max(max_depth, current_depth)
        current_depth -= stripped.count('}')
    
    metrics['nesting_depth'] = max_depth
    return metrics

def extract_function_signatures(code: str) -> List[Dict[str, str]]:
    """
    Extract function signatures from C code
    
    Returns:
        List of function signature information
    """
    signatures = []
    lines = code.split('\n')
    
    for line_num, line in enumerate(lines):
        # Match function definitions
        func_match = re.search(
            r'(\w+\s+)?(\w+)\s+(\w+)\s*\(([^)]*)\)\s*{',
            line.strip()
        )
        
        if func_match:
            storage_class = func_match.group(1) or ''
            return_type = func_match.group(2)
            function_name = func_match.group(3)
            parameters = func_match.group(4)
            
            signatures.append({
                'name': function_name,
                'return_type': return_type.strip(),
                'parameters': parameters.strip(),
                'storage_class': storage_class.strip(),
                'line': line_num + 1
            })
    
    return signatures

if __name__ == "__main__":
    # Test AST feature extraction
    sample_code = """
    #include <stdio.h>
    #include <stdlib.h>
    
    int main() {
        int *arr = malloc(10 * sizeof(int));
        if (arr == NULL) {
            return 1;
        }
        
        for (int i = 0; i < 10; i++) {
            arr[i] = i * 2;
        }
        
        free(arr);
        return 0;
    }
    """
    
    # Analyze memory patterns
    memory_patterns = analyze_memory_patterns(sample_code)
    print("Memory patterns:", memory_patterns)
    
    # Detect UB patterns
    ub_patterns = detect_undefined_behavior_patterns(sample_code)
    print("UB patterns:", ub_patterns)
    
    # Analyze complexity
    complexity = analyze_control_flow_complexity(sample_code)
    print("Complexity metrics:", complexity)
    
    # Extract signatures
    signatures = extract_function_signatures(sample_code)
    print("Function signatures:", signatures)