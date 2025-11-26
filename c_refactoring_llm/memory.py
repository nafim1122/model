"""
Memory Safety Analysis for C Code
Advanced memory leak and UB detection
"""

import re
import ast
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum

class MemoryIssueType(Enum):
    MEMORY_LEAK = "memory_leak"
    DOUBLE_FREE = "double_free"
    USE_AFTER_FREE = "use_after_free"
    NULL_POINTER_DEREF = "null_pointer_dereference"
    BUFFER_OVERFLOW = "buffer_overflow"
    UNINITIALIZED_MEMORY = "uninitialized_memory"
    DANGLING_POINTER = "dangling_pointer"

@dataclass
class MemoryAllocation:
    """Track memory allocation"""
    variable: str
    line: int
    function: str
    allocation_type: str  # malloc, calloc, realloc
    size_expr: str
    freed: bool = False
    freed_line: Optional[int] = None

@dataclass
class MemoryIssue:
    """Memory safety issue detection"""
    issue_type: MemoryIssueType
    line: int
    variable: Optional[str]
    description: str
    severity: str  # high, medium, low
    fix_suggestion: str

class MemoryTracker:
    """Track memory allocations and deallocations"""
    
    def __init__(self):
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.current_function = "global"
        self.variable_scopes: Dict[str, Set[str]] = {"global": set()}
        self.issues: List[MemoryIssue] = []
    
    def analyze_memory_safety(self, code: str) -> List[MemoryIssue]:
        """Comprehensive memory safety analysis"""
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            self._analyze_line(line, line_num)
        
        # Check for leaks at end
        self._check_memory_leaks()
        
        return self.issues
    
    def _analyze_line(self, line: str, line_num: int):
        """Analyze single line for memory operations"""
        stripped = line.strip()
        
        # Function entry/exit
        if re.search(r'\w+\s+\w+\s*\([^)]*\)\s*{', stripped):
            func_match = re.search(r'(\w+)\s*\([^)]*\)\s*{', stripped)
            if func_match:
                self.current_function = func_match.group(1)
                self.variable_scopes[self.current_function] = set()
        
        # Memory allocation
        malloc_match = re.search(r'(\w+)\s*=\s*(malloc|calloc|realloc)\s*\(([^)]+)\)', stripped)
        if malloc_match:
            var_name = malloc_match.group(1)
            alloc_type = malloc_match.group(2)
            size_expr = malloc_match.group(3)
            
            allocation = MemoryAllocation(
                variable=var_name,
                line=line_num,
                function=self.current_function,
                allocation_type=alloc_type,
                size_expr=size_expr
            )
            
            self.allocations[var_name] = allocation
            self.variable_scopes[self.current_function].add(var_name)
            
            # Check for NULL check
            self._check_null_check_after_allocation(var_name, line_num)
        
        # Memory deallocation
        free_match = re.search(r'free\s*\(\s*(\w+)\s*\)', stripped)
        if free_match:
            var_name = free_match.group(1)
            self._handle_free(var_name, line_num)
        
        # Pointer usage
        deref_matches = re.findall(r'\*(\w+)', stripped)
        for var_name in deref_matches:
            self._check_pointer_usage(var_name, line_num)
        
        # Array access
        array_matches = re.findall(r'(\w+)\[([^]]+)\]', stripped)
        for var_name, index in array_matches:
            self._check_array_access(var_name, index, line_num)
    
    def _handle_free(self, var_name: str, line_num: int):
        """Handle free() call"""
        if var_name in self.allocations:
            if self.allocations[var_name].freed:
                # Double free
                self.issues.append(MemoryIssue(
                    issue_type=MemoryIssueType.DOUBLE_FREE,
                    line=line_num,
                    variable=var_name,
                    description=f"Double free of variable '{var_name}' (first freed at line {self.allocations[var_name].freed_line})",
                    severity="high",
                    fix_suggestion=f"Remove duplicate free() call or add NULL check"
                ))
            else:
                self.allocations[var_name].freed = True
                self.allocations[var_name].freed_line = line_num
        else:
            # Free without allocation
            self.issues.append(MemoryIssue(
                issue_type=MemoryIssueType.MEMORY_LEAK,
                line=line_num,
                variable=var_name,
                description=f"free() called on untracked variable '{var_name}'",
                severity="medium",
                fix_suggestion="Ensure variable is properly allocated before freeing"
            ))
    
    def _check_pointer_usage(self, var_name: str, line_num: int):
        """Check pointer dereference safety"""
        if var_name in self.allocations:
            if self.allocations[var_name].freed:
                # Use after free
                self.issues.append(MemoryIssue(
                    issue_type=MemoryIssueType.USE_AFTER_FREE,
                    line=line_num,
                    variable=var_name,
                    description=f"Use after free of variable '{var_name}' (freed at line {self.allocations[var_name].freed_line})",
                    severity="high",
                    fix_suggestion="Do not use pointer after calling free(), or set to NULL after free"
                ))
    
    def _check_array_access(self, var_name: str, index: str, line_num: int):
        """Check array bounds safety"""
        # Simple bounds checking patterns
        if re.search(r'\d+', index) and var_name in self.allocations:
            # Extract numeric index
            index_match = re.search(r'(\d+)', index)
            if index_match:
                idx_value = int(index_match.group(1))
                
                # Check against allocation size (basic)
                alloc = self.allocations[var_name]
                if 'sizeof' in alloc.size_expr:
                    # Extract array size from malloc(N * sizeof(...))
                    size_match = re.search(r'(\d+)\s*\*', alloc.size_expr)
                    if size_match:
                        array_size = int(size_match.group(1))
                        if idx_value >= array_size:
                            self.issues.append(MemoryIssue(
                                issue_type=MemoryIssueType.BUFFER_OVERFLOW,
                                line=line_num,
                                variable=var_name,
                                description=f"Array access '{var_name}[{idx_value}]' exceeds allocated size {array_size}",
                                severity="high",
                                fix_suggestion=f"Ensure index is less than {array_size}"
                            ))
    
    def _check_null_check_after_allocation(self, var_name: str, line_num: int):
        """Check if NULL check follows allocation"""
        # This would need lookahead in real implementation
        # For now, just record the need to check
        pass
    
    def _check_memory_leaks(self):
        """Check for memory leaks at end of analysis"""
        for var_name, allocation in self.allocations.items():
            if not allocation.freed:
                self.issues.append(MemoryIssue(
                    issue_type=MemoryIssueType.MEMORY_LEAK,
                    line=allocation.line,
                    variable=var_name,
                    description=f"Memory allocated to '{var_name}' is never freed",
                    severity="medium",
                    fix_suggestion=f"Add free({var_name}) before function return/end"
                ))

class UBDetector:
    """Undefined behavior detection"""
    
    @staticmethod
    def detect_ub_patterns(code: str) -> List[Dict[str, Any]]:
        """Detect undefined behavior patterns"""
        ub_patterns = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Dangerous functions
            dangerous_funcs = {
                'gets': "Use fgets() instead to prevent buffer overflow",
                'strcpy': "Use strncpy() or strlcpy() to prevent buffer overflow",
                'strcat': "Use strncat() or strlcat() to prevent buffer overflow",
                'sprintf': "Use snprintf() to prevent buffer overflow"
            }
            
            for func, suggestion in dangerous_funcs.items():
                if re.search(rf'\b{func}\s*\(', stripped):
                    ub_patterns.append({
                        'type': 'unsafe_function',
                        'line': line_num,
                        'severity': 'high',
                        'description': f"Unsafe function '{func}()' used",
                        'suggestion': suggestion
                    })
            
            # Signed integer overflow
            if re.search(r'\w+\s*\+=\s*\w+', stripped) or re.search(r'\w+\s*\+\s*\w+', stripped):
                ub_patterns.append({
                    'type': 'potential_overflow',
                    'line': line_num,
                    'severity': 'medium',
                    'description': "Potential signed integer overflow",
                    'suggestion': "Check for overflow before arithmetic operations"
                })
            
            # Array to pointer decay without bounds
            if re.search(r'\w+\[\]', stripped) and 'sizeof' not in stripped:
                ub_patterns.append({
                    'type': 'array_decay',
                    'line': line_num,
                    'severity': 'low',
                    'description': "Array parameter loses size information",
                    'suggestion': "Pass array size as separate parameter"
                })
            
            # Uninitialized variables
            if re.search(r'\b(int|char|float|double)\s+\w+\s*;', stripped):
                ub_patterns.append({
                    'type': 'uninitialized_variable',
                    'line': line_num,
                    'severity': 'medium',
                    'description': "Variable declared but not initialized",
                    'suggestion': "Initialize variable at declaration"
                })
        
        return ub_patterns

class SecurityAnalyzer:
    """Security vulnerability detection"""
    
    @staticmethod
    def analyze_security_issues(code: str) -> List[Dict[str, Any]]:
        """Analyze security vulnerabilities"""
        issues = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Format string vulnerabilities
            if re.search(r'printf\s*\(\s*\w+\s*\)', stripped):
                issues.append({
                    'type': 'format_string_vulnerability',
                    'line': line_num,
                    'severity': 'high',
                    'description': "Format string vulnerability in printf",
                    'fix': "Use printf(\"%s\", variable) instead of printf(variable)"
                })
            
            # Command injection
            if re.search(r'system\s*\(', stripped) or re.search(r'exec\w*\s*\(', stripped):
                issues.append({
                    'type': 'command_injection',
                    'line': line_num,
                    'severity': 'high',
                    'description': "Potential command injection vulnerability",
                    'fix': "Validate and sanitize input before system calls"
                })
            
            # Time-of-check to time-of-use (TOCTOU)
            if re.search(r'access\s*\(', stripped) and 'open' in stripped:
                issues.append({
                    'type': 'toctou_vulnerability',
                    'line': line_num,
                    'severity': 'medium',
                    'description': "Potential TOCTOU race condition",
                    'fix': "Use open() directly instead of access() + open()"
                })
            
            # Hardcoded credentials
            if re.search(r'(password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', stripped, re.IGNORECASE):
                issues.append({
                    'type': 'hardcoded_credentials',
                    'line': line_num,
                    'severity': 'high',
                    'description': "Hardcoded credentials detected",
                    'fix': "Use environment variables or config files for credentials"
                })
        
        return issues

def comprehensive_code_analysis(code: str) -> Dict[str, Any]:
    """Comprehensive analysis combining all detectors"""
    
    # Memory safety analysis
    memory_tracker = MemoryTracker()
    memory_issues = memory_tracker.analyze_memory_safety(code)
    
    # UB detection
    ub_patterns = UBDetector.detect_ub_patterns(code)
    
    # Security analysis
    security_issues = SecurityAnalyzer.analyze_security_issues(code)
    
    return {
        'memory_issues': [
            {
                'type': issue.issue_type.value,
                'line': issue.line,
                'variable': issue.variable,
                'description': issue.description,
                'severity': issue.severity,
                'fix_suggestion': issue.fix_suggestion
            }
            for issue in memory_issues
        ],
        'ub_patterns': ub_patterns,
        'security_issues': security_issues,
        'total_issues': len(memory_issues) + len(ub_patterns) + len(security_issues)
    }

if __name__ == "__main__":
    # Test memory analysis
    test_code = """
    #include <stdio.h>
    #include <stdlib.h>
    
    int main() {
        int *arr = malloc(10 * sizeof(int));
        // Missing NULL check
        
        for (int i = 0; i <= 10; i++) {  // Buffer overflow
            arr[i] = i;
        }
        
        free(arr);
        arr[0] = 5;  // Use after free
        
        char *str = malloc(100);
        // Memory leak - never freed
        
        return 0;
    }
    """
    
    analysis = comprehensive_code_analysis(test_code)
    print("Analysis Results:")
    for category, issues in analysis.items():
        if isinstance(issues, list):
            print(f"\n{category}:")
            for issue in issues:
                print(f"  - Line {issue.get('line', 'N/A')}: {issue.get('description', 'N/A')}")
        else:
            print(f"{category}: {issues}")