"""
Inference and Evaluation Pipeline for C-Refactoring LLM
Test the trained model on various C code refactoring tasks
"""

import torch
import json
from transformers import AutoTokenizer
import tempfile
import subprocess
import os
from typing import Dict, List, Any
import time

from c_refactoring_llm.architecture import CRefactoringLLM, CRefactorConfig
from c_refactoring_llm.memory import comprehensive_code_analysis

class CRefactoringInference:
    """Inference pipeline for C code refactoring"""
    
    def __init__(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = CRefactoringLLM(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from {checkpoint_path}")
        print(f"🔧 Using device: {self.device}")
    
    def refactor_code(self, 
                     instruction: str, 
                     input_code: str,
                     max_length: int = 512,
                     temperature: float = 0.7,
                     num_beams: int = 4) -> str:
        """
        Refactor C code based on instruction
        
        Args:
            instruction: Task description
            input_code: Input C code to refactor
            max_length: Maximum output length
            temperature: Sampling temperature
            num_beams: Beam search width
            
        Returns:
            Refactored C code
        """
        
        # Prepare input
        input_text = f"{instruction} {input_code}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_code = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        return generated_code
    
    def check_compilation(self, code: str) -> Dict[str, Any]:
        """Check if generated code compiles"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            # Add includes if missing
            if '#include' not in code:
                full_code = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n" + code
            else:
                full_code = code
            
            f.write(full_code)
            f.flush()
            
            try:
                # Compile with GCC
                result = subprocess.run(
                    f"gcc -Wall -Wextra -pedantic -std=c11 {f.name} -o {f.name}.out",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                success = result.returncode == 0
                errors = result.stderr
                warnings = result.stdout
                
                # Clean up
                try:
                    os.unlink(f.name)
                    if os.path.exists(f"{f.name}.out"):
                        os.unlink(f"{f.name}.out")
                except:
                    pass
                
                return {
                    'compiles': success,
                    'errors': errors,
                    'warnings': warnings,
                    'error_count': errors.count('error:'),
                    'warning_count': errors.count('warning:')
                }
                
            except subprocess.TimeoutExpired:
                return {
                    'compiles': False,
                    'errors': "Compilation timeout",
                    'warnings': "",
                    'error_count': 1,
                    'warning_count': 0
                }

def run_evaluation_suite():
    """Run comprehensive evaluation on test cases"""
    
    # Test cases covering different refactoring tasks
    test_cases = [
        {
            'task_type': 'syntax_fix',
            'instruction': 'Fix the syntax errors in this C code',
            'input': 'int main() {\n    int x = 5\n    printf("Value: %d", x)\n    return 0',
            'expected_features': ['semicolons', 'includes', 'proper_main']
        },
        {
            'task_type': 'memory_fix', 
            'instruction': 'Fix the memory leak in this code',
            'input': '#include <stdlib.h>\nint main() {\n    int *arr = malloc(10 * sizeof(int));\n    arr[0] = 5;\n    return 0;\n}',
            'expected_features': ['free_call', 'null_check']
        },
        {
            'task_type': 'security_fix',
            'instruction': 'Fix the buffer overflow vulnerability',
            'input': '#include <stdio.h>\nint main() {\n    char buffer[10];\n    gets(buffer);\n    return 0;\n}',
            'expected_features': ['fgets', 'buffer_size_check']
        },
        {
            'task_type': 'optimization',
            'instruction': 'Optimize this bubble sort implementation',
            'input': 'void bubble_sort(int arr[], int n) {\n    for (int i = 0; i < n-1; i++) {\n        for (int j = 0; j < n-i-1; j++) {\n            if (arr[j] > arr[j+1]) {\n                int temp = arr[j];\n                arr[j] = arr[j+1];\n                arr[j+1] = temp;\n            }\n        }\n    }\n}',
            'expected_features': ['early_termination', 'swap_optimization']
        },
        {
            'task_type': 'convert_to_c',
            'instruction': 'Write a C function to check if a number is prime',
            'input': 'Write a function that takes an integer and returns 1 if it\'s prime, 0 otherwise',
            'expected_features': ['function_definition', 'prime_algorithm', 'return_statements']
        }
    ]
    
    try:
        # Load model (use latest checkpoint)
        checkpoint_path = "checkpoints/c_refactoring_epoch_3.pt"
        if not os.path.exists(checkpoint_path):
            print("❌ No trained model checkpoint found")
            print("💡 Run training first: python run_training.py")
            return
        
        inference = CRefactoringInference(checkpoint_path)
        
        print("\n🧪 RUNNING C-REFACTORING LLM EVALUATION SUITE")
        print("=" * 60)
        
        results = []
        total_compilation_success = 0
        total_processing_time = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}: {test_case['task_type']}")
            print("-" * 40)
            
            start_time = time.time()
            
            # Generate refactored code
            refactored_code = inference.refactor_code(
                instruction=test_case['instruction'],
                input_code=test_case['input'],
                temperature=0.3,
                num_beams=4
            )
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Check compilation
            compilation_result = inference.check_compilation(refactored_code)
            
            # Analyze code quality
            analysis = comprehensive_code_analysis(refactored_code)
            
            # Evaluate results
            compilation_success = compilation_result['compiles']
            if compilation_success:
                total_compilation_success += 1
            
            # Display results
            print(f"✅ Generated Code:")
            print(refactored_code[:200] + ("..." if len(refactored_code) > 200 else ""))
            print(f"\n📊 Evaluation:")
            print(f"  - Compiles: {'✅' if compilation_success else '❌'}")
            print(f"  - Processing Time: {processing_time:.2f}s")
            print(f"  - Memory Issues: {len(analysis['memory_issues'])}")
            print(f"  - Security Issues: {len(analysis['security_issues'])}")
            print(f"  - UB Patterns: {len(analysis['ub_patterns'])}")
            
            if not compilation_success:
                print(f"  - Compilation Errors: {compilation_result['errors'][:100]}...")
            
            results.append({
                'test_case': i,
                'task_type': test_case['task_type'],
                'compiles': compilation_success,
                'processing_time': processing_time,
                'analysis': analysis,
                'generated_code': refactored_code
            })
        
        # Summary statistics
        print(f"\n📈 EVALUATION SUMMARY")
        print("=" * 60)
        print(f"✅ Compilation Success Rate: {total_compilation_success}/{len(test_cases)} ({total_compilation_success/len(test_cases)*100:.1f}%)")
        print(f"⏱️  Average Processing Time: {total_processing_time/len(test_cases):.2f}s")
        print(f"🔧 Total Test Cases: {len(test_cases)}")
        
        # Save detailed results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Detailed results saved to: evaluation_results.json")
        
        # Grade the model
        if total_compilation_success == len(test_cases):
            print("🏆 GRADE: A+ (All code compiles)")
        elif total_compilation_success >= len(test_cases) * 0.8:
            print("🎯 GRADE: A (80%+ compilation success)")
        elif total_compilation_success >= len(test_cases) * 0.6:
            print("👍 GRADE: B (60%+ compilation success)")
        else:
            print("⚠️  GRADE: C (Need improvement)")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def interactive_mode():
    """Interactive mode for testing custom code"""
    checkpoint_path = "checkpoints/c_refactoring_epoch_3.pt"
    
    if not os.path.exists(checkpoint_path):
        print("❌ No trained model checkpoint found")
        return
    
    inference = CRefactoringInference(checkpoint_path)
    
    print("\n🎮 INTERACTIVE C-CODE REFACTORING MODE")
    print("=" * 50)
    print("Enter your C code and refactoring instructions.")
    print("Type 'quit' to exit.")
    
    while True:
        print("\n" + "-" * 30)
        
        instruction = input("📝 Instruction (e.g., 'Fix memory leaks'): ").strip()
        if instruction.lower() == 'quit':
            break
        
        print("📄 Enter your C code (press Ctrl+D or Ctrl+Z when done):")
        code_lines = []
        try:
            while True:
                line = input()
                code_lines.append(line)
        except EOFError:
            pass
        
        input_code = '\n'.join(code_lines)
        
        if not input_code.strip():
            print("❌ No code provided")
            continue
        
        print("\n🔄 Refactoring code...")
        
        try:
            # Refactor
            refactored = inference.refactor_code(instruction, input_code)
            
            # Check compilation
            compilation = inference.check_compilation(refactored)
            
            print(f"\n✨ REFACTORED CODE:")
            print("-" * 30)
            print(refactored)
            print("-" * 30)
            
            print(f"\n📊 ANALYSIS:")
            print(f"Compiles: {'✅' if compilation['compiles'] else '❌'}")
            if not compilation['compiles']:
                print(f"Errors: {compilation['errors']}")
            
        except Exception as e:
            print(f"❌ Error during refactoring: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        run_evaluation_suite()