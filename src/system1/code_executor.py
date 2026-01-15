"""Safe code execution sandbox for ARC tasks."""

from __future__ import annotations
import signal
import traceback
from contextlib import contextmanager
from typing import Any
import numpy as np

from ..models.task import Grid


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


class ExecutionError(Exception):
    """Raised when code execution fails."""
    pass


@contextmanager
def timeout(seconds: float):
    """Context manager for timeout (Unix only)."""
    def handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds}s")
    
    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class CodeExecutor:
    """Safe executor for Python code."""
    
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds
        
        # Safe namespace with limited builtins
        self.safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'frozenset': frozenset,
            'int': int,
            'isinstance': isinstance,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'print': print,  # Allow for debugging
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None,
            # Additional needed builtins
            'slice': slice,
            'iter': iter,
            'next': next,
            'pow': pow,
            'divmod': divmod,
            'abs': abs,
            'ord': ord,
            'chr': chr,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'callable': callable,
            'type': type,
            'object': object,
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'IndexError': IndexError,
            'KeyError': KeyError,
        }
    
    def _strip_imports(self, code: str) -> str:
        """Remove import statements since we inject numpy directly."""
        lines = code.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip import lines - numpy is already provided
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)
    
    def execute(self, code: str, input_grid: Grid) -> tuple[Grid | None, str | None]:
        """
        Execute code on input grid.
        
        Returns:
            (result_grid, error_message) - one will be None
        """
        # Strip import statements - numpy is already in namespace
        code = self._strip_imports(code)
        
        # Create safe namespace with commonly used libraries
        from collections import deque, Counter, defaultdict
        try:
            from scipy import ndimage
        except ImportError:
            ndimage = None
        
        namespace = {
            '__builtins__': self.safe_builtins,
            'np': np,
            'numpy': np,
            # Collections
            'deque': deque,
            'Counter': Counter,
            'defaultdict': defaultdict,
            # Scipy (if available)
            'ndimage': ndimage,
        }
        
        try:
            with timeout(self.timeout):
                # Compile and execute code
                exec(compile(code, '<string>', 'exec'), namespace)
                
                # Get transform function
                if 'transform' not in namespace:
                    return None, "No 'transform' function defined"
                
                transform = namespace['transform']
                
                # Convert grid to numpy for execution, then back
                result = transform(input_grid)
                
                # Validate result
                if not isinstance(result, (list, np.ndarray)):
                    return None, f"transform() returned {type(result)}, expected list"
                
                # Convert numpy array to list if needed
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                
                # Validate structure
                if not result:
                    return None, "transform() returned empty grid"
                
                if not all(isinstance(row, list) for row in result):
                    return None, "transform() must return list of lists"
                
                # Validate values are 0-9
                for row in result:
                    for cell in row:
                        if not isinstance(cell, (int, np.integer)):
                            return None, f"Grid cell is {type(cell)}, expected int"
                        if not 0 <= int(cell) <= 9:
                            return None, f"Grid cell value {cell} not in 0-9"
                
                # Convert all to Python int
                result = [[int(cell) for cell in row] for row in result]
                
                return result, None
                
        except TimeoutError as e:
            return None, str(e)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            return None, error_msg
    
    def test_on_training(self, code: str, train_pairs: list) -> tuple[float, list[str]]:
        """
        Test code on all training pairs.
        
        Returns:
            (score, list of errors) - score is 0.0 to 1.0 (% pairs correct)
        """
        from ..models.task import grids_equal
        
        correct = 0
        errors = []
        
        for i, pair in enumerate(train_pairs):
            result, error = self.execute(code, pair.input)
            
            if error:
                errors.append(f"Pair {i+1}: {error}")
            elif result is None:
                errors.append(f"Pair {i+1}: No result returned")
            elif not grids_equal(result, pair.output):
                errors.append(f"Pair {i+1}: Output mismatch")
            else:
                correct += 1
        
        score = correct / len(train_pairs) if train_pairs else 0.0
        return score, errors
