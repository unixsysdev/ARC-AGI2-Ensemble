"""Grid diff utilities."""

from __future__ import annotations
from ..models.task import Grid


def generate_ascii_diff(expected: Grid, actual: Grid) -> str:
    """
    Generate ASCII diff showing differences between grids.
    
    Format:
    - Matching cells show the value
    - Mismatches show [expected>actual]
    """
    lines = []
    
    # Size header
    exp_h, exp_w = len(expected), len(expected[0]) if expected else 0
    act_h, act_w = len(actual), len(actual[0]) if actual else 0
    
    if (exp_h, exp_w) != (act_h, act_w):
        lines.append(f"SIZE MISMATCH: Expected {exp_h}x{exp_w}, Got {act_h}x{act_w}")
    
    max_rows = max(exp_h, act_h)
    
    for i in range(max_rows):
        exp_row = expected[i] if i < exp_h else []
        act_row = actual[i] if i < act_h else []
        
        max_cols = max(len(exp_row), len(act_row))
        diff_chars = []
        
        for j in range(max_cols):
            exp_val = exp_row[j] if j < len(exp_row) else "?"
            act_val = act_row[j] if j < len(act_row) else "?"
            
            if exp_val == act_val:
                diff_chars.append(str(exp_val))
            else:
                diff_chars.append(f"[{exp_val}>{act_val}]")
        
        lines.append(f"Row {i:2d}: {''.join(diff_chars)}")
    
    return "\n".join(lines)


def count_differences(expected: Grid, actual: Grid) -> tuple[int, int]:
    """
    Count cell differences between grids.
    
    Returns:
        (num_different, total_cells)
    """
    if len(expected) != len(actual):
        # Size mismatch - count all as different
        total = sum(len(row) for row in expected)
        return total, total
    
    different = 0
    total = 0
    
    for exp_row, act_row in zip(expected, actual):
        if len(exp_row) != len(act_row):
            different += max(len(exp_row), len(act_row))
            total += max(len(exp_row), len(act_row))
        else:
            for exp_val, act_val in zip(exp_row, act_row):
                total += 1
                if exp_val != act_val:
                    different += 1
    
    return different, total
