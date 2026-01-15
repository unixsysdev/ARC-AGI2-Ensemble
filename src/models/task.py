"""Data models for ARC tasks."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from pydantic import BaseModel


# Type alias for grids
Grid = list[list[int]]


class Pair(BaseModel):
    """Input/output pair for training or testing."""
    input: Grid
    output: Grid | None = None  # None for test pairs we're solving


class Task(BaseModel):
    """An ARC task with training and test pairs."""
    task_id: str
    train: list[Pair]
    test: list[Pair]
    
    @classmethod
    def from_json(cls, path: Path) -> "Task":
        """Load task from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        task_id = path.stem
        train = [Pair(input=p["input"], output=p["output"]) for p in data["train"]]
        test = [Pair(input=p["input"], output=p.get("output")) for p in data["test"]]
        
        return cls(task_id=task_id, train=train, test=test)
    
    @property
    def grid_size(self) -> int:
        """Maximum dimension across all grids."""
        all_grids = []
        for pair in self.train + self.test:
            all_grids.append(pair.input)
            if pair.output:
                all_grids.append(pair.output)
        
        return max(
            max(len(g), max(len(row) for row in g) if g else 0)
            for g in all_grids
        ) if all_grids else 0
    
    @property
    def num_colors(self) -> int:
        """Number of unique colors in the task."""
        colors = set()
        for pair in self.train + self.test:
            for row in pair.input:
                colors.update(row)
            if pair.output:
                for row in pair.output:
                    colors.update(row)
        return len(colors)


@dataclass
class ScoredCandidate:
    """A candidate solution with its score."""
    grid: Grid
    score: float  # 0.0 to 1.0 (fraction of correct cells)
    source: str  # e.g., "code", "instruction", "revision"
    code: str | None = None  # Python code if applicable
    instruction: str | None = None  # English instruction if applicable
    error: str | None = None  # Error message if failed


@dataclass
class Attempt:
    """Final attempt for a test pair."""
    task_id: str
    test_index: int
    attempt_1: Grid
    attempt_2: Grid | None = None


def load_tasks(data_dir: Path, split: str = "evaluation") -> list[Task]:
    """Load all tasks from a directory."""
    task_dir = data_dir / split
    tasks = []
    
    for json_file in sorted(task_dir.glob("*.json")):
        tasks.append(Task.from_json(json_file))
    
    return tasks


def grid_to_ascii(grid: Grid) -> str:
    """Convert grid to ASCII representation."""
    color_chars = "0123456789"
    lines = []
    for row in grid:
        lines.append("".join(color_chars[c] for c in row))
    return "\n".join(lines)


def grids_equal(a: Grid, b: Grid) -> bool:
    """Check if two grids are exactly equal."""
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if row_a != row_b:
            return False
    return True


def score_grid(candidate: Grid, expected: Grid) -> float:
    """
    Calculate cell-wise accuracy score.
    Returns fraction of cells that match (0.0 to 1.0).
    """
    if not expected:
        return 0.0
    
    # Handle size mismatch
    if len(candidate) != len(expected):
        return 0.0
    
    total_cells = 0
    correct_cells = 0
    
    for row_c, row_e in zip(candidate, expected):
        if len(row_c) != len(row_e):
            return 0.0
        for cell_c, cell_e in zip(row_c, row_e):
            total_cells += 1
            if cell_c == cell_e:
                correct_cells += 1
    
    return correct_cells / total_cells if total_cells > 0 else 0.0


def generate_grid_diff(expected: Grid, actual: Grid) -> str:
    """Generate ASCII diff showing differences between grids."""
    lines = []
    lines.append("Expected vs Actual:")
    lines.append(f"Size: {len(expected)}x{len(expected[0]) if expected else 0} vs {len(actual)}x{len(actual[0]) if actual else 0}")
    
    max_rows = max(len(expected), len(actual))
    for i in range(max_rows):
        exp_row = expected[i] if i < len(expected) else []
        act_row = actual[i] if i < len(actual) else []
        
        max_cols = max(len(exp_row), len(act_row))
        diff_chars = []
        for j in range(max_cols):
            exp_val = exp_row[j] if j < len(exp_row) else "?"
            act_val = act_row[j] if j < len(act_row) else "?"
            
            if exp_val == act_val:
                diff_chars.append(str(exp_val))
            else:
                diff_chars.append(f"[{exp_val}>{act_val}]")
        
        lines.append(f"Row {i}: {''.join(diff_chars)}")
    
    return "\n".join(lines)
