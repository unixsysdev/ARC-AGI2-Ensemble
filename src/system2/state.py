"""State representation for System 2 controller."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Any

from ..models.task import Task, Grid


class Action(Enum):
    """Actions the System 2 controller can take."""
    FAST_GUESS_CODE = auto()      # Generate Python code candidates quickly
    FAST_GUESS_INSTRUCTION = auto()  # Generate English instruction candidates
    DEEP_REFINE_CODE = auto()     # Revise Python code based on errors
    DEEP_REFINE_INSTRUCTION = auto()  # Revise instructions based on errors
    POOLED_REFINE = auto()        # Synthesize from top instructions
    VLM_VERIFY = auto()           # Filter candidates through visual critic
    HYBRID_EXPLORE = auto()       # Local exploration + remote refinement
    SUBMIT = auto()               # Submit best candidates
    GIVE_UP = auto()              # Move to next task


@dataclass
class ScoredSolution:
    """A solution with its metadata."""
    output: Grid
    score: float  # 0.0 to 1.0
    source: str   # "code" or "instruction"
    artifact: str | None = None  # The code or instruction text
    verified_by_vlm: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class TaskState:
    """State for System 2 decision making."""
    
    # Task info
    task_id: str
    grid_size: int
    num_colors: int
    num_train_pairs: int
    num_test_pairs: int
    
    # Progress
    attempts_used: int = 0
    time_elapsed: float = 0.0
    
    # Best results so far
    best_score: float = 0.0
    best_solutions: list[ScoredSolution] = field(default_factory=list)
    
    # History
    actions_taken: list[Action] = field(default_factory=list)
    last_action: Action | None = None
    last_error: str | None = None
    
    # Budgets
    max_attempts: int = 100
    
    @classmethod
    def from_task(cls, task: Task, max_attempts: int = 100) -> "TaskState":
        """Create initial state from task."""
        return cls(
            task_id=task.task_id,
            grid_size=task.grid_size,
            num_colors=task.num_colors,
            num_train_pairs=len(task.train),
            num_test_pairs=len(task.test),
            max_attempts=max_attempts
        )
    
    @property
    def complexity_score(self) -> float:
        """Estimate task complexity (0.0 to 1.0)."""
        # Factors: grid size, number of colors, training pairs
        size_factor = min(self.grid_size / 30.0, 1.0)
        color_factor = self.num_colors / 10.0
        pair_factor = min(self.num_train_pairs / 5.0, 1.0)
        
        return (size_factor * 0.4 + color_factor * 0.3 + pair_factor * 0.3)
    
    @property
    def budget_remaining(self) -> float:
        """Remaining budget as fraction."""
        return max(0, (self.max_attempts - self.attempts_used) / self.max_attempts)
    
    @property
    def has_perfect_solution(self) -> bool:
        """Check if we have a perfect (1.0) solution."""
        return self.best_score >= 0.999
    
    @property
    def has_good_solution(self) -> bool:
        """Check if we have a good (>0.9) solution."""
        return self.best_score >= 0.9
    
    def add_solution(self, solution: ScoredSolution):
        """Add a new solution to the state."""
        self.attempts_used += 1
        
        if solution.score > self.best_score:
            self.best_score = solution.score
        
        # Keep top solutions sorted by score
        self.best_solutions.append(solution)
        self.best_solutions.sort(key=lambda s: s.score, reverse=True)
        
        # Keep only top 20
        self.best_solutions = self.best_solutions[:20]
    
    def record_action(self, action: Action, error: str | None = None):
        """Record an action taken."""
        self.actions_taken.append(action)
        self.last_action = action
        self.last_error = error
    
    def get_top_solutions(self, n: int = 5, source: str | None = None) -> list[ScoredSolution]:
        """Get top n solutions, optionally filtered by source."""
        solutions = self.best_solutions
        if source:
            solutions = [s for s in solutions if s.source == source]
        return solutions[:n]
