"""Heuristic policy for System 2 controller."""

from __future__ import annotations
import logging

from .state import TaskState, Action

logger = logging.getLogger(__name__)


class HeuristicPolicy:
    """
    Rule-based policy for deciding System 2 actions.
    
    This is the "Manager" that decides strategy without solving puzzles directly.
    Future work: Replace with RL-trained policy.
    """
    
    def __init__(self, hybrid_mode: bool = False):
        self.hybrid_mode = hybrid_mode
    
    def decide(self, state: TaskState) -> Action:
        """
        Decide next action based on current state.
        
        Returns the action to take next.
        """
        # Rule 1: If we have a perfect solution, submit
        if state.has_perfect_solution:
            logger.info(f"Task {state.task_id}: Perfect solution found, submitting")
            return Action.SUBMIT
        
        # Rule 2: If we've exhausted budget, submit best or give up
        if state.attempts_used >= state.max_attempts:
            if state.best_score > 0:
                logger.info(f"Task {state.task_id}: Budget exhausted, submitting best ({state.best_score:.2f})")
                return Action.SUBMIT
            else:
                logger.info(f"Task {state.task_id}: Budget exhausted with no solutions, giving up")
                return Action.GIVE_UP
        
        # Rule 3: HYBRID MODE - use local+remote exploration
        if self.hybrid_mode and state.attempts_used < 5:
            # Use hybrid in early phase (generates many candidates)
            return Action.HYBRID_EXPLORE
        
        # Rule 4: Initial exploration phase (first 30 attempts)
        if state.attempts_used < 30:
            # Alternate between code and instruction generation for diversity
            if state.attempts_used % 2 == 0:
                return Action.FAST_GUESS_CODE
            else:
                return Action.FAST_GUESS_INSTRUCTION
        
        # Rule 4: If we have good solutions (>0.9), try VLM verification
        if state.has_good_solution and not self._has_vlm_verified(state):
            logger.info(f"Task {state.task_id}: Good solution found, verifying with VLM")
            return Action.VLM_VERIFY
        
        # Rule 5: Refinement phase based on what's working better
        code_score = self._best_score_by_source(state, "code")
        instruction_score = self._best_score_by_source(state, "instruction")
        
        # If code is doing better, refine code
        if code_score >= instruction_score and code_score > 0:
            # Check if we should do pooled revision
            if state.attempts_used % 15 == 0 and len(state.get_top_solutions(source="instruction")) >= 3:
                return Action.POOLED_REFINE
            return Action.DEEP_REFINE_CODE
        
        # If instructions are doing better, refine instructions
        if instruction_score > 0:
            if state.attempts_used % 15 == 0 and len(state.get_top_solutions(source="instruction")) >= 3:
                return Action.POOLED_REFINE
            return Action.DEEP_REFINE_INSTRUCTION
        
        # Rule 6: If nothing is working, keep exploring
        if state.attempts_used < 60:
            # Try more instructions (they work better for abstract patterns)
            return Action.FAST_GUESS_INSTRUCTION
        
        # Rule 7: Late game - try pooled revision if we have candidates
        if len(state.best_solutions) >= 3:
            return Action.POOLED_REFINE
        
        # Last resort: keep generating
        return Action.FAST_GUESS_INSTRUCTION
    
    def _best_score_by_source(self, state: TaskState, source: str) -> float:
        """Get best score from a specific source."""
        solutions = [s for s in state.best_solutions if s.source == source]
        return max((s.score for s in solutions), default=0.0)
    
    def _has_vlm_verified(self, state: TaskState) -> bool:
        """Check if VLM verification has been done."""
        return Action.VLM_VERIFY in state.actions_taken
    
    def explain(self, state: TaskState, action: Action) -> str:
        """Explain why an action was chosen (for logging/debugging)."""
        explanations = {
            Action.SUBMIT: f"Best score {state.best_score:.2f} is acceptable",
            Action.GIVE_UP: "No viable solutions and budget exhausted",
            Action.FAST_GUESS_CODE: f"Exploring with code generation (attempt {state.attempts_used})",
            Action.FAST_GUESS_INSTRUCTION: f"Exploring with instruction generation (attempt {state.attempts_used})",
            Action.DEEP_REFINE_CODE: f"Refining code solutions (best: {self._best_score_by_source(state, 'code'):.2f})",
            Action.DEEP_REFINE_INSTRUCTION: f"Refining instruction solutions (best: {self._best_score_by_source(state, 'instruction'):.2f})",
            Action.POOLED_REFINE: f"Synthesizing from {len(state.best_solutions)} top candidates",
            Action.VLM_VERIFY: f"Verifying good solution ({state.best_score:.2f}) with VLM",
        }
        return explanations.get(action, f"Action: {action.name}")
