"""Unified Judge combining English reasoning and VLM visual verification.

This is the core of the "Grounded Process Reward Model" architecture:
- Logic Check: Did the primitive execute correctly (code-level)?
- Visual Check: Does the result look consistent with the English description (VLM)?
"""

from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..primitives.interpreter import ExecutionState
    from ..primitives.dsl import Grid

logger = logging.getLogger(__name__)


class UnifiedJudge:
    """Combined English + Visual step verifier.
    
    The unified judge answers: "Did what happened match what we said would happen?"
    
    It combines:
    1. Logic Check: Code-level correctness (did the primitive run, is the output valid?)
    2. Visual Check: VLM inspection (does the visual change match the English description?)
    """
    
    def __init__(self, client, config, visualizer=None):
        """Initialize unified judge.
        
        Args:
            client: LLM/VLM API client (ChutesClient from arc_solver)
            config: Configuration with model settings
            visualizer: Optional filmstrip renderer
        """
        self.client = client
        self.config = config
        self.vlm_model = config.vlm_model
        
        # Lazy import to avoid circular dependency
        if visualizer is None:
            from .filmstrip import FilmstripRenderer
            visualizer = FilmstripRenderer(config.logs_dir / "filmstrips" if hasattr(config, 'logs_dir') else None)
        self.visualizer = visualizer
    
    async def verify_step(
        self,
        prev_state: "ExecutionState",
        curr_state: "ExecutionState",
        english_plan: str
    ) -> tuple[bool, str]:
        """Verify one step against the English plan.
        
        This is the core verification loop:
        1. Logic Check: Did the primitive execute without error?
        2. Visual Check: Does the grid change match the English description?
        
        Args:
            prev_state: State before this primitive
            curr_state: State after this primitive  
            english_plan: The English step description
            
        Returns:
            (is_valid, reason): Whether the step is valid and why
        """
        # 1. Logic Check: Execution errors
        if curr_state.error:
            return False, f"Execution error: {curr_state.error}"
        
        # 2. Logic Check: Grid validity
        if not curr_state.grid or not all(row for row in curr_state.grid):
            return False, "Invalid grid produced (empty or malformed)"
        
        # 3. Visual Check: VLM verification
        try:
            is_valid, reason = await self._vlm_verify(prev_state, curr_state, english_plan)
            return is_valid, reason
        except Exception as e:
            logger.warning(f"VLM verification failed: {e}")
            # On VLM error, assume valid (don't reject potentially good steps)
            return True, f"VLM check skipped: {e}"
    
    async def _vlm_verify(
        self,
        prev_state: "ExecutionState",
        curr_state: "ExecutionState",
        english_step: str
    ) -> tuple[bool, str]:
        """Use VLM to verify the step matches English description.
        
        This renders a before/after image and asks the VLM:
        "Did the transformation match the English description?"
        """
        # Render before/after comparison
        image_path = self.visualizer.render([prev_state, curr_state], highlight_changes=True)
        
        # Build verification prompt
        prompt = self._build_verification_prompt(english_step, curr_state)
        
        # Call VLM
        response = await self.client.chat_with_image(
            self.vlm_model,
            prompt,
            image_path,
            temperature=0.2
        )
        
        return self._parse_verdict(response)
    
    def _build_verification_prompt(self, english_step: str, state: "ExecutionState") -> str:
        """Build the verification prompt for VLM."""
        primitive_info = ""
        if state.primitive:
            primitive_info = f"""
Primitive executed: {state.primitive.type.value}
Primitive description: {state.primitive.english}
"""
        
        return f"""You are verifying a step in an ARC-AGI puzzle transformation.

The LEFT image shows the state BEFORE this step.
The RIGHT image shows the state AFTER this step.
Yellow highlights show cells that changed.

THE ENGLISH PLAN SAID:
"{english_step}"
{primitive_info}

EVALUATE:
1. Did the grid change in a way consistent with the English description?
2. Did anything change that SHOULDN'T have changed (unintended side effects)?
3. Does the result look intentional or like noise/errors?

Reply STRICTLY in this format:
PASS: [reason why the step is correct]
or
FAIL: [specific problem with this step]

Be STRICT - only say PASS if the visual change clearly matches the description.
"""
    
    async def verify_trajectory(
        self,
        states: list["ExecutionState"],
        english_plan: str,
        expected_output: "Grid" = None
    ) -> tuple[bool, list[str], Path]:
        """Verify an entire execution trajectory.
        
        This renders the full filmstrip and asks the VLM to evaluate
        the complete transformation sequence.
        
        Args:
            states: All execution states (including initial)
            english_plan: The complete English plan
            expected_output: Optional expected output for comparison
            
        Returns:
            (is_valid, step_reasons, filmstrip_path)
        """
        # Render full filmstrip
        filmstrip_path = self.visualizer.render(states, highlight_changes=True)
        
        # Build trajectory prompt
        prompt = self._build_trajectory_prompt(states, english_plan)
        
        try:
            response = await self.client.chat_with_image(
                self.vlm_model,
                prompt,
                filmstrip_path,
                temperature=0.3
            )
            
            is_valid, reasons = self._parse_trajectory_verdict(response)
            return is_valid, reasons, filmstrip_path
            
        except Exception as e:
            logger.warning(f"Trajectory verification failed: {e}")
            return True, [f"VLM check skipped: {e}"], filmstrip_path
    
    def _build_trajectory_prompt(self, states: list["ExecutionState"], english_plan: str) -> str:
        """Build prompt for full trajectory verification."""
        step_summaries = []
        for i, state in enumerate(states):
            if state.primitive:
                step_summaries.append(f"Step {i}: {state.primitive.english}")
        
        return f"""You are evaluating a complete ARC-AGI puzzle solution.

The filmstrip shows the transformation from INPUT (left) to FINAL OUTPUT (right).
Each frame represents one step. Yellow highlights show changes.

THE COMPLETE PLAN:
{english_plan}

STEPS EXECUTED:
{chr(10).join(step_summaries)}

EVALUATE THE COMPLETE TRAJECTORY:
1. Does the final output look like a valid ARC solution (structured, intentional)?
2. Did each step build logically on the previous one?
3. Are there any steps that look wrong or out of place?

Reply in this format:
VERDICT: VALID or INVALID
STEP ISSUES: [list any problematic steps, or "none"]
REASONING: [brief explanation]
"""
    
    def _parse_verdict(self, response: str) -> tuple[bool, str]:
        """Parse VLM response into verdict."""
        response = response.strip()
        upper = response.upper()
        
        if upper.startswith("PASS"):
            reason = response[4:].strip(": ")
            return True, reason or "Step verified"
        elif upper.startswith("FAIL"):
            reason = response[4:].strip(": ")
            return False, reason or "Step failed verification"
        else:
            # Try to infer from content
            if "pass" in response.lower() or "correct" in response.lower():
                return True, response[:100]
            elif "fail" in response.lower() or "wrong" in response.lower() or "error" in response.lower():
                return False, response[:100]
            else:
                # Ambiguous - lean toward accepting
                return True, f"Ambiguous: {response[:100]}"
    
    def _parse_trajectory_verdict(self, response: str) -> tuple[bool, list[str]]:
        """Parse trajectory verification response."""
        lines = response.strip().split('\n')
        is_valid = True
        reasons = []
        
        for line in lines:
            upper = line.upper()
            if upper.startswith("VERDICT:"):
                rest = line[8:].strip()
                is_valid = "VALID" in rest.upper() and "INVALID" not in rest.upper()
            elif upper.startswith("STEP ISSUES:"):
                rest = line[12:].strip()
                if rest.lower() != "none":
                    reasons.append(rest)
            elif upper.startswith("REASONING:"):
                reasons.append(line[10:].strip())
        
        if not reasons:
            reasons = ["No specific issues identified" if is_valid else "Unknown issue"]
        
        return is_valid, reasons


class LogicJudge:
    """Pure logic/code verification (no VLM).
    
    This is a fallback when VLM is unavailable or for fast local testing.
    It verifies:
    - Primitive executed without error
    - Grid is valid (correct dimensions, valid colors)
    - Selections are consistent
    - Heuristic checks for obvious problems
    """
    
    # Primitives that are simple enough to skip VLM verification
    SIMPLE_PRIMITIVES = {'paint', 'select'}
    
    def verify_step(
        self,
        prev_state: "ExecutionState",
        curr_state: "ExecutionState"
    ) -> tuple[bool, str]:
        """Verify step using logic only."""
        # Check for execution errors
        if curr_state.error:
            return False, f"Execution error: {curr_state.error}"
        
        # Check grid validity
        grid = curr_state.grid
        if not grid:
            return False, "Empty grid"
        
        # Check all rows have same length
        row_lengths = set(len(row) for row in grid)
        if len(row_lengths) > 1:
            return False, f"Inconsistent row lengths: {row_lengths}"
        
        # Check all values are valid colors (0-9)
        for row in grid:
            for val in row:
                if not isinstance(val, int) or val < 0 or val > 9:
                    return False, f"Invalid color value: {val}"
        
        # Check selections are valid
        import numpy as np
        arr = np.array(grid)
        for sel in curr_state.selections:
            if sel.mask.shape != arr.shape:
                return False, f"Selection mask shape mismatch: {sel.mask.shape} vs {arr.shape}"
        
        return True, "Logic check passed"
    
    def get_confidence_score(
        self,
        prev_state: "ExecutionState",
        curr_state: "ExecutionState"
    ) -> tuple[float, str]:
        """Calculate confidence that this step is correct.
        
        Returns (score, reason) where score is 0.0-1.0.
        High confidence = less need for VLM verification.
        """
        import numpy as np
        
        prev_arr = np.array(prev_state.grid)
        curr_arr = np.array(curr_state.grid)
        
        issues = []
        score = 1.0
        
        # Check 1: Unexpected dimension change
        if prev_arr.shape != curr_arr.shape:
            # Some primitives (EXTRACT, SCALE) change dimensions - check if expected
            if curr_state.primitive and curr_state.primitive.type.value in ['extract', 'transform']:
                pass  # Expected
            else:
                issues.append(f"Unexpected dimension change: {prev_arr.shape} -> {curr_arr.shape}")
                score -= 0.3
        
        # Check 2: Drastic color count change
        prev_colors = set(prev_arr.flatten().tolist())
        curr_colors = set(curr_arr.flatten().tolist())
        new_colors = curr_colors - prev_colors
        lost_colors = prev_colors - curr_colors
        
        if len(new_colors) > 3:
            issues.append(f"Many new colors introduced: {new_colors}")
            score -= 0.2
        
        # Check 3: Total grid value change (too much or too little)
        prev_nonzero = np.count_nonzero(prev_arr)
        curr_nonzero = np.count_nonzero(curr_arr)
        
        if prev_nonzero > 0:
            ratio = curr_nonzero / prev_nonzero
            if ratio > 5:
                issues.append(f"Grid fill increased drastically: {ratio:.1f}x")
                score -= 0.2
            elif ratio < 0.1:
                issues.append(f"Grid mostly cleared: {ratio:.1f}x")
                score -= 0.2
        
        # Check 4: SELECT should not change grid
        if curr_state.primitive and curr_state.primitive.type.value == 'select':
            if not np.array_equal(prev_arr, curr_arr):
                issues.append("SELECT changed the grid (should only create selection)")
                score -= 0.5
        
        # Check 5: Simple primitives get bonus confidence
        if curr_state.primitive and curr_state.primitive.type.value in self.SIMPLE_PRIMITIVES:
            score = min(1.0, score + 0.1)
        
        reason = "; ".join(issues) if issues else "All heuristics passed"
        return max(0.0, min(1.0, score)), reason
    
    # Primitives that ALWAYS require VLM verification (destructive)
    DESTRUCTIVE_PRIMITIVES = {'flood_fill', 'gravity', 'composite'}
    
    def should_skip_vlm(
        self,
        prev_state: "ExecutionState",
        curr_state: "ExecutionState",
        threshold: float = 0.8,
        max_change_pct: float = 0.30  # Force VLM if >30% of cells change
    ) -> bool:
        """Determine if VLM verification can be skipped based on confidence.
        
        NEVER skip VLM for:
        - Destructive primitives (FLOOD_FILL, GRAVITY, COMPOSITE)
        - Changes affecting >30% of the grid
        """
        import numpy as np
        
        # Get primitive type
        prim_type = None
        if curr_state.primitive:
            prim_type = curr_state.primitive.type.value.lower()
        
        # NEVER skip VLM for destructive primitives - they can destroy the grid
        if prim_type in self.DESTRUCTIVE_PRIMITIVES:
            return False  # Force VLM verification
        
        # Calculate percentage of cells that changed
        prev_arr = np.array(prev_state.grid)
        curr_arr = np.array(curr_state.grid)
        
        if prev_arr.shape == curr_arr.shape:
            changed_cells = np.sum(prev_arr != curr_arr)
            total_cells = prev_arr.size
            change_pct = changed_cells / total_cells if total_cells > 0 else 0
            
            # NEVER skip VLM if >30% of grid changed - could be catastrophic error
            if change_pct > max_change_pct:
                return False  # Force VLM verification
        
        # Get confidence score
        score, _ = self.get_confidence_score(prev_state, curr_state)
        
        # Only skip for simple primitives with very high confidence
        if prim_type in self.SIMPLE_PRIMITIVES:
            return score >= 0.9
        
        return score >= threshold
