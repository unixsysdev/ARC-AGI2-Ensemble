"""Main solver pipeline using primitives + step verification.

Orchestrates:
1. English Planner: Generate strategy from training examples
2. Primitive Translator: Convert English to DSL
3. Primitive Interpreter: Execute with state tracking
4. Unified Judge: Verify each step (logic + visual)
"""

from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Any

from .config import Config
from .primitives import PrimitiveInterpreter, ExecutionState, Program
from .judge import UnifiedJudge, FilmstripRenderer
from .planner import EnglishPlanner, PrimitiveTranslator, VisualPlanner

logger = logging.getLogger(__name__)


# Type aliases
Grid = list[list[int]]


class PrimitivesSolver:
    """Main solver using primitives + step verification.
    
    The pipeline:
    1. Plan: Generate English strategy from training examples
    2. Translate: Convert English steps to primitives
    3. Execute: Run primitives with step-by-step verification
    4. Verify: Check each step against the plan (logic + visual)
    """
    
    def __init__(self, client, config: Config):
        """Initialize solver.
        
        Args:
            client: LLM/VLM API client
            config: Solver configuration
        """
        self.client = client
        self.config = config
        
        # Initialize planners (text, visual, and ensemble)
        self.text_planner = EnglishPlanner(client, config)
        self.visual_planner = VisualPlanner(client, config)
        
        # Ensemble planner (combines VLM + LLM)
        from .planner.ensemble import EnsemblePlanner
        self.ensemble_planner = EnsemblePlanner(client, config)
        
        # Translator and interpreter
        self.translator = PrimitiveTranslator(client, config)
        self.interpreter = PrimitiveInterpreter()
        
        # Initialize filmstrip renderer and judge
        self.filmstrip_renderer = FilmstripRenderer(config.filmstrips_dir)
        self.judge = UnifiedJudge(
            client, 
            config, 
            self.filmstrip_renderer
        )
    
    async def solve(self, task: Any, test_index: int = 0) -> Grid:
        """Solve a single test case.
        
        Args:
            task: ARC Task with training examples
            test_index: Which test case to solve
            
        Returns:
            Predicted output grid
        """
        logger.info(f"Solving task {task.task_id}, test {test_index}")
        
        # Get input grid
        test_input = task.test[test_index].input
        
        # 1. Plan: Generate English strategy
        logger.info("Step 1: Generating plan...")
        if self.config.use_visual_planning:
            logger.info("  Using VISUAL planner (VLM with grid images)")
            english_plan = await self.visual_planner.generate_plan(task)
        else:
            logger.info("  Using TEXT planner (English descriptions)")
            english_plan = await self.text_planner.plan(task)
        logger.debug(f"Plan:\\n{english_plan}")
        
        # 2. Translate: Convert to primitives
        logger.info("Step 2: Translating to primitives...")
        program = await self.translator.translate(english_plan)
        logger.debug(f"Program: {program}")
        
        # 3. Execute with verification
        logger.info("Step 3: Executing with verification...")
        final_state = await self._execute_with_verification(
            test_input, 
            program, 
            english_plan
        )
        
        return final_state.grid
    
    async def _execute_with_verification(
        self,
        input_grid: Grid,
        program: Program,
        english_plan: str
    ) -> ExecutionState:
        """Execute program with step-by-step verification.
        
        Uses confidence-based VLM skip for simple operations to reduce latency.
        
        Args:
            input_grid: Input grid
            program: Program to execute
            english_plan: English plan for verification context
            
        Returns:
            Final execution state
        """
        from .judge.unified import LogicJudge
        logic_judge = LogicJudge()
        
        # Initialize state
        current_state = ExecutionState.initial(input_grid)
        all_states = [current_state]
        step_failures = []  # Collect failures for feedback
        vlm_calls_skipped = 0
        
        for i, primitive in enumerate(program.steps):
            logger.info(f"  Executing step {i+1}/{len(program.steps)}: {primitive.english[:50]}...")
            
            # Execute primitive
            new_state = self.interpreter.execute_step(current_state, primitive)
            
            # Verify step
            if self.config.use_vlm_verification:
                # Check if we can skip VLM based on confidence
                if logic_judge.should_skip_vlm(current_state, new_state):
                    vlm_calls_skipped += 1
                    score, reason = logic_judge.get_confidence_score(current_state, new_state)
                    logger.debug(f"  Step {i+1} VLM skipped (confidence={score:.2f}): {reason}")
                else:
                    # Full VLM verification
                    is_valid, reason = await self.judge.verify_step(
                        current_state,
                        new_state,
                        primitive.english
                    )
                    
                    if not is_valid:
                        logger.warning(f"  Step {i+1} failed verification: {reason}")
                        # Capture failure for feedback
                        step_failures.append(f"Step {i+1} ({primitive.english[:30]}...): {reason[:100]}")
                    else:
                        logger.debug(f"  Step {i+1} verified: {reason}")
            
            current_state = new_state
            all_states.append(current_state)
            
            # Stop on error
            if current_state.error:
                logger.error(f"  Execution error at step {i+1}: {current_state.error}")
                step_failures.append(f"Step {i+1} crashed: {current_state.error}")
                break
        
        if vlm_calls_skipped > 0:
            logger.info(f"  VLM calls skipped: {vlm_calls_skipped}/{len(program.steps)} (confidence-based)")
        
        # Render final filmstrip
        filmstrip_path = self.filmstrip_renderer.render(all_states)
        logger.info(f"Filmstrip saved to: {filmstrip_path}")
        
        return current_state, step_failures
    
    async def solve_with_feedback(
        self, 
        task: Any, 
        test_index: int = 0,
        previous_feedback: list[str] = None
    ) -> tuple[Grid, list[str], Program]:
        """Solve with optional feedback from previous attempts.
        
        Args:
            task: ARC Task with training examples
            test_index: Which test case to solve
            previous_feedback: List of failures from previous attempt
            
        Returns:
            (predicted grid, list of step failures, program used)"""
        logger.info(f"Solving task {task.task_id}, test {test_index}")
        
        test_input = task.test[test_index].input
        
        # 1. Plan with feedback
        logger.info("Step 1: Generating plan...")
        if self.config.use_ensemble_planning:
            logger.info("  Using ENSEMBLE planner (VLM + LLM dual-path)")
            english_plan = await self.ensemble_planner.generate_plan(task, previous_feedback)
        elif self.config.use_visual_planning:
            logger.info("  Using VISUAL planner (VLM with grid images)")
            english_plan = await self.visual_planner.generate_plan(task, previous_feedback)
        else:
            logger.info("  Using TEXT planner (English descriptions)")
            english_plan = await self.text_planner.plan(task)
        logger.debug(f"Plan:\n{english_plan}")
        
        # 2. Translate
        logger.info("Step 2: Translating to primitives...")
        program = await self.translator.translate(english_plan)
        
        # 3. Execute with verification
        logger.info("Step 3: Executing with verification...")
        final_state, step_failures = await self._execute_with_verification(
            test_input, 
            program, 
            english_plan
        )
        
        return final_state.grid, step_failures, program
    
    async def solve_with_retry(
        self,
        task: Any,
        test_index: int = 0,
        max_attempts: int = 3
    ) -> list[Grid]:
        """Solve with multiple attempts and LEARNING LOOP.
        
        Each attempt:
        1. Generate plan and translate to primitives
        2. Execute on TEST input
        3. VALIDATE by running same program on TRAINING inputs
        4. Compare to expected TRAINING outputs
        5. If wrong, add specific failures to feedback and retry
        
        Args:
            task: ARC Task
            test_index: Which test to solve
            max_attempts: Maximum attempts
            
        Returns:
            List of candidate solutions (best first)
        """
        candidates = []
        feedback = None  # Starts with no feedback
        
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt + 1}/{max_attempts}")
            
            if feedback:
                logger.info(f"  Using feedback from {len(feedback)} previous failures")
            
            try:
                grid, step_failures, program = await self.solve_with_feedback(
                    task, test_index, feedback
                )
                candidates.append(grid)
                
                # VALIDATE on training examples - this is ground truth!
                training_failures = self._validate_on_training(task, program)
                
                if training_failures:
                    # Solution is WRONG - add failures to feedback
                    logger.warning(f"  VALIDATION FAILED: {len(training_failures)} training examples wrong")
                    for f in training_failures[:3]:  # Show first 3
                        logger.warning(f"    - {f}")
                    step_failures.extend(training_failures)
                else:
                    # Solution works on ALL training examples!
                    logger.info("  ✓ VALIDATION PASSED: Correct on all training examples!")
                    step_failures = []  # Clear failures since we're good
                
                # Pass failures to next attempt as feedback
                feedback = step_failures if step_failures else None
                
                if not step_failures:
                    logger.info("  All steps verified! Stopping early.")
                    break
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                feedback = [f"Attempt crashed: {str(e)}"]
        
        return candidates
    
    def _validate_on_training(self, task: Any, program: Program) -> list[str]:
        """Run program on ALL training inputs and compare to expected outputs.
        
        This is the GROUND TRUTH test - if our solution works on training
        examples, it's likely correct for test.
        
        Args:
            task: ARC Task with training examples
            program: The program to validate
            
        Returns:
            List of failure messages (empty if all correct)
        """
        failures = []
        
        for i, pair in enumerate(task.train):
            train_input = pair.input
            expected_output = pair.output
            
            try:
                # Execute the SAME program on training input
                states = self.interpreter.execute_program(train_input, program)
                actual = states[-1].grid  # Final state
                
                # Check dimensions
                if len(actual) != len(expected_output) or len(actual[0]) != len(expected_output[0]):
                    failures.append(f"Train {i+1}: Size {len(actual)}x{len(actual[0])} != expected {len(expected_output)}x{len(expected_output[0])}")
                    continue
                
                # Cell-by-cell comparison
                wrong_cells = 0
                for r in range(len(expected_output)):
                    for c in range(len(expected_output[0])):
                        if actual[r][c] != expected_output[r][c]:
                            wrong_cells += 1
                
                if wrong_cells > 0:
                    total = len(expected_output) * len(expected_output[0])
                    pct = 100 * wrong_cells / total
                    failures.append(f"Train {i+1}: {wrong_cells}/{total} cells wrong ({pct:.0f}%)")
                else:
                    logger.debug(f"  Train {i+1}: ✓ correct")
                    
            except Exception as e:
                failures.append(f"Train {i+1}: Execution failed: {str(e)}")
        
        return failures


async def solve_task(task_id: str, config: Config | None = None) -> list[Grid]:
    """Convenience function to solve a task by ID.
    
    Args:
        task_id: Task ID (filename without .json)
        config: Optional configuration
        
    Returns:
        List of candidate solutions
    """
    import sys
    from pathlib import Path
    
    if config is None:
        from .config import load_config
        config = load_config()
    
    # Import API client from arc_solver
    try:
        from systems.llms.chutes_client import ChutesClient
        from systems.models.task import Task
    except ImportError:
        # Fallback to relative import
        import sys
        arc_solver_path = Path(__file__).parent.parent.parent.parent / "arc_solver"
        sys.path.insert(0, str(arc_solver_path))
        from systems.llms.chutes_client import ChutesClient
        from systems.models.task import Task
    
    # Load task
    task_path = config.data_dir / "training" / f"{task_id}.json"
    if not task_path.exists():
        task_path = config.data_dir / "evaluation" / f"{task_id}.json"
    
    if not task_path.exists():
        raise FileNotFoundError(f"Task not found: {task_id}")
    
    task = Task.from_json(task_path)
    
    # Solve
    async with ChutesClient(config) as client:
        solver = PrimitivesSolver(client, config)
        return await solver.solve_with_retry(task)
