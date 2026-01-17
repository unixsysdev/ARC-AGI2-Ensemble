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
from .planner.feedback_analyzer import FeedbackAnalyzer

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
        
        # Free-form interpreter, critic, and translator
        from .planner.freeform_interpreter import FreeFormInterpreter
        from .planner.freeform_critic import FreeFormCritic
        self.freeform_interpreter = FreeFormInterpreter(client, config)
        self.freeform_critic = FreeFormCritic(client, config)
        self.translator = PrimitiveTranslator(client, config)
        self.interpreter = PrimitiveInterpreter()
        
        # Track partial successes (idea correct, code failed)
        self.idea_correct_code_failed = 0
        
        # Initialize filmstrip renderer and judge
        self.filmstrip_renderer = FilmstripRenderer(config.filmstrips_dir)
        self.judge = UnifiedJudge(
            client, 
            config, 
            self.filmstrip_renderer
        )
        
        # Feedback analyzer for reasoning over failures
        self.feedback_analyzer = FeedbackAnalyzer(client, config)
    
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
        
        # 2. Interpret free-form arguments (if freeform mode enabled)
        logger.info("Step 2: Translating to primitives...")
        if getattr(self.config, 'use_freeform', False):
            interpreted_plan = await self.freeform_interpreter.interpret(english_plan)
        else:
            interpreted_plan = english_plan
        program = await self.translator.translate(interpreted_plan)
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
        vlm_calls_made = 0
        
        for i, primitive in enumerate(program.steps):
            logger.info(f"  Executing step {i+1}/{len(program.steps)}: {primitive.english[:50]}...")
            
            # Execute primitive
            new_state = self.interpreter.execute_step(current_state, primitive)
            
            # Always get LogicJudge feedback first
            logic_score, logic_reason = logic_judge.get_confidence_score(current_state, new_state)
            
            # Get VLM feedback if enabled
            vlm_reason = None
            if self.config.use_vlm_verification:
                vlm_calls_made += 1
                is_valid, vlm_reason = await self.judge.verify_step(
                    current_state,
                    new_state,
                    primitive.english
                )
                
                if not is_valid:
                    logger.warning(f"  Step {i+1} failed VLM verification: {vlm_reason[:100]}...")
            
            # Combine feedback if logic is uncertain OR VLM failed
            if logic_score < 0.8 or (vlm_reason and "failed" in vlm_reason.lower()):
                combined_feedback = f"Step {i+1} ({primitive.english[:30]}...): "
                
                if logic_score < 0.8:
                    combined_feedback += f"Logic: {logic_reason}. "
                
                if vlm_reason and "failed" in str(vlm_reason).lower():
                    combined_feedback += f"Visual: {vlm_reason[:80]}"
                elif vlm_reason:
                    combined_feedback += f"Visual: {vlm_reason[:80]}"
                
                step_failures.append(combined_feedback)
                logger.debug(f"  Step {i+1} feedback: score={logic_score:.2f}, {combined_feedback[:80]}...")
        
            current_state = new_state
            all_states.append(current_state)
            
            # Stop on error
            if current_state.error:
                logger.error(f"  Execution error at step {i+1}: {current_state.error}")
                step_failures.append(f"Step {i+1} crashed: {current_state.error}")
                break
        
        if vlm_calls_made > 0:
            logger.info(f"  VLM calls made: {vlm_calls_made}/{len(program.steps)}")
        
        # Render final filmstrip
        filmstrip_path = self.filmstrip_renderer.render(all_states)
        logger.info(f"Filmstrip saved to: {filmstrip_path}")
        
        return current_state, step_failures
    
    async def solve_with_feedback(
        self, 
        task: Any, 
        test_index: int = 0,
        vlm_feedback: list[str] = None,
        llm_feedback: list[str] = None
    ) -> tuple[Grid, list[str], Program]:
        """Solve with optional feedback from previous attempts.
        
        Args:
            task: ARC Task with training examples
            test_index: Which test case to solve
            vlm_feedback: Feedback for VLM planner (None = no feedback)
            llm_feedback: Feedback for LLM planner (None = no feedback)
            
        Returns:
            (predicted grid, list of step failures, program used)"""
        logger.info(f"Solving task {task.task_id}, test {test_index}")
        
        test_input = task.test[test_index].input
        
        # 1. Plan with feedback
        logger.info("Step 1: Generating plan...")
        if self.config.use_ensemble_planning:
            logger.info("  Using ENSEMBLE planner (VLM + LLM dual-path)")
            english_plan = await self.ensemble_planner.generate_plan(
                task, 
                vlm_feedback=vlm_feedback,
                llm_feedback=llm_feedback
            )
        elif self.config.use_visual_planning:
            logger.info("  Using VISUAL planner (VLM with grid images)")
            english_plan = await self.visual_planner.generate_plan(task, vlm_feedback)
        else:
            logger.info("  Using TEXT planner (English descriptions)")
            english_plan = await self.text_planner.plan(task)
        logger.debug(f"Plan:\n{english_plan}")
        
        # 2. Validate and interpret (if freeform mode enabled)
        logger.info("Step 2: Translating to primitives...")
        critic_verdict = None
        if getattr(self.config, 'use_freeform', False):
            # First, validate the idea with critic
            critic_verdict = await self.freeform_critic.validate(english_plan, task)
            # Then interpret to structured DSL
            interpreted_plan = await self.freeform_interpreter.interpret(english_plan)
        else:
            interpreted_plan = english_plan
        program = await self.translator.translate(interpreted_plan)
        
        # 3. Execute with verification
        logger.info("Step 3: Executing with verification...")
        final_state, step_failures = await self._execute_with_verification(
            test_input, 
            program, 
            english_plan
        )
        
        # Track partial success: idea was correct but code failed
        if critic_verdict and critic_verdict.valid and step_failures:
            self.idea_correct_code_failed += 1
            logger.warning(f"[PARTIAL SUCCESS] 💡 Idea was CORRECT (critic: {critic_verdict.confidence}) but code failed!")
            logger.warning(f"[PARTIAL SUCCESS] This indicates a DSL expressiveness issue, not a reasoning failure.")
        
        return final_state.grid, step_failures, program
    
    async def solve_with_retry(
        self,
        task: Any,
        test_index: int = 0,
        max_attempts: int = 3,
        vlm_feedback: bool = True,
        llm_feedback: bool = True,
        feedback_limit: int = None,
        use_reasoning: bool = True
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
            vlm_feedback: If True, send failures to VLM planner (default: True)
            llm_feedback: If True, send failures to LLM planner (default: True)
            feedback_limit: If set, only include last N failures (default: all)
            use_reasoning: If True, use LLM to reason over failures (default: True)
            
        Returns:
            List of candidate solutions (best first)
        """
        candidates = []
        all_failures = []  # Accumulate ALL failures across attempts
        reasoned_advice = ""  # Synthesized recommendation from analyzer
        
        # Set up run context for organized filmstrip output
        self.filmstrip_renderer.set_run_context(task.task_id)
        logger.info(f"  Run output: {self.filmstrip_renderer.output_dir}")
        
        # Log feedback settings
        if not vlm_feedback and not llm_feedback:
            logger.info("  Feedback DISABLED for both VLM and LLM")
        elif not vlm_feedback:
            logger.info("  Feedback DISABLED for VLM (LLM still gets feedback)")
        elif not llm_feedback:
            logger.info("  Feedback DISABLED for LLM (VLM still gets feedback)")
        if feedback_limit:
            logger.info(f"  Feedback LIMITED to last {feedback_limit} failures")
        if use_reasoning:
            logger.info("  Feedback REASONING enabled (LLM will analyze failures)")
        
        for attempt in range(max_attempts):
            self.filmstrip_renderer.next_attempt()  # Increment attempt counter
            logger.info(f"Attempt {attempt + 1}/{max_attempts}")
            
            # Apply limit if set (take last N failures)
            limited_failures = all_failures[-feedback_limit:] if feedback_limit else all_failures
            
            # Generate reasoned advice if enabled and we have failures
            if use_reasoning and limited_failures and not reasoned_advice:
                reasoned_advice = await self.feedback_analyzer.analyze(limited_failures)
            
            # Build feedback: raw + reasoned advice
            combined_feedback_vlm = None
            combined_feedback_llm = None
            
            if vlm_feedback and limited_failures:
                combined_feedback_vlm = limited_failures.copy()
                if reasoned_advice:
                    combined_feedback_vlm.insert(0, f"[ANALYSIS] {reasoned_advice}")
            
            if llm_feedback and limited_failures:
                combined_feedback_llm = limited_failures.copy()
                if reasoned_advice:
                    combined_feedback_llm.insert(0, f"[ANALYSIS] {reasoned_advice}")
            
            if combined_feedback_vlm or combined_feedback_llm:
                logger.info(f"  Using feedback from {len(limited_failures)} failures (of {len(all_failures)} total)")
                if reasoned_advice:
                    logger.info(f"  + Reasoned advice: {reasoned_advice[:80]}...")
            
            try:
                grid, step_failures, program = await self.solve_with_feedback(
                    task, test_index, 
                    vlm_feedback=combined_feedback_vlm,
                    llm_feedback=combined_feedback_llm
                )
                candidates.append(grid)
                
                # VALIDATE on training examples - this is ground truth!
                training_failures = self._validate_on_training(task, program)
                
                if training_failures:
                    # Solution is WRONG - ACCUMULATE failures
                    logger.warning(f"  VALIDATION FAILED: {len(training_failures)} training examples wrong")
                    for f in training_failures[:3]:  # Show first 3
                        logger.warning(f"    - {f}")
                    # Add to accumulated failures
                    all_failures.extend(training_failures)
                    all_failures.extend(step_failures)
                    # Reset reasoned advice so we re-analyze with new failures
                    reasoned_advice = ""
                else:
                    # Solution works on ALL training examples!
                    logger.info("  ✓ VALIDATION PASSED: Correct on all training examples!")
                    
                    # Rename filmstrip to mark as WINNER
                    self.filmstrip_renderer.mark_winner()
                    break  # Success!
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                all_failures.append(f"Attempt {attempt + 1} crashed: {str(e)}")
        
        # Log summary for free-form mode
        if getattr(self.config, 'use_freeform', False):
            logger.info("=" * 60)
            logger.info("[FREE-FORM SUMMARY]")
            logger.info(f"  Total attempts: {max_attempts}")
            logger.info(f"  Ideas validated as CORRECT by critic: (tracked per-attempt)")
            logger.info(f"  Ideas correct but CODE FAILED: {self.idea_correct_code_failed}")
            if self.idea_correct_code_failed > 0:
                logger.info(f"  ⚠️  DSL expressiveness gap detected - model reasoning is ahead of our code!")
            logger.info("=" * 60)
        
        return candidates
    
    def _validate_on_training(self, task: Any, program: Program) -> list[str]:
        """Run program on ALL training inputs and compare to expected outputs.
        
        This is the GROUND TRUTH test - if our solution works on training
        examples, it's likely correct for test.
        
        Provides DETAILED feedback for the LLM to learn from failures.
        
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
                    actual_h, actual_w = len(actual), len(actual[0])
                    exp_h, exp_w = len(expected_output), len(expected_output[0])
                    
                    msg = f"Train {i+1}: Size {actual_h}x{actual_w} != expected {exp_h}x{exp_w}"
                    
                    # Note if extraction is needed (let FeedbackAnalyzer reason out the details)
                    if exp_h < actual_h or exp_w < actual_w:
                        msg += f" → Need {exp_h}x{exp_w} output, use extract()"
                    
                    failures.append(msg)
                    continue
                
                # Cell-by-cell comparison with detail
                wrong_cells = 0
                actual_colors = {}
                expected_colors = {}
                
                for r in range(len(expected_output)):
                    for c in range(len(expected_output[0])):
                        expected_val = expected_output[r][c]
                        actual_val = actual[r][c]
                        
                        expected_colors[expected_val] = expected_colors.get(expected_val, 0) + 1
                        actual_colors[actual_val] = actual_colors.get(actual_val, 0) + 1
                        
                        if actual_val != expected_val:
                            wrong_cells += 1
                
                if wrong_cells > 0:
                    total = len(expected_output) * len(expected_output[0])
                    pct = 100 * wrong_cells / total
                    
                    # Provide specific feedback about color distribution
                    expected_dist = ", ".join(f"{k}:{v}" for k, v in sorted(expected_colors.items()))
                    actual_dist = ", ".join(f"{k}:{v}" for k, v in sorted(actual_colors.items()))
                    
                    feedback = f"Train {i+1}: {wrong_cells}/{total} cells wrong ({pct:.0f}%)"
                    
                    # Add color distribution if significantly different
                    if actual_colors != expected_colors:
                        feedback += f" | Colors expected [{expected_dist}] got [{actual_dist}]"
                    
                    # Add specific hint if entire grid is one color (common failure)
                    if len(actual_colors) == 1:
                        only_color = list(actual_colors.keys())[0]
                        feedback += f" | Output is ALL color {only_color} - transformation likely wrong"
                    
                    failures.append(feedback)
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
