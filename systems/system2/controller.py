"""System 2 Controller - The meta-cognitive manager."""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Any

from ..config import Config
from ..llms.chutes_client import ChutesClient
from ..llms.local_client import LocalLLMClient, LocalCodeGenerator
from ..models.task import Task, Grid, score_grid
from ..system1.code_generator import CodeGenerator
from ..system1.code_executor import CodeExecutor
from ..system1.instruction_generator import InstructionGenerator
from ..system1.instruction_executor import InstructionExecutor
from ..critic.visualizer import GridVisualizer
from ..critic.vlm_critic import VLMCritic
from .state import TaskState, Action, ScoredSolution
from .policy import HeuristicPolicy

logger = logging.getLogger(__name__)


class System2Controller:
    """
    The meta-cognitive controller that orchestrates System 1 workers.
    
    This is the "Manager" that decides WHAT to do without doing the work itself.
    It observes state, decides actions via policy, and dispatches to workers.
    """
    
    def __init__(self, client: ChutesClient, config: Config):
        self.client = client
        self.config = config
        
        # Local LLM client (if enabled)
        self.local_client = None
        self.local_code_generator = None
        if config.use_local:
            # Model name "auto" = detect from vLLM /v1/models endpoint
            self.local_client = LocalLLMClient(
                base_url=config.local_url,
                model_name="auto",  # Will auto-detect from vLLM
                max_concurrency=config.local_concurrency,  # Use config value
                timeout=float(config.local_timeout)  # Configurable timeout
            )
            self.local_code_generator = LocalCodeGenerator(self.local_client)
            logger.info(f"Local LLM enabled at {config.local_url} (concurrency={config.local_concurrency}, timeout={config.local_timeout}s)")
        
        # System 1 workers (remote API)
        self.code_generator = CodeGenerator(client, config)
        self.code_executor = CodeExecutor(config.code_execution_timeout)
        self.instruction_generator = InstructionGenerator(client, config)
        self.instruction_executor = InstructionExecutor(client, config)
        
        # Critic
        self.visualizer = GridVisualizer(config.logs_dir / "viz")
        self.vlm_critic = VLMCritic(client, config, self.visualizer)
        
        # Policy (with hybrid mode flag)
        self.policy = HeuristicPolicy(hybrid_mode=config.hybrid_mode)
    
    async def solve_task(self, task: Task) -> list[Grid]:
        """
        Main solve loop for a single task.
        
        Returns up to 2 candidate outputs for each test input.
        """
        start_time = time.time()
        state = TaskState.from_task(task, self.config.max_attempts_per_task)
        
        logger.info(f"Starting task {task.task_id} (grid_size={state.grid_size}, colors={state.num_colors})")
        
        # Main loop
        while True:
            # Update time
            state.time_elapsed = time.time() - start_time
            
            # Get action from policy
            action = self.policy.decide(state)
            explanation = self.policy.explain(state, action)
            logger.debug(f"Task {task.task_id}: {explanation}")
            
            # Execute action
            if action == Action.SUBMIT:
                break
            elif action == Action.GIVE_UP:
                logger.warning(f"Task {task.task_id}: Giving up")
                break
            else:
                await self._execute_action(task, state, action)
        
        # Collect final outputs for test inputs
        return self._collect_outputs(task, state)
    
    async def _execute_action(self, task: Task, state: TaskState, action: Action):
        """Execute a single action and update state."""
        try:
            match action:
                case Action.FAST_GUESS_CODE:
                    await self._fast_guess_code(task, state)
                case Action.FAST_GUESS_INSTRUCTION:
                    await self._fast_guess_instruction(task, state)
                case Action.DEEP_REFINE_CODE:
                    await self._deep_refine_code(task, state)
                case Action.DEEP_REFINE_INSTRUCTION:
                    await self._deep_refine_instruction(task, state)
                case Action.POOLED_REFINE:
                    await self._pooled_refine(task, state)
                case Action.VLM_VERIFY:
                    await self._vlm_verify(task, state)
                case Action.HYBRID_EXPLORE:
                    await self._hybrid_explore(task, state)
            
            state.record_action(action)
            
        except Exception as e:
            logger.error(f"Action {action.name} failed: {e}")
            state.record_action(action, error=str(e))
    
    async def _fast_guess_code(self, task: Task, state: TaskState, n: int = 5):
        """Generate and test Python code candidates."""
        # Use local LLM if enabled, otherwise use remote API
        if self.local_code_generator:
            # Use larger batch size for local to maximize GPU utilization
            batch_size = self.config.local_batch_size
            codes = await self.local_code_generator.generate(task, n=batch_size)
            logger.info(f"Task {task.task_id}: Generated {len(codes)} LOCAL code candidates (attempt {state.attempts_used})")
        else:
            codes = await self.code_generator.generate(task, n=n)
            logger.info(f"Task {task.task_id}: Generated {len(codes)} code candidates (attempt {state.attempts_used})")
        
        # Always increment attempts by at least 1 per action
        if not codes:
            state.attempts_used += 1
            logger.warning(f"Task {task.task_id}: No valid code extracted from responses")
            return
        
        for code in codes:
            # Test on training pairs
            score, errors = self.code_executor.test_on_training(code, task.train)
            
            # Get output for first training input as representative
            output, exec_error = self.code_executor.execute(code, task.train[0].input)
            
            if output:
                solution = ScoredSolution(
                    output=output,
                    score=score,
                    source="code",
                    artifact=code,
                    errors=errors
                )
                state.add_solution(solution)
                logger.info(f"Task {task.task_id}: Code scored {score:.2f} (best: {state.best_score:.2f})")
                
                if score >= 0.999:
                    logger.info(f"Task {task.task_id}: Perfect code solution found!")
                    break
            else:
                state.attempts_used += 1  # Count failed executions too
                if exec_error:
                    logger.debug(f"Code execution failed: {exec_error[:100]}")
    
    async def _fast_guess_instruction(self, task: Task, state: TaskState, n: int = 5):
        """Generate and test English instruction candidates."""
        instructions = await self.instruction_generator.generate(task, n=n)
        
        logger.info(f"Task {task.task_id}: Generated {len(instructions)} instruction candidates (attempt {state.attempts_used})")
        
        if not instructions:
            state.attempts_used += 1
            logger.warning(f"Task {task.task_id}: No instructions generated")
            return
        
        for instruction in instructions:
            # Test on training pairs
            score, errors = await self.instruction_executor.test_on_training(
                instruction, 
                task.train
            )
            
            # Get output for first training input
            output, _ = await self.instruction_executor.execute(
                instruction, 
                task.train[0].input
            )
            
            if output:
                solution = ScoredSolution(
                    output=output,
                    score=score,
                    source="instruction",
                    artifact=instruction,
                    errors=[f"Training error on pair {i}" for i, e in enumerate(errors) if e]
                )
                state.add_solution(solution)
                logger.info(f"Task {task.task_id}: Instruction scored {score:.2f} (best: {state.best_score:.2f})")
                
                if score >= 0.999:
                    logger.info(f"Task {task.task_id}: Perfect instruction solution found!")
                    break
            else:
                state.attempts_used += 1
    
    async def _deep_refine_code(self, task: Task, state: TaskState):
        """Refine top code solutions based on errors."""
        top_codes = state.get_top_solutions(n=3, source="code")
        
        for solution in top_codes:
            if solution.errors and solution.artifact:
                error_msg = "; ".join(solution.errors[:3])
                revised = await self.code_generator.revise(task, solution.artifact, error_msg)
                
                if revised:
                    score, errors = self.code_executor.test_on_training(revised, task.train)
                    output, _ = self.code_executor.execute(revised, task.train[0].input)
                    
                    if output:
                        new_solution = ScoredSolution(
                            output=output,
                            score=score,
                            source="code",
                            artifact=revised,
                            errors=errors
                        )
                        state.add_solution(new_solution)
    
    async def _deep_refine_instruction(self, task: Task, state: TaskState):
        """Refine top instruction solutions based on errors."""
        top_instructions = state.get_top_solutions(n=3, source="instruction")
        
        for solution in top_instructions:
            if solution.artifact:
                # Get detailed errors
                _, error_tuples = await self.instruction_executor.test_on_training(
                    solution.artifact,
                    task.train
                )
                
                if error_tuples:
                    revised = await self.instruction_generator.revise_individual(
                        task,
                        solution.artifact,
                        error_tuples[:3]  # Limit errors to avoid token overflow
                    )
                    
                    if revised:
                        score, _ = await self.instruction_executor.test_on_training(
                            revised,
                            task.train
                        )
                        output, _ = await self.instruction_executor.execute(
                            revised,
                            task.train[0].input
                        )
                        
                        if output:
                            new_solution = ScoredSolution(
                                output=output,
                                score=score,
                                source="instruction",
                                artifact=revised
                            )
                            state.add_solution(new_solution)
    
    async def _pooled_refine(self, task: Task, state: TaskState):
        """Synthesize from top instruction candidates."""
        top = state.get_top_solutions(n=5, source="instruction")
        if len(top) < 2:
            return
        
        scored_instructions = [
            (s.artifact, s.score) 
            for s in top 
            if s.artifact
        ]
        
        if len(scored_instructions) >= 2:
            synthesized = await self.instruction_generator.revise_pooled(
                task,
                scored_instructions
            )
            
            if synthesized:
                score, _ = await self.instruction_executor.test_on_training(
                    synthesized,
                    task.train
                )
                output, _ = await self.instruction_executor.execute(
                    synthesized,
                    task.train[0].input
                )
                
                if output:
                    new_solution = ScoredSolution(
                        output=output,
                        score=score,
                        source="instruction",
                        artifact=synthesized
                    )
                    state.add_solution(new_solution)
    
    async def _vlm_verify(self, task: Task, state: TaskState):
        """Verify top solutions with VLM critic and revise rejects."""
        top = state.get_top_solutions(n=5)
        
        for solution in top:
            if not solution.verified_by_vlm:
                is_valid, reason = await self.vlm_critic.verify(
                    task.train[0].input,
                    solution.output
                )
                solution.verified_by_vlm = True
                
                if not is_valid:
                    # Penalize invalid solutions
                    logger.info(f"VLM rejected solution: {reason}")
                    solution.score *= 0.5
                    
                    # VLM-in-the-loop: Send rejection reason to remote for targeted fix
                    if solution.artifact and solution.source == "code":
                        logger.info(f"Task {task.task_id}: VLM-guided revision - sending rejection to remote")
                        vlm_feedback = f"VLM visual inspection found: {reason}"
                        revised = await self.code_generator.revise(
                            task,
                            solution.artifact,
                            [vlm_feedback]  # Use VLM reason as error
                        )
                        
                        if revised:
                            new_score, _ = self.code_executor.test_on_training(revised, task.train)
                            new_output, _ = self.code_executor.execute(revised, task.train[0].input)
                            
                            if new_output:
                                new_solution = ScoredSolution(
                                    output=new_output,
                                    score=new_score,
                                    source="code",
                                    artifact=revised
                                )
                                state.add_solution(new_solution)
                                logger.info(f"Task {task.task_id}: VLM-guided fix: {solution.score:.2f} -> {new_score:.2f}")
                else:
                    # Boost verified solutions
                    logger.info(f"VLM approved solution: {reason}")
                    solution.score = min(1.0, solution.score * 1.1)
        
        # Re-sort after score adjustments
        state.best_solutions.sort(key=lambda s: s.score, reverse=True)
    
    async def _feedback_to_local(
        self, 
        task: Task, 
        state: TaskState,
        failed_code: str,
        errors: list[str],
        successful_code: str
    ):
        """
        Phase 4: Send feedback to local model so it can learn from remote's fix.
        
        This teaches the small local model by showing what was wrong and what works,
        then letting it attempt its own fix based on that knowledge.
        """
        if not self.local_generator:
            return
        
        n = self.config.feedback_local_attempts
        logger.info(f"Task {task.task_id}: FEEDBACK Phase - Teaching local from remote's fix ({n} attempts)")
        
        # Get revised codes from local using the feedback
        revised_codes = await self.local_generator.revise_with_feedback(
            task=task,
            failed_code=failed_code,
            errors=errors,
            successful_code=successful_code,
            n=n
        )
        
        if not revised_codes:
            logger.info(f"Task {task.task_id}: FEEDBACK - Local produced no valid revisions")
            state.metadata["local_solved_with_feedback"] = False
            return
        
        # Test revised codes
        local_successes = 0
        for i, code in enumerate(revised_codes):
            score, test_errors = self.code_executor.test_on_training(code, task.train)
            output, _ = self.code_executor.execute(code, task.train[0].input)
            
            if output and score >= 0.999:
                local_successes += 1
                logger.info(f"Task {task.task_id}: FEEDBACK - Local learned! Candidate {i+1} scored {score:.2f}")
                
                solution = ScoredSolution(
                    output=output,
                    score=score,
                    source="code",
                    artifact=code
                )
                state.add_solution(solution)
        
        if local_successes > 0:
            logger.info(f"Task {task.task_id}: FEEDBACK SUCCESS - Local solved {local_successes}/{len(revised_codes)} after learning")
            state.metadata["local_solved_with_feedback"] = True
        else:
            logger.info(f"Task {task.task_id}: FEEDBACK - Local could not reproduce despite feedback")
            state.metadata["local_solved_with_feedback"] = False
    
    async def _hybrid_explore(self, task: Task, state: TaskState):
        """
        Hybrid exploration: Local generates many, remote refines top K.
        
        Phase 1: Local LLM generates many diverse candidates quickly
        Phase 2: Execute and score all candidates
        Phase 3: Remote LLM refines top K candidates
        """
        if not self.local_code_generator:
            logger.warning("Hybrid mode requires local LLM - falling back to remote")
            await self._fast_guess_code(task, state)
            return
        
        # Phase 1: Local exploration - generate many candidates
        n_local = self.config.hybrid_local_candidates
        logger.info(f"Task {task.task_id}: HYBRID Phase 1 - Local generating {n_local} candidates")
        
        local_codes = await self.local_code_generator.generate(task, n=n_local)
        logger.info(f"Task {task.task_id}: Phase 1 complete - extracted {len(local_codes)} valid code candidates")
        
        # Phase 2: Execute and score all local candidates
        logger.info(f"Task {task.task_id}: HYBRID Phase 2 - Testing {len(local_codes)} candidates")
        local_solutions = []
        output_groups: dict[str, list[ScoredSolution]] = {}  # Group by output hash
        
        for i, code in enumerate(local_codes):
            score, errors = self.code_executor.test_on_training(code, task.train)
            output, exec_error = self.code_executor.execute(code, task.train[0].input)
            
            if output and score > 0:
                solution = ScoredSolution(
                    output=output,
                    score=score,
                    source="code",
                    artifact=code,
                    errors=errors
                )
                local_solutions.append(solution)
                state.add_solution(solution)
                
                # Group by output for voting
                output_key = str(output)  # Use string repr as hash
                if output_key not in output_groups:
                    output_groups[output_key] = []
                output_groups[output_key].append(solution)
                
                logger.info(f"Task {task.task_id}: Candidate {i+1}/{len(local_codes)} scored {score:.2f}")
            else:
                logger.debug(f"Task {task.task_id}: Candidate {i+1}/{len(local_codes)} failed (score=0 or error)")
        
        # Voting: Rank output groups by vote count (number of identical outputs)
        if output_groups:
            sorted_groups = sorted(output_groups.items(), key=lambda x: len(x[1]), reverse=True)
            logger.info(f"Task {task.task_id}: VOTING - {len(output_groups)} unique outputs from {len(local_solutions)} candidates")
            for idx, (key, group) in enumerate(sorted_groups[:5]):
                best_in_group = max(group, key=lambda s: s.score)
                logger.info(f"  Group {idx+1}: {len(group)} votes, best score {best_in_group.score:.2f}")
            
            # Diversity-first: Pick best solution from each unique output group
            diverse_solutions = []
            for _, group in sorted_groups:
                best_in_group = max(group, key=lambda s: s.score)
                diverse_solutions.append(best_in_group)
            
            # Re-rank: vote count (descending), then score (descending)
            local_solutions = diverse_solutions
        
        logger.info(f"Task {task.task_id}: Phase 2 complete - {len(local_solutions)} diverse candidates (best: {state.best_score:.2f})") 
        
        # Check for perfect solution
        if state.has_perfect_solution:
            logger.info(f"Task {task.task_id}: Perfect solution found in local exploration!")
            return
        
        # Phase 3: Remote refinement of top K (from diverse set)
        top_k = self.config.hybrid_top_k
        top_local = local_solutions[:top_k]  # Already sorted by votes
        
        if not top_local:
            logger.warning(f"Task {task.task_id}: No viable local candidates for refinement")
            return
        
        logger.info(f"Task {task.task_id}: HYBRID Phase 3 - Remote refining top {len(top_local)} candidates")
        
        for solution in top_local:
            if solution.score >= 0.999:
                continue  # Already perfect
            
            # Get errors from this candidate
            _, errors = self.code_executor.test_on_training(solution.artifact, task.train)
            
            if errors:
                # Use remote to revise
                for _ in range(self.config.hybrid_remote_revisions):
                    revised = await self.code_generator.revise(
                        task,
                        solution.artifact,
                        errors[:3]  # Limit errors
                    )
                    
                    if revised:
                        new_score, new_errors = self.code_executor.test_on_training(revised, task.train)
                        new_output, _ = self.code_executor.execute(revised, task.train[0].input)
                        
                        if new_output:
                            new_solution = ScoredSolution(
                                output=new_output,
                                score=new_score,
                                source="code",
                                artifact=revised
                            )
                            state.add_solution(new_solution)
                            logger.info(f"Task {task.task_id}: Remote refined: {solution.score:.2f} -> {new_score:.2f}")
                            
                            if new_score >= 0.999:
                                logger.info(f"Task {task.task_id}: Perfect solution after remote refinement!")
                                
                                # Phase 4: Feedback to Local (if enabled)
                                if self.config.feedback_to_local and self.local_generator:
                                    await self._feedback_to_local(
                                        task, state,
                                        failed_code=solution.artifact,
                                        errors=errors,
                                        successful_code=revised
                                    )
                                return
        
        # Phase 5: VLM verification of top candidates
        if state.has_good_solution:
            logger.info(f"Task {task.task_id}: HYBRID Phase 5 - VLM verifying top candidates")
            await self._vlm_verify(task, state)
    
    def _collect_outputs(self, task: Task, state: TaskState) -> list[Grid]:
        """Collect final outputs for all test inputs."""
        outputs = []
        
        for test_idx, test_pair in enumerate(task.test):
            # Get best 2 unique outputs for this test input
            test_outputs = []
            seen = set()
            
            # Try to execute best solutions on this test input
            for solution in state.best_solutions[:10]:
                if solution.source == "code" and solution.artifact:
                    output, _ = self.code_executor.execute(
                        solution.artifact,
                        test_pair.input
                    )
                else:
                    # For instructions, we'd need async - skip for now
                    # In production, we'd run parallel execution
                    output = solution.output  # Use training output as fallback
                
                if output:
                    output_key = str(output)
                    if output_key not in seen:
                        seen.add(output_key)
                        test_outputs.append(output)
                        if len(test_outputs) >= 2:
                            break
            
            # Pad with empty if needed
            while len(test_outputs) < 2:
                if test_outputs:
                    test_outputs.append(test_outputs[0])  # Duplicate best
                else:
                    # No solutions - use empty grid same size as input
                    h, w = len(test_pair.input), len(test_pair.input[0])
                    test_outputs.append([[0] * w for _ in range(h)])
            
            outputs.extend(test_outputs[:2])
        
        return outputs
