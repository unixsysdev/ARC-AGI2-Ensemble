"""Main entry point for ARC-AGI-2 Solver."""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import load_config, Config
from .models.task import Task, load_tasks, grids_equal
from .llms.chutes_client import ChutesClient
from .system2.controller import System2Controller


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def solve_single_task(
    controller: System2Controller,
    task: Task,
    verbose: bool = False
) -> dict:
    """Solve a single task and return results."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Solving task: {task.task_id}")
    
    try:
        outputs = await controller.solve_task(task)
        
        # Format outputs for submission
        result = {
            "task_id": task.task_id,
            "outputs": outputs,
            "num_test_pairs": len(task.test)
        }
        
        # If ground truth available, check accuracy
        if all(pair.output for pair in task.test):
            correct = 0
            for i, pair in enumerate(task.test):
                # Get the two attempts for this test pair
                attempt_1 = outputs[i * 2] if i * 2 < len(outputs) else None
                attempt_2 = outputs[i * 2 + 1] if i * 2 + 1 < len(outputs) else None
                
                if grids_equal(attempt_1, pair.output) or grids_equal(attempt_2, pair.output):
                    correct += 1
            
            result["correct"] = correct
            result["total"] = len(task.test)
            result["accuracy"] = correct / len(task.test)
            
            logger.info(f"Task {task.task_id}: {correct}/{len(task.test)} test pairs correct")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to solve task {task.task_id}: {e}", exc_info=True)
        return {
            "task_id": task.task_id,
            "error": str(e),
            "outputs": []
        }


async def run_evaluation(
    config: Config,
    split: str = "evaluation",
    limit: int | None = None,
    offset: int = 0,
    task_ids: list[str] | None = None,
    verbose: bool = False
) -> dict:
    """Run evaluation on a set of tasks."""
    
    # Load tasks
    tasks = load_tasks(config.data_dir, split)
    logger.info(f"Loaded {len(tasks)} tasks from {split}")
    
    # Filter by task_ids if provided
    if task_ids:
        tasks = [t for t in tasks if t.task_id in task_ids]
        logger.info(f"Filtered to {len(tasks)} specified tasks")
    
    # Apply offset and limit
    tasks = tasks[offset:]
    if limit:
        tasks = tasks[:limit]
    
    logger.info(f"Running on {len(tasks)} tasks (offset={offset}, limit={limit})")
    
    # Solve tasks
    results = []
    async with ChutesClient(config) as client:
        controller = System2Controller(client, config)
        
        for task in tasks:
            result = await solve_single_task(controller, task, verbose)
            results.append(result)
            
            # Save intermediate result
            save_attempt(config, result)
    
    # Calculate overall stats
    total_correct = sum(r.get("correct", 0) for r in results)
    total_tests = sum(r.get("total", 0) for r in results)
    
    summary = {
        "tasks_attempted": len(results),
        "tasks_with_correct": sum(1 for r in results if r.get("correct", 0) > 0),
        "total_correct_tests": total_correct,
        "total_tests": total_tests,
        "accuracy": total_correct / total_tests if total_tests > 0 else 0,
        "results": results
    }
    
    logger.info(f"Evaluation complete: {summary['accuracy']:.1%} ({total_correct}/{total_tests})")
    
    return summary


def save_attempt(config: Config, result: dict):
    """Save a single task attempt."""
    attempts_file = config.attempts_dir / f"{result['task_id']}.json"
    with open(attempts_file, "w") as f:
        json.dump(result, f, indent=2)


def save_submission(config: Config, results: list[dict], split: str):
    """Save results in ARC submission format."""
    submission = {}
    
    for result in results:
        if "outputs" in result and result["outputs"]:
            task_id = result["task_id"]
            outputs = result["outputs"]
            num_tests = result.get("num_test_pairs", 1)
            
            attempts = []
            for i in range(num_tests):
                attempt_1 = outputs[i * 2] if i * 2 < len(outputs) else []
                attempt_2 = outputs[i * 2 + 1] if i * 2 + 1 < len(outputs) else []
                attempts.append({
                    "attempt_1": attempt_1,
                    "attempt_2": attempt_2
                })
            
            submission[task_id] = attempts
    
    submission_file = config.attempts_dir / f"arc-agi_{split}_attempts.json"
    with open(submission_file, "w") as f:
        json.dump(submission, f)
    
    logger.info(f"Saved submission to {submission_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Solver")
    
    parser.add_argument(
        "--split", 
        default="evaluation",
        choices=["training", "evaluation"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Maximum number of tasks to solve"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset into task list"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Specific task ID to solve"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load tasks but don't solve (for testing)"
    )
    
    # Model selection
    parser.add_argument(
        "--reasoner",
        type=str,
        default=None,
        choices=["deepseek", "kimi", "qwen-reasoning", "gpt-oss-120b"],
        help="Reasoner model (default: kimi)"
    )
    parser.add_argument(
        "--coder",
        type=str,
        default=None,
        choices=["qwen-coder", "qwen-coder-small"],
        help="Coder model (default: qwen-coder)"
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local LLM for code generation"
    )
    parser.add_argument(
        "--local-url",
        type=str,
        default=None,
        help="Local LLM server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid mode: local exploration + remote refinement"
    )
    parser.add_argument(
        "--local-concurrency",
        type=int,
        default=64,
        help="Max concurrent requests to local LLM (default: 64)"
    )
    parser.add_argument(
        "--hybrid-candidates", 
        type=int,
        default=64,
        help="Number of candidates for hybrid local exploration (default: 64)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--no-feedback-loop",
        action="store_true",
        help="Disable feedback-to-local learning loop (default: enabled)"
    )
    parser.add_argument(
        "--local-timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds for local LLM requests (default: 600 = 10 min)"
    )
    parser.add_argument(
        "--initial-candidates",
        type=int,
        default=30,
        help="Number of initial code candidates to generate (default: 30)"
    )
    parser.add_argument(
        "--remote-concurrency",
        type=int,
        default=20,
        help="Max concurrent requests to remote API (default: 20)"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    from .config import AVAILABLE_MODELS
    
    args = parse_args()
    
    # List models mode
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 60)
        for key, name in AVAILABLE_MODELS.items():
            print(f"  {key:20s} -> {name}")
        print("-" * 60)
        return
    
    config = load_config(
        reasoner=args.reasoner,
        coder=args.coder,
        use_local=args.use_local,
        local_url=args.local_url,
        hybrid_mode=args.hybrid,
        local_concurrency=args.local_concurrency,
        hybrid_candidates=args.hybrid_candidates,
        feedback_to_local=not args.no_feedback_loop,  # Invert flag
        local_timeout=args.local_timeout
    )
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Using reasoner: {config.reasoner_model.name}")
    logger.info(f"Using coder: {config.coder_model.name}")
    if config.use_local:
        logger.info(f"Local LLM enabled at: {config.local_url}")
    if config.hybrid_mode:
        logger.info(f"HYBRID MODE: Local({config.hybrid_local_candidates}) -> Top({config.hybrid_top_k}) -> Remote refine")
    
    # Single task mode
    task_ids = [args.task_id] if args.task_id else None
    
    if args.dry_run:
        tasks = load_tasks(config.data_dir, args.split)
        logger.info(f"Dry run: would solve {len(tasks)} tasks")
        return
    
    # Run evaluation
    summary = await run_evaluation(
        config=config,
        split=args.split,
        limit=args.limit,
        offset=args.offset,
        task_ids=task_ids,
        verbose=args.verbose
    )
    
    # Save submission
    if summary["results"]:
        save_submission(config, summary["results"], args.split)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Tasks attempted: {summary['tasks_attempted']}")
    print(f"Tasks with at least 1 correct: {summary['tasks_with_correct']}")
    print(f"Total accuracy: {summary['accuracy']:.1%}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
