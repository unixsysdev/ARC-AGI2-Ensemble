#!/usr/bin/env python3
"""CLI entry point for primitives-based ARC solver."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add arc_solver to path for imports (MUST be first, before any local imports)
ARC_SOLVER_PATH = Path(__file__).parent.parent / "arc_solver"
sys.path.insert(0, str(ARC_SOLVER_PATH))

# Now import from arc_solver (uses their src/)
from src.llms.chutes_client import ChutesClient
from src.models.task import Task

# Add our package to path
sys.path.insert(0, str(Path(__file__).parent))

# Now import our local modules (uses arc_prims/, no collision)
from arc_prims.config import load_config, MODEL_PRESETS
from arc_prims.solver import PrimitivesSolver


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Primitives-based ARC-AGI-2 Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Presets:
  fast      - Smaller models, faster responses (Qwen-32B)
  balanced  - Good balance of speed and quality (Kimi + Qwen-30B)
  quality   - Best quality, slower (DeepSeek + Qwen-480B)

Examples:
  python run.py --task-id 00d62c1b --preset fast
  python run.py --task-id 00d62c1b --preset quality --visual-planning
"""
    )
    
    # Task selection
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="Task ID to solve (e.g., 00d62c1b)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "evaluation"],
        help="Dataset split"
    )
    parser.add_argument(
        "--test-index",
        type=int,
        default=0,
        help="Which test case to solve (default: 0)"
    )
    
    # Model options
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset: fast, balanced, or quality"
    )
    parser.add_argument(
        "--visual-planning",
        action="store_true",
        help="Use VLM with rendered images for planning (better pattern recognition)"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ENSEMBLE planner (VLM visual + LLM symbolic dual-path)"
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable VLM verification (faster but less accurate)"
    )
    
    # Solver options
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Maximum solve attempts"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for solution (JSON)"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger("run")
    logger.info("Primitives-based ARC Solver")
    logger.info(f"Task: {args.task_id} ({args.split})")
    logger.info(f"Model preset: {args.preset}")
    
    # Load configuration with preset
    config = load_config(preset=args.preset)
    
    if args.no_vlm:
        config.use_vlm_verification = False
        logger.info("VLM verification disabled")
    
    if args.ensemble:
        config.use_ensemble_planning = True
        logger.info("ENSEMBLE planning enabled (VLM visual + LLM symbolic)")
    elif args.visual_planning:
        config.use_visual_planning = True
        logger.info("Visual planning enabled (VLM will analyze grid images)")
    else:
        config.use_visual_planning = False
    
    # Load task
    task_path = config.data_dir / args.split / f"{args.task_id}.json"
    if not task_path.exists():
        logger.error(f"Task not found: {task_path}")
        sys.exit(1)
    
    task = Task.from_json(task_path)
    logger.info(f"Loaded task with {len(task.train)} training examples")
    
    # Solve
    async with ChutesClient(config) as client:
        solver = PrimitivesSolver(client, config)
        candidates = await solver.solve_with_retry(
            task,
            test_index=args.test_index,
            max_attempts=args.attempts
        )
    
    if candidates:
        logger.info(f"Generated {len(candidates)} candidate(s)")
        
        # Print first candidate
        print("\nBest candidate solution:")
        for row in candidates[0]:
            print("".join(str(c) for c in row))
        
        # Save if output specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump({
                    "task_id": args.task_id,
                    "candidates": candidates
                }, f, indent=2)
            logger.info(f"Saved to {args.output}")
    else:
        logger.warning("No candidates generated")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
