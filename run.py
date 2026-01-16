#!/usr/bin/env python3
"""Unified CLI entry point for ARC-AGI-2 Solvers.

Two solver modes:
  systems    - System 1/2 architecture (code + instructions)
  primitives - DSL primitives with VLM verification
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="ARC-AGI-2 Ensemble Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Solver Modes:
  systems     - System 1/2 architecture (Python code + English instructions)
  primitives  - DSL primitives with step-by-step VLM verification

Examples:
  # Systems solver (default)
  python run.py --solver systems --task-id a47bf94d -v
  
  # Primitives solver with ensemble planning
  python run.py --solver primitives --task-id 00d62c1b --ensemble --attempts 3
"""
    )
    
    parser.add_argument(
        "--solver",
        type=str,
        default="systems",
        choices=["systems", "primitives"],
        help="Solver to use: systems or primitives (default: systems)"
    )
    
    # Parse known args to get solver choice, pass rest to sub-solver
    args, remaining = parser.parse_known_args()
    
    if args.solver == "systems":
        # Import and run systems solver
        sys.argv = [sys.argv[0]] + remaining
        from systems.run import main as systems_main
        systems_main()
    else:
        # Import and run primitives solver
        sys.argv = [sys.argv[0]] + remaining
        import asyncio
        from run_primitives import main as primitives_main
        asyncio.run(primitives_main())


if __name__ == "__main__":
    main()
