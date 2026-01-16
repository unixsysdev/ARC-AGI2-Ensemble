"""Ensemble Planner: Dual-path VLM + LLM planning with synthesis.

This planner gets TWO perspectives on the ARC task:
1. VLM (visual): Analyzes grid images for spatial patterns
2. LLM (symbolic): Analyzes grid data for logical patterns

Then synthesizes them into a single unified plan.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any

from ..config import Config

logger = logging.getLogger(__name__)


class EnsemblePlanner:
    """Dual-path planner combining VLM visual + LLM symbolic analysis.
    
    Architecture:
    ┌─────────────────┐     ┌──────────────────┐
    │  VLM (visual)   │     │  LLM (symbolic)  │
    │  "I see shapes  │     │  "Grid has 3s,   │
    │   with holes"   │     │   find 0s inside"│
    └────────┬────────┘     └────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
                  ┌──────────────┐
                  │  Synthesizer │
                  └──────────────┘
    """
    
    SYMBOLIC_PROMPT = """Analyze this ARC-AGI puzzle using LOGICAL reasoning (no images).

INPUT GRID (2D array, colors 0-9):
{input_grid}

OUTPUT GRID (expected result):
{output_grid}

COLOR KEY: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=pink, 7=orange, 8=cyan, 9=brown

## ANALYZE SYMBOLICALLY:
1. What colors appear in input vs output?
2. What cells CHANGE? What cells STAY THE SAME?
3. Is there a pattern like: "all X become Y" or "cells surrounded by X become Y"?
4. Count objects, check for symmetry, look for enclosed regions.

## GENERATE DSL PROGRAM using these primitives:

```
# Selection
select(criteria="color", value=N)           # Select cells of color N
select(criteria="connected")                 # Find connected components
select(criteria="enclosed", enclosing_color=N)  # Regions enclosed by color N

# Painting
paint(color=N)                               # Paint selected cells
replace(source_color=A, target_color=B)      # Replace A with B everywhere

# Flood Fill (CRITICAL: always specify target_color!)
flood_fill(color=N, start_position="border", target_color=0)

# Transformations
transform(action="rotate_90"|"flip_horizontal"|etc)
gravity(direction="down"|"up"|"left"|"right")
```

## OUTPUT FORMAT:

### Analysis
[Your logical analysis - 2-3 sentences]

### DSL Program
```dsl
1. function_call_here(param=value)
2. next_function(param=value)
```

CRITICAL: For flood_fill, ALWAYS include target_color parameter!
"""

    # Meta-Reviewer prompt: SELECT the best expert, don't just merge
    SELECTION_PROMPT = """You are the LEAD ARCHITECT solving an ARC-AGI puzzle. 

You have TWO CANDIDATE SOLUTIONS from different experts:

═══════════════════════════════════════════════════════════════════════════════
## CANDIDATE A: VISUAL EXPERT (VLM)
*Strengths: Seeing topology, holes, enclosed regions, gravity, adjacency, shapes*
{visual_plan}
═══════════════════════════════════════════════════════════════════════════════
## CANDIDATE B: SYMBOLIC EXPERT (LLM)  
*Strengths: Counting, arithmetic, sorting, exact color codes, coordinate math*
{symbolic_plan}
═══════════════════════════════════════════════════════════════════════════════

## DECISION PROTOCOL - Choose which expert to trust:

**TRUST VISUAL (A) for:**
- "Fill holes", "Enclosed areas", "Inside/Outside" → VLM sees topology
- "Gravity", "Falling objects" → VLM sees physics layout
- "Connected components", "Touching/Adjacent" → VLM sees relationships
- flood_fill operations → VLM understands what's "inside"

**TRUST SYMBOLIC (B) for:**
- Counting ("3 red pixels"), Sequences, Arithmetic
- Exact color mapping ("Replace 1 with 6") → VLM often hallucinates colors
- Coordinate math ("Move 3 right", "Every 2nd row")
- Sorting by size, filtering by count

**HYBRID FIX:** If task is topological (A's domain) but A has wrong colors:
→ Use A's STRUCTURE/ALGORITHM + B's COLOR CODES

## YOUR OUTPUT:

### Expert Selection
[Which expert (A/B) is better for this task and why - 1-2 sentences]

### Final DSL Program
```dsl
1. function_call(params)
2. next_function(params)
```

CRITICAL: 
- For flood_fill, ALWAYS include target_color (usually 0)
- Max 5 steps
- Use EXACT function syntax from the candidates
"""

    def __init__(self, client: Any, config: Config):
        """Initialize ensemble planner."""
        self.client = client
        self.config = config
        
        # Import component planners
        from .visual import VisualPlanner
        from .english import EnglishPlanner
        
        self.visual_planner = VisualPlanner(client, config)
        self.text_planner = EnglishPlanner(client, config)
        
        self.vlm_model = config.vlm_model
        self.coder_model = config.coder_model
    
    async def generate_plan(
        self, 
        task: Any, 
        previous_feedback: list[str] = None
    ) -> str:
        """Generate plan using both VLM and LLM in parallel, then synthesize.
        
        Args:
            task: ARC Task with training examples
            previous_feedback: Optional failures from previous attempt
            
        Returns:
            Synthesized plan string
        """
        logger.info("[ENSEMBLE] Dual-path planning: VLM visual + LLM symbolic")
        
        # Run both analyses in parallel - BOTH now receive feedback
        visual_task = self._get_visual_analysis(task, previous_feedback)
        symbolic_task = self._get_symbolic_analysis(task, previous_feedback)
        
        try:
            visual_plan, symbolic_plan = await asyncio.gather(
                visual_task,
                symbolic_task,
                return_exceptions=True
            )
        except Exception as e:
            logger.warning(f"Parallel analysis failed: {e}")
            # Fallback to visual only
            return await self.visual_planner.generate_plan(task, previous_feedback)
        
        # Handle individual failures
        if isinstance(visual_plan, Exception):
            logger.warning(f"VLM analysis failed: {visual_plan}")
            visual_plan = "[VLM analysis unavailable]"
        
        if isinstance(symbolic_plan, Exception):
            logger.warning(f"LLM analysis failed: {symbolic_plan}")
            symbolic_plan = "[Symbolic analysis unavailable]"
        
        logger.info(f"[ENSEMBLE] VLM DSL: {len(str(visual_plan))} chars")
        logger.info(f"[ENSEMBLE] LLM DSL: {len(str(symbolic_plan))} chars")
        
        # Meta-Reviewer selects the best expert (not just merge)
        unified_plan = await self._select_best_expert(visual_plan, symbolic_plan)
        
        return unified_plan
    
    async def _get_visual_analysis(
        self, 
        task: Any, 
        previous_feedback: list[str] = None
    ) -> str:
        """Get VLM visual analysis."""
        return await self.visual_planner.generate_plan(task, previous_feedback)
    
    async def _get_symbolic_analysis(
        self, 
        task: Any,
        previous_feedback: list[str] = None
    ) -> str:
        """Get LLM symbolic analysis from grid data.
        
        Now also receives feedback from previous attempts to learn from failures.
        """
        # Use first training example
        pair = task.train[0]
        
        # Format grids as readable arrays
        input_str = self._format_grid(pair.input)
        output_str = self._format_grid(pair.output)
        
        prompt = self.SYMBOLIC_PROMPT.format(
            input_grid=input_str,
            output_grid=output_str
        )
        
        # Add feedback from previous failures (same as VLM gets)
        if previous_feedback:
            feedback_text = "\n".join(previous_feedback[:3])  # Limit to top 3
            prompt += f"""

IMPORTANT - PREVIOUS ATTEMPT FAILED:
{feedback_text}

Learn from these failures and try a DIFFERENT approach. The previous solution was WRONG.
"""
            logger.info(f"[LLM SYMBOLIC] Including feedback from {len(previous_feedback)} failures")
        
        messages = [
            {"role": "system", "content": "You are an expert at ARC-AGI puzzles. Analyze grids logically."},
            {"role": "user", "content": prompt}
        ]
        
        logger.info(f"[LLM SYMBOLIC] Model: {self.coder_model.name}")
        
        response = await self.client.chat(
            self.coder_model,
            messages,
            temperature=0.3
        )
        
        logger.info(f"[LLM SYMBOLIC] Response: {len(response)} chars")
        return response
    
    async def _select_best_expert(self, visual_plan: str, symbolic_plan: str) -> str:
        """Select the best expert solution (not just merge).
        
        The Meta-Reviewer acts as a Judge, choosing:
        - Visual Expert (VLM) for topology, holes, gravity
        - Symbolic Expert (LLM) for counting, math, color codes
        - Or a HYBRID fix using VLM structure + LLM colors
        """
        prompt = self.SELECTION_PROMPT.format(
            visual_plan=visual_plan,
            symbolic_plan=symbolic_plan
        )
        
        messages = [
            {"role": "system", "content": "You are a Lead Architect selecting the best ARC solution from two experts."},
            {"role": "user", "content": prompt}
        ]
        
        logger.info(f"[META-REVIEWER] Model: {self.coder_model.name}")
        logger.info(f"[META-REVIEWER] Selecting best expert (Visual vs Symbolic)...")
        
        try:
            response = await self.client.chat(
                self.coder_model,
                messages,
                temperature=0.1  # Low temp for strict selection
            )
            logger.info(f"[META-REVIEWER] Selection complete: {len(response)} chars")
            return response
        except Exception as e:
            logger.warning(f"[META-REVIEWER] Selection failed: {e}, using visual plan")
            return visual_plan
    
    def _format_grid(self, grid: list[list[int]]) -> str:
        """Format grid as readable string."""
        lines = []
        for row in grid:
            lines.append("[" + ", ".join(str(c) for c in row) + "]")
        return "[\n  " + ",\n  ".join(lines) + "\n]"
