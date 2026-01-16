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
1. Compare INPUT SIZE vs OUTPUT SIZE - if different, we need extract()!
2. What colors appear in input vs output?
3. What cells CHANGE? What cells STAY THE SAME?
4. Is there a pattern like: "all X become Y" or "cells surrounded by X become Y"?
5. Count objects, check for symmetry, look for enclosed regions.

## GENERATE DSL PROGRAM using these primitives:

```
# Selection (mode: set=replace, intersect=keep overlap, union=add)
select(criteria="color", value=N)           # Select cells of color N
select(criteria="connected")                 # Find connected components
select(criteria="largest")                   # Select largest object
select(criteria="smallest")                  # Select smallest object
select(criteria="size_rank", value=N)        # By size rank (0=smallest, -1=largest)
select(criteria="color", value=N, mode="intersect")  # Keep only cells that were already selected
select(criteria="enclosed", enclosing_color=N)  # Regions enclosed by color N

# Painting
paint(color=N)                               # Paint selected cells
replace(source_color=A, target_color=B)      # Replace A with B everywhere

# Flood Fill (CRITICAL: always specify target_color!)
flood_fill(color=N, start_position="border", target_color=0)

# Filters (refine selection to ONE object)
filter(condition="area_eq", value=9)         # Keep objects with area=9 (e.g., 3x3)
filter(condition="has_colors", value=[1,3,4]) # Keep objects containing ALL these colors

# Extraction (USE WHEN OUTPUT IS SMALLER THAN INPUT!)
extract()                                    # Crop grid to selection bounding box

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

CRITICAL: 
- If OUTPUT SIZE != INPUT SIZE → filter to ONE object, then extract()!
- ⚠️ extract() crops to bounding box of ALL selections - filter to ONE first!
- ⚠️ Each select() REPLACES previous selection!
- For flood_fill, ALWAYS include target_color parameter!
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
- SIZE CHANGES (output smaller than input) → use extract()

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
- If OUTPUT SIZE != INPUT SIZE → use extract()!
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
        vlm_feedback: list[str] = None,
        llm_feedback: list[str] = None
    ) -> str:
        """Generate plan using both VLM and LLM in parallel, then synthesize.
        
        Args:
            task: ARC Task with training examples
            vlm_feedback: Feedback for VLM planner (None = no feedback)
            llm_feedback: Feedback for LLM planner (None = no feedback)
            
        Returns:
            Synthesized plan string
        """
        logger.info("[ENSEMBLE] Dual-path planning: VLM visual + LLM symbolic")
        
        # Detect expected output size from training examples
        expected_size = self._get_expected_output_size(task)
        input_size = self._get_input_size(task)
        needs_extract = expected_size and input_size and (
            expected_size[0] < input_size[0] or expected_size[1] < input_size[1]
        )
        
        if needs_extract:
            logger.info(f"[ENSEMBLE] ⚠️ Output {expected_size} < Input {input_size} → plans MUST use extract()!")
        
        # Run both analyses in parallel with SEPARATE feedback
        visual_task = self._get_visual_analysis(task, vlm_feedback)
        symbolic_task = self._get_symbolic_analysis(task, llm_feedback)
        
        try:
            visual_plan, symbolic_plan = await asyncio.gather(
                visual_task,
                symbolic_task,
                return_exceptions=True
            )
        except Exception as e:
            logger.warning(f"Parallel analysis failed: {e}")
            # Fallback to visual only
            return await self.visual_planner.generate_plan(task, vlm_feedback)
        
        # Handle individual failures
        if isinstance(visual_plan, Exception):
            logger.warning(f"VLM analysis failed: {visual_plan}")
            visual_plan = "[VLM analysis unavailable]"
        
        if isinstance(symbolic_plan, Exception):
            logger.warning(f"LLM analysis failed: {symbolic_plan}")
            symbolic_plan = "[Symbolic analysis unavailable]"
        
        logger.info(f"[ENSEMBLE] VLM DSL: {len(str(visual_plan))} chars")
        logger.info(f"[ENSEMBLE] LLM DSL: {len(str(symbolic_plan))} chars")
        
        # Meta-Reviewer selects the best expert (with size constraint awareness)
        unified_plan = await self._select_best_expert(
            visual_plan, symbolic_plan, expected_size, needs_extract
        )
        
        return unified_plan
    
    def _get_expected_output_size(self, task: Any) -> tuple[int, int] | None:
        """Get expected output size from training examples (if consistent)."""
        if not task.train:
            return None
        
        sizes = []
        for pair in task.train:
            h, w = len(pair.output), len(pair.output[0]) if pair.output else 0
            sizes.append((h, w))
        
        # Check if all training outputs have same size
        if all(s == sizes[0] for s in sizes):
            return sizes[0]
        return None  # Inconsistent sizes
    
    def _get_input_size(self, task: Any) -> tuple[int, int] | None:
        """Get input size from first training example."""
        if not task.train:
            return None
        first_input = task.train[0].input
        return (len(first_input), len(first_input[0]) if first_input else 0)
    
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
    
    async def _select_best_expert(
        self, 
        visual_plan: str, 
        symbolic_plan: str,
        expected_size: tuple[int, int] = None,
        needs_extract: bool = False
    ) -> str:
        """Select the best expert solution (not just merge).
        
        The Meta-Reviewer acts as a Judge, choosing:
        - Visual Expert (VLM) for topology, holes, gravity
        - Symbolic Expert (LLM) for counting, math, color codes
        - Or a HYBRID fix using VLM structure + LLM colors
        
        Also validates that plans use extract() when output size is smaller.
        """
        # Pre-check: If output needs to be smaller, validate plans have extract()
        size_warning = ""
        if needs_extract:
            vlm_has_extract = "extract()" in visual_plan.lower()
            llm_has_extract = "extract()" in symbolic_plan.lower()
            
            if not vlm_has_extract:
                logger.warning("[META-REVIEWER] ⚠️ VLM plan MISSING extract() but output should be smaller!")
            if not llm_has_extract:
                logger.warning("[META-REVIEWER] ⚠️ LLM plan MISSING extract() but output should be smaller!")
            
            size_warning = f"""

⚠️ CRITICAL SIZE CONSTRAINT: 
The expected output is {expected_size[0]}x{expected_size[1]} (SMALLER than input).
Any valid plan MUST end with extract() to crop the output!
If a candidate doesn't use extract(), REJECT IT or add extract() at the end.
"""
        
        prompt = self.SELECTION_PROMPT.format(
            visual_plan=visual_plan,
            symbolic_plan=symbolic_plan
        )
        
        # Inject size warning into prompt
        if size_warning:
            prompt = prompt + size_warning
        
        messages = [
            {"role": "system", "content": "You are a Lead Architect selecting the best ARC solution from two experts."},
            {"role": "user", "content": prompt}
        ]
        
        logger.info(f"[META-REVIEWER] Model: {self.coder_model.name}")
        logger.info(f"[META-REVIEWER] Selecting best expert (Visual vs Symbolic)...")
        if needs_extract:
            logger.info(f"[META-REVIEWER] Size constraint: output must be {expected_size}")
        
        try:
            response = await self.client.chat(
                self.coder_model,
                messages,
                temperature=0.1  # Low temp for strict selection
            )
            logger.info(f"[META-REVIEWER] Selection complete: {len(response)} chars")
            
            # Post-check: If extract was needed but not in response, append warning
            if needs_extract and "extract()" not in response.lower():
                logger.warning("[META-REVIEWER] ⚠️ Final plan MISSING extract(), appending it!")
                # Find DSL block and append extract()
                if "```dsl" in response:
                    response = response.replace("```\n\nCRITICAL", "extract()\n```\n\nCRITICAL")
                    response = response.rstrip("```") + "\nextract()\n```"
            
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
