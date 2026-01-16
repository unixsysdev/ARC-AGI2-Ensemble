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

## GENERATE PLAN USING ONLY THESE PRIMITIVES:
- SELECT all [color] cells
- SELECT connected components
- SELECT the largest/smallest object
- PAINT with [color]
- REPLACE [color1] with [color2]
- FLOOD_FILL from border with [color] (marks exterior cells)
- FLOOD_FILL from position (x,y) with [color]
- FILTER keep only cells touching border
- ROTATE 90/180/270 degrees
- FLIP horizontal/vertical
- GRAVITY down/up/left/right

IMPORTANT: Use ONLY the primitives listed above. Do NOT invent new ones.

FORMAT:
## Analysis
[Your logical analysis]

## Steps
STEP 1: [primitive from list above]
STEP 2: [primitive from list above]
...
"""

    SYNTHESIS_PROMPT = """You have TWO analyses of the same ARC puzzle:

## VISUAL ANALYSIS (from VLM seeing the grids as images):
{visual_plan}

## SYMBOLIC ANALYSIS (from LLM analyzing the grid numbers):
{symbolic_plan}

SYNTHESIZE these into ONE unified plan using ONLY these primitives:
- SELECT all [color] cells
- SELECT connected components  
- SELECT the largest/smallest object
- PAINT with [color]
- REPLACE [color1] with [color2]
- FLOOD_FILL from border with [color]
- FILTER keep only cells touching border
- ROTATE/FLIP/GRAVITY

Do NOT use any other primitives. Do NOT invent primitives like "SET DIFFERENCE".

OUTPUT FORMAT:
## Unified Plan
STEP 1: [primitive]
STEP 2: [primitive]
...

Keep it to MAX 5 steps.
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
        logger.info("Ensemble planning: VLM + LLM dual-path")
        
        # Run both analyses in parallel
        visual_task = self._get_visual_analysis(task, previous_feedback)
        symbolic_task = self._get_symbolic_analysis(task)
        
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
        
        logger.debug(f"Visual plan: {len(str(visual_plan))} chars")
        logger.debug(f"Symbolic plan: {len(str(symbolic_plan))} chars")
        
        # Synthesize the two analyses
        unified_plan = await self._synthesize(visual_plan, symbolic_plan)
        
        return unified_plan
    
    async def _get_visual_analysis(
        self, 
        task: Any, 
        previous_feedback: list[str] = None
    ) -> str:
        """Get VLM visual analysis."""
        return await self.visual_planner.generate_plan(task, previous_feedback)
    
    async def _get_symbolic_analysis(self, task: Any) -> str:
        """Get LLM symbolic analysis from grid data."""
        # Use first training example
        pair = task.train[0]
        
        # Format grids as readable arrays
        input_str = self._format_grid(pair.input)
        output_str = self._format_grid(pair.output)
        
        prompt = self.SYMBOLIC_PROMPT.format(
            input_grid=input_str,
            output_grid=output_str
        )
        
        messages = [
            {"role": "system", "content": "You are an expert at ARC-AGI puzzles. Analyze grids logically."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat(
            self.coder_model,
            messages,
            temperature=0.3
        )
        
        return response
    
    async def _synthesize(self, visual_plan: str, symbolic_plan: str) -> str:
        """Synthesize visual and symbolic analyses into unified plan."""
        prompt = self.SYNTHESIS_PROMPT.format(
            visual_plan=visual_plan,
            symbolic_plan=symbolic_plan
        )
        
        messages = [
            {"role": "system", "content": "You synthesize multiple ARC puzzle analyses into one plan."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.client.chat(
                self.coder_model,
                messages,
                temperature=0.2
            )
            logger.info("Ensemble synthesis complete")
            return response
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}, using visual plan")
            return visual_plan
    
    def _format_grid(self, grid: list[list[int]]) -> str:
        """Format grid as readable string."""
        lines = []
        for row in grid:
            lines.append("[" + ", ".join(str(c) for c in row) + "]")
        return "[\n  " + ",\n  ".join(lines) + "\n]"
