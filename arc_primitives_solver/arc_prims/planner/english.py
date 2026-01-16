"""English planner for generating step-by-step strategy.

The planner analyzes ARC training examples and generates:
1. High-level goal description
2. Step-by-step English instructions
3. Expected visual outcomes for each step
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Type alias for Grid (matching arc_solver)
Grid = list[list[int]]


def grid_to_ascii(grid: Grid) -> str:
    """Convert grid to ASCII representation."""
    color_chars = "0123456789"
    lines = []
    for row in grid:
        lines.append("".join(color_chars[c] for c in row))
    return "\n".join(lines)


class EnglishPlanner:
    """Generate English strategy from training examples.
    
    The planner creates a detailed plan that the translator will
    convert to primitives. Each step should be specific enough
    to map to a single primitive operation.
    """
    
    def __init__(self, client, config):
        """Initialize planner.
        
        Args:
            client: LLM API client (ChutesClient from arc_solver)
            config: Configuration with model settings
        """
        self.client = client
        self.config = config
        self.model = config.reasoner_model
    
    async def plan(self, task: Any) -> str:
        """Generate step-by-step English instructions.
        
        Args:
            task: ARC Task with training examples
            
        Returns:
            Multi-line English plan with numbered steps
        """
        # Format training examples
        examples = self._format_examples(task)
        
        # Build planning prompt
        messages = self._build_prompt(examples)
        
        # Generate plan
        try:
            response = await self.client.chat(
                self.model,
                messages,
                temperature=0.4  # Low temperature for consistent planning
            )
            return self._parse_plan(response)
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return f"ERROR: {e}"
    
    def _format_examples(self, task: Any) -> str:
        """Format training examples for the prompt."""
        examples = []
        for i, pair in enumerate(task.train):
            examples.append(f"Example {i + 1}:")
            examples.append(f"Input ({len(pair.input)}x{len(pair.input[0])}):") 
            examples.append(grid_to_ascii(pair.input))
            examples.append(f"Output ({len(pair.output)}x{len(pair.output[0])}):")
            examples.append(grid_to_ascii(pair.output))
            examples.append("")
        return "\n".join(examples)
    
    def _build_prompt(self, examples: str) -> list[dict]:
        """Build the planning prompt."""
        return [
            {
                "role": "system",
                "content": """You are an expert at analyzing visual transformations in ARC puzzles.

Your task: Generate a DETAILED step-by-step plan to transform input grids to output grids.

IMPORTANT RULES:
1. Each step should describe ONE atomic operation
2. Use precise language that maps to these operations:
   - SELECT: "Select all [color] cells", "Find the largest object", "Identify connected regions"
   - TRANSFORM: "Rotate 90 degrees", "Flip horizontally", "Shift down by 2"
   - PAINT: "Fill with [color]", "Replace [color1] with [color2]", "Outline in [color]"
   - FILTER: "Keep only objects touching the border", "Remove small objects (area < 5)"
   - COMPOSITE: "Place on background", "Overlay on original grid"

3. Be SPECIFIC about:
   - Which objects/cells are affected
   - Exact colors (use numbers 0-9)
   - Exact positions or relative movements

4. Your plan must work for ALL examples, not just one

Output format:
GOAL: [one sentence describing the overall transformation]

STEP 1: [specific action]
STEP 2: [specific action]
...

Be thorough - missing a step means wrong output!"""
            },
            {
                "role": "user",
                "content": f"""Analyze these input/output examples and create a detailed transformation plan:

{examples}

Generate the step-by-step plan:"""
            }
        ]
    
    def _parse_plan(self, response: str) -> str:
        """Parse and validate the plan."""
        # Clean up the response
        lines = response.strip().split('\n')
        
        # Find GOAL line
        goal = ""
        steps = []
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith("GOAL:"):
                goal = line[5:].strip()
            elif line.upper().startswith("STEP"):
                # Extract step content after "STEP N:"
                if ":" in line:
                    step_content = line.split(":", 1)[1].strip()
                    steps.append(step_content)
        
        # Reconstruct clean plan
        if goal:
            result = f"GOAL: {goal}\n\n"
        else:
            result = ""
        
        for i, step in enumerate(steps, 1):
            result += f"STEP {i}: {step}\n"
        
        return result if result else response  # Return original if parsing failed


class PlanWithExamples:
    """Enhanced plan that includes example-specific guidance."""
    
    def __init__(self, goal: str, steps: list[str], examples: list[dict]):
        self.goal = goal
        self.steps = steps
        self.examples = examples  # [(input, expected_output) for each training example]
