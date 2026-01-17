"""Free-form DSL Interpreter.

Converts natural language arguments in DSL primitives to structured parameters.
Example: select("the small colorful object") → select(criteria="unique", value="colors")
"""

from __future__ import annotations
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class FreeFormInterpreter:
    """Interpret natural language DSL arguments into structured primitives."""
    
    INTERPRETATION_PROMPT = """You are a DSL interpreter. Convert natural language arguments to structured parameters.

INPUT: A DSL command with natural language argument
OUTPUT: The same command with structured parameters

AVAILABLE STRUCTURED PARAMETERS:

For select():
- select(criteria="color", value=N)        # Select cells of color N (0-9)
- select(criteria="connected")              # Find all connected components
- select(criteria="largest")                # Select largest object
- select(criteria="smallest")               # Select smallest object  
- select(criteria="unique", value="colors") # Select object with unique color set
- select(criteria="unique", value="size")   # Select object with unique size
- select(criteria="size_rank", value=N)     # By size rank (0=smallest, -1=largest)

For filter():
- filter(condition="area_eq", value=N)      # Keep objects with area=N
- filter(condition="area_lt", value=N)      # Keep objects smaller than N cells
- filter(condition="has_colors", value=[...]) # Keep objects with ALL these colors
- filter(condition="touches_border")        # Keep objects touching edge

EXAMPLES:
- select("the small colorful object") → select(criteria="unique", value="colors")
- select("the largest one") → select(criteria="largest")
- select("objects with blue and red") → filter(condition="has_colors", value=[1, 2])
- filter("keep only the smallest") → select(criteria="smallest")
- select("the one that's different from the noise") → select(criteria="unique", value="colors")

Now interpret this command:
{command}

Respond with ONLY the structured command, nothing else."""

    def __init__(self, client: Any, config: Any):
        """Initialize interpreter."""
        self.client = client
        self.config = config
        self.model = config.coder_model
    
    async def interpret(self, plan: str) -> str:
        """Interpret a plan with natural language arguments.
        
        Args:
            plan: DSL plan potentially containing natural language arguments
            
        Returns:
            Plan with structured arguments
        """
        lines = plan.split('\n')
        interpreted_lines = []
        
        for line in lines:
            # Check if line contains a free-form argument (string in quotes after primitive)
            if self._has_freeform_arg(line):
                interpreted = await self._interpret_line(line)
                interpreted_lines.append(interpreted)
                logger.info(f"[INTERPRETER] {line.strip()} → {interpreted.strip()}")
            else:
                interpreted_lines.append(line)
        
        return '\n'.join(interpreted_lines)
    
    def _has_freeform_arg(self, line: str) -> bool:
        """Check if line contains a free-form natural language argument."""
        # Match patterns like: select("some natural language description")
        # But NOT: select(criteria="color", value=1)
        pattern = r'(select|filter)\s*\(\s*"[^"]{10,}"'  # 10+ chars = likely natural language
        return bool(re.search(pattern, line))
    
    async def _interpret_line(self, line: str) -> str:
        """Interpret a single line with free-form argument."""
        try:
            messages = [
                {"role": "system", "content": "You are a precise DSL interpreter. Output only the structured command."},
                {"role": "user", "content": self.INTERPRETATION_PROMPT.format(command=line)}
            ]
            
            response = await self.client.chat(
                self.model,
                messages,
                temperature=0.1
            )
            
            # Extract just the command from response
            response = response.strip()
            # If response contains multiple lines, take the one with select/filter
            for resp_line in response.split('\n'):
                if 'select(' in resp_line or 'filter(' in resp_line:
                    return resp_line.strip()
            
            return response
            
        except Exception as e:
            logger.warning(f"[INTERPRETER] Failed to interpret: {e}, returning original")
            return line
