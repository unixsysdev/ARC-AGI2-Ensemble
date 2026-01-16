"""Prompts for LLM interactions."""

from ..models.task import Grid, Task, Pair, grid_to_ascii, generate_grid_diff


class Prompts:
    """All prompts for the ARC solver."""
    
    @staticmethod
    def format_task_examples(task: Task) -> str:
        """Format training examples for prompts."""
        examples = []
        for i, pair in enumerate(task.train):
            examples.append(f"Example {i + 1}:")
            examples.append(f"Input ({len(pair.input)}x{len(pair.input[0])}):")
            examples.append(grid_to_ascii(pair.input))
            examples.append(f"Output ({len(pair.output)}x{len(pair.output[0])}):")
            examples.append(grid_to_ascii(pair.output))
            examples.append("")
        return "\n".join(examples)
    
    @staticmethod
    def code_generation(task: Task) -> list[dict]:
        """Prompt for generating Python solver code."""
        examples = Prompts.format_task_examples(task)
        
        return [
            {
                "role": "system",
                "content": """You are an expert Python programmer solving ARC-AGI puzzles.

Your task: Write a Python function that transforms input grids to output grids.

Rules:
1. The function signature MUST be: def transform(grid: list[list[int]]) -> list[list[int]]
2. Use numpy if helpful (it's imported as np)
3. The function must work for ANY valid input, not just the examples
4. Return ONLY the Python code, no explanations
5. Handle edge cases gracefully

CRITICAL: First analyze the pattern, then write concise, correct code."""
            },
            {
                "role": "user",
                "content": f"""Analyze these input/output examples and write a transform function:

{examples}

Write the Python function that transforms inputs to outputs.
Think step by step about what transformation is happening, then write the code.

```python
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    # Your code here
```"""
            }
        ]
    
    @staticmethod
    def instruction_generation(task: Task) -> list[dict]:
        """Prompt for generating English instructions."""
        examples = Prompts.format_task_examples(task)
        
        return [
            {
                "role": "system",
                "content": """You are an expert at describing visual transformations in precise English.

Your task: Describe the transformation rule that converts input grids to output grids.

Rules:
1. Be PRECISE and UNAMBIGUOUS
2. Describe the GENERAL rule, not specific to one example
3. Use terms like: rotate, flip, mirror, shift, fill, copy, delete, expand, shrink
4. Reference colors by their numbers (0-9)
5. Reference positions as (row, column) with 0-indexed coordinates
6. Your instructions must be detailed enough for someone else to follow exactly

Output format:
STEP 1: [First action]
STEP 2: [Second action]
...

Be thorough - missing a step means wrong output."""
            },
            {
                "role": "user",
                "content": f"""Analyze these input/output examples and describe the transformation rule:

{examples}

Describe the complete transformation as numbered steps."""
            }
        ]
    
    @staticmethod
    def instruction_execution(instruction: str, input_grid: Grid) -> list[dict]:
        """Prompt for executing English instructions on a grid."""
        grid_str = grid_to_ascii(input_grid)
        h, w = len(input_grid), len(input_grid[0]) if input_grid else 0
        
        return [
            {
                "role": "system",
                "content": """You are a precise grid transformation executor.

Given instructions and an input grid, you must produce the exact output grid.

Rules:
1. Follow the instructions EXACTLY step by step
2. Output ONLY the final grid as JSON: {"grid": [[...]]}
3. Each cell must be an integer 0-9
4. Double-check dimensions and values

Be meticulous - every cell matters."""
            },
            {
                "role": "user",
                "content": f"""Follow these instructions to transform the input grid:

INSTRUCTIONS:
{instruction}

INPUT GRID ({h}x{w}):
{grid_str}

Apply the instructions and output the result as JSON: {{"grid": [[...]]}}"""
            }
        ]
    
    @staticmethod
    def individual_revision(
        instruction: str,
        errors: list[tuple[Grid, Grid, Grid]]  # (input, expected, actual)
    ) -> list[dict]:
        """Prompt for revising instructions based on errors."""
        error_details = []
        for i, (inp, exp, act) in enumerate(errors):
            diff = generate_grid_diff(exp, act)
            error_details.append(f"Error {i + 1}:\nInput:\n{grid_to_ascii(inp)}\n\n{diff}")
        
        return [
            {
                "role": "system",
                "content": """You are refining transformation instructions based on errors.

Analyze where the current instructions fail and produce IMPROVED instructions.

Focus on:
1. What step is missing or wrong?
2. What edge case wasn't handled?
3. What was misinterpreted?

Output the COMPLETE revised instructions, not just changes."""
            },
            {
                "role": "user",
                "content": f"""The current instructions have errors. Fix them.

CURRENT INSTRUCTIONS:
{instruction}

ERRORS (expected vs actual):
{chr(10).join(error_details)}

Write the COMPLETE revised instructions:"""
            }
        ]
    
    @staticmethod
    def pooled_revision(
        scored_instructions: list[tuple[str, float]]  # (instruction, score)
    ) -> list[dict]:
        """Prompt for synthesizing from multiple instructions."""
        instructions_text = []
        for i, (inst, score) in enumerate(scored_instructions):
            instructions_text.append(f"Instruction {i + 1} (score: {score:.2f}):\n{inst}")
        
        return [
            {
                "role": "system",
                "content": """You are synthesizing the best instruction from multiple candidates.

Analyze what each instruction got right and wrong, then create a SUPERIOR instruction
that combines the best insights from all of them.

Output the COMPLETE new instruction."""
            },
            {
                "role": "user",
                "content": f"""Synthesize a better instruction from these candidates:

{chr(10).join(instructions_text)}

Write a NEW instruction that combines the best parts:"""
            }
        ]
    
    @staticmethod
    def vlm_verification() -> str:
        """Prompt for VLM to verify grid transformation."""
        return """Look at this ARC puzzle transformation.

The LEFT side shows the INPUT grid.
The RIGHT side shows a CANDIDATE OUTPUT grid.

Evaluate if the transformation looks correct:
1. Does the output follow logical patterns from the input?
2. Are there any obvious visual errors (wrong symmetry, missing objects, extra noise)?
3. Does the transformation look intentional or random?

Reply in this format:
VALID: [reason] 
or
INVALID: [specific problem]

Be strict - only say VALID if you're confident."""
    
    @staticmethod
    def code_revision(code: str, error: str, task: Task) -> list[dict]:
        """Prompt for fixing Python code based on errors."""
        examples = Prompts.format_task_examples(task)
        
        return [
            {
                "role": "system",
                "content": """You are debugging Python code for ARC-AGI puzzles.

Analyze the error and fix the code. Common issues:
- Index out of bounds (check grid dimensions)
- Wrong loop direction
- Off-by-one errors
- Missing edge cases

Output ONLY the corrected Python code."""
            },
            {
                "role": "user",
                "content": f"""Fix this code that failed:

TASK EXAMPLES:
{examples}

CURRENT CODE:
```python
{code}
```

ERROR:
{error}

Write the FIXED code:"""
            }
        ]
