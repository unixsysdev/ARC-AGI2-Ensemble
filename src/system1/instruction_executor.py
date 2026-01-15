"""LLM-based instruction executor for ARC tasks."""

from __future__ import annotations
import json
import logging
from typing import Any

from ..config import Config
from ..llms.chutes_client import ChutesClient
from ..llms.prompts import Prompts
from ..models.task import Grid

logger = logging.getLogger(__name__)


class InstructionExecutor:
    """Executes English instructions on grids using LLM as computer."""
    
    def __init__(self, client: ChutesClient, config: Config):
        self.client = client
        self.config = config
    
    async def execute(self, instruction: str, input_grid: Grid) -> tuple[Grid | None, str | None]:
        """
        Execute instruction on input grid.
        
        Returns:
            (result_grid, error_message) - one will be None
        """
        messages = Prompts.instruction_execution(instruction, input_grid)
        
        try:
            response = await self.client.chat(
                self.config.reasoner_model,
                messages,
                temperature=0.2,  # Low temperature for precision
                max_tokens=4096
            )
            
            # Parse the grid from response
            grid = self._parse_grid(response)
            
            if grid is None:
                return None, f"Could not parse grid from response: {response[:200]}"
            
            return grid, None
            
        except Exception as e:
            return None, f"Execution failed: {e}"
    
    def _parse_grid(self, response: str) -> Grid | None:
        """Parse grid from LLM response."""
        # Try JSON parsing first
        parsed = self.client.parse_json_response(response)
        
        if isinstance(parsed, dict) and "grid" in parsed:
            return self._validate_grid(parsed["grid"])
        
        if isinstance(parsed, list):
            return self._validate_grid(parsed)
        
        # Try to find grid-like structure in text
        lines = response.strip().split("\n")
        grid_lines = []
        
        in_grid = False
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and labels
            if not line or line.startswith(("Output", "Result", "GRID", "{")):
                continue
            
            # Check if line looks like a grid row
            if all(c in "0123456789 ,[]" for c in line):
                # Clean and parse
                cleaned = line.strip("[], ")
                if cleaned:
                    try:
                        if "," in cleaned:
                            row = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
                        else:
                            row = [int(c) for c in cleaned if c.isdigit()]
                        
                        if row:
                            grid_lines.append(row)
                            in_grid = True
                    except ValueError:
                        continue
            elif in_grid:
                # Stop when we hit non-grid content after grid
                break
        
        if grid_lines:
            return self._validate_grid(grid_lines)
        
        return None
    
    def _validate_grid(self, grid: Any) -> Grid | None:
        """Validate grid structure and values."""
        if not isinstance(grid, list) or not grid:
            return None
        
        # Check all rows are lists
        if not all(isinstance(row, list) for row in grid):
            return None
        
        # Check all values are valid
        try:
            validated = []
            for row in grid:
                validated_row = []
                for cell in row:
                    val = int(cell)
                    if not 0 <= val <= 9:
                        return None
                    validated_row.append(val)
                validated.append(validated_row)
            
            return validated
        except (ValueError, TypeError):
            return None
    
    async def test_on_training(
        self,
        instruction: str,
        train_pairs: list
    ) -> tuple[float, list[tuple]]:
        """
        Test instruction on all training pairs.
        
        Returns:
            (score, list of (input, expected, actual) error tuples)
        """
        from ..models.task import grids_equal, score_grid
        
        correct = 0
        errors = []
        total_score = 0.0
        
        for pair in train_pairs:
            result, error = await self.execute(instruction, pair.input)
            
            if error or result is None:
                errors.append((pair.input, pair.output, []))
            elif grids_equal(result, pair.output):
                correct += 1
                total_score += 1.0
            else:
                # Partial credit for close matches
                partial = score_grid(result, pair.output)
                total_score += partial
                errors.append((pair.input, pair.output, result))
        
        score = total_score / len(train_pairs) if train_pairs else 0.0
        return score, errors
