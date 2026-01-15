"""VLM-based visual critic for grid verification."""

from __future__ import annotations
import logging
from pathlib import Path

from ..config import Config
from ..llms.chutes_client import ChutesClient
from ..llms.prompts import Prompts
from ..models.task import Grid
from .visualizer import GridVisualizer

logger = logging.getLogger(__name__)


class VLMCritic:
    """Visual Language Model critic for verifying grid transformations."""
    
    def __init__(self, client: ChutesClient, config: Config, visualizer: GridVisualizer | None = None):
        self.client = client
        self.config = config
        self.visualizer = visualizer or GridVisualizer(config.logs_dir / "viz")
    
    async def verify(
        self,
        input_grid: Grid,
        candidate_output: Grid,
        context: str = ""
    ) -> tuple[bool, str]:
        """
        Verify if a candidate output looks correct.
        
        Returns:
            (is_valid, reason)
        """
        # Render comparison image
        image_path = self.visualizer.render_comparison(
            input_grid,
            candidate_output,
            label="candidate"
        )
        
        # Get VLM verdict
        prompt = Prompts.vlm_verification()
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        try:
            response = await self.client.chat_with_image(
                self.config.vlm_model,
                prompt,
                image_path,
                temperature=0.2
            )
            
            return self._parse_verdict(response)
            
        except Exception as e:
            logger.warning(f"VLM verification failed: {e}")
            # On error, assume valid (don't reject potentially good solutions)
            return True, f"VLM error: {e}"
    
    def _parse_verdict(self, response: str) -> tuple[bool, str]:
        """Parse VLM response into verdict."""
        response = response.strip().upper()
        
        if response.startswith("VALID"):
            reason = response[5:].strip(": ")
            return True, reason or "Transformation looks correct"
        elif response.startswith("INVALID"):
            reason = response[7:].strip(": ")
            return False, reason or "Transformation looks incorrect"
        else:
            # Try to infer from content
            if "correct" in response.lower() or "valid" in response.lower():
                return True, response
            elif "wrong" in response.lower() or "invalid" in response.lower() or "error" in response.lower():
                return False, response
            else:
                # Ambiguous - assume valid
                return True, f"Ambiguous response: {response[:100]}"
    
    async def filter_candidates(
        self,
        input_grid: Grid,
        candidates: list[Grid],
        max_to_check: int = 10
    ) -> list[tuple[Grid, str]]:
        """
        Filter candidates through VLM critique.
        
        Returns:
            List of (grid, reason) for valid candidates
        """
        import asyncio
        
        # Only check top candidates to save API calls
        to_check = candidates[:max_to_check]
        
        tasks = [
            self.verify(input_grid, candidate)
            for candidate in to_check
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid = []
        for candidate, result in zip(to_check, results):
            if isinstance(result, Exception):
                # On error, include the candidate
                valid.append((candidate, "VLM check failed"))
            else:
                is_valid, reason = result
                if is_valid:
                    valid.append((candidate, reason))
                else:
                    logger.debug(f"VLM rejected candidate: {reason}")
        
        return valid
