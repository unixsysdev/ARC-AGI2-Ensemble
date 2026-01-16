"""English instruction generator for ARC tasks."""

from __future__ import annotations
import asyncio
import logging

from ..config import Config
from ..llms.chutes_client import ChutesClient
from ..llms.prompts import Prompts
from ..models.task import Task

logger = logging.getLogger(__name__)


class InstructionGenerator:
    """Generates English transformation instructions for ARC tasks."""
    
    def __init__(self, client: ChutesClient, config: Config):
        self.client = client
        self.config = config
    
    async def generate(self, task: Task, n: int = 1, temperature: float | None = None) -> list[str]:
        """Generate n instruction candidates for the task."""
        messages = Prompts.instruction_generation(task)
        temp = temperature or self.config.reasoner_model.temperature
        
        # Generate n instructions in parallel with varying temperatures
        tasks = []
        for i in range(n):
            t = temp + (i * 0.05) if n > 1 else temp
            t = min(t, 1.5)
            tasks.append(self._generate_single(messages, t))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        instructions = []
        for r in results:
            if isinstance(r, str) and r.strip():
                instructions.append(r.strip())
            elif isinstance(r, Exception):
                logger.warning(f"Instruction generation failed: {r}")
        
        return instructions
    
    async def _generate_single(self, messages: list[dict], temperature: float) -> str:
        """Generate a single instruction response."""
        return await self.client.chat(
            self.config.reasoner_model,
            messages,
            temperature=temperature
        )
    
    async def revise_individual(
        self,
        task: Task,
        instruction: str,
        errors: list[tuple]  # (input, expected, actual)
    ) -> str | None:
        """Revise instruction based on specific errors."""
        messages = Prompts.individual_revision(instruction, errors)
        
        try:
            response = await self.client.chat(
                self.config.reasoner_model,
                messages,
                temperature=0.6
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Individual revision failed: {e}")
            return None
    
    async def revise_pooled(
        self,
        task: Task,
        scored_instructions: list[tuple[str, float]]
    ) -> str | None:
        """Synthesize new instruction from top performers."""
        messages = Prompts.pooled_revision(scored_instructions)
        
        try:
            response = await self.client.chat(
                self.config.reasoner_model,
                messages,
                temperature=0.8
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Pooled revision failed: {e}")
            return None
