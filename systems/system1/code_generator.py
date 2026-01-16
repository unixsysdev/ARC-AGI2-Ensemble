"""Python code generator for ARC tasks."""

from __future__ import annotations
import asyncio
import logging
import re
from typing import Any

from ..config import Config
from ..llms.chutes_client import ChutesClient
from ..llms.prompts import Prompts
from ..models.task import Task

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generates Python solver code for ARC tasks."""
    
    def __init__(self, client: ChutesClient, config: Config):
        self.client = client
        self.config = config
    
    async def generate(self, task: Task, n: int = 1, temperature: float | None = None) -> list[str]:
        """Generate n Python code candidates for the task."""
        messages = Prompts.code_generation(task)
        temp = temperature or self.config.coder_model.temperature
        
        # Generate n codes in parallel with different temperatures for diversity
        tasks = []
        for i in range(n):
            # Slightly vary temperature for diversity
            t = temp + (i * 0.05) if n > 1 else temp
            t = min(t, 1.5)  # Cap at 1.5
            tasks.append(self._generate_single(messages, t))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        codes = []
        failed = 0
        for r in results:
            if isinstance(r, str):
                code = self._extract_code(r)
                if code:
                    codes.append(code)
                else:
                    failed += 1
                    logger.debug(f"Code extraction failed for response ({len(r)} chars)")
            elif isinstance(r, Exception):
                failed += 1
                logger.warning(f"Code generation failed: {r}")
        
        if failed > 0:
            logger.debug(f"Code generation: {len(codes)} success, {failed} failed")
        
        return codes
    
    async def _generate_single(self, messages: list[dict], temperature: float) -> str:
        """Generate a single code response."""
        return await self.client.chat(
            self.config.coder_model,
            messages,
            temperature=temperature
        )
    
    def _extract_code(self, response: str) -> str | None:
        """Extract Python code from response."""
        # Try to find code block
        patterns = [
            r"```python\s*(.*?)```",
            r"```\s*(.*?)```",
            r"(def transform\(.*?(?=\n\ndef |\n```|\Z))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if "def transform" in code:
                    return self._clean_code(code)
        
        # If no code block, check if entire response is code
        if "def transform" in response:
            return self._clean_code(response)
        
        return None
    
    def _clean_code(self, code: str) -> str:
        """Clean and validate Python code."""
        lines = code.strip().split("\n")
        
        # Remove markdown artifacts
        cleaned = []
        for line in lines:
            if line.strip().startswith("```"):
                continue
            cleaned.append(line)
        
        code = "\n".join(cleaned)
        
        # Ensure import is present
        if "import numpy" not in code:
            code = "import numpy as np\n\n" + code
        
        return code
    
    async def revise(self, task: Task, code: str, error: str) -> str | None:
        """Revise code based on an error."""
        messages = Prompts.code_revision(code, error, task)
        
        try:
            response = await self.client.chat(
                self.config.coder_model,
                messages,
                temperature=0.5  # Lower temperature for fixing
            )
            return self._extract_code(response)
        except Exception as e:
            logger.warning(f"Code revision failed: {e}")
            return None
