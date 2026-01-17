"""Free-form Critic Layer.

Validates if a free-form plan idea would work BEFORE attempting to code it.
This separates "reasoning correctness" from "DSL expressiveness".

If Critic says YES but Coder fails → Model was RIGHT, DSL couldn't express it.
If Critic says NO → Model's reasoning was wrong.
"""

from __future__ import annotations
import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CriticVerdict:
    """Result of critic validation."""
    valid: bool  # Would this approach work?
    confidence: str  # "high", "medium", "low"
    reasoning: str  # Why it would/wouldn't work
    suggestion: str | None = None  # How to improve if invalid


class FreeFormCritic:
    """Validate free-form plans before coding.
    
    The Critic answers: "Would this approach correctly transform input → output?"
    WITHOUT actually implementing it.
    """
    
    CRITIC_PROMPT = """You are a CRITIC for ARC-AGI puzzle solutions. Your job is to validate if a proposed approach would work.

## ARC PUZZLE FORMAT
- Grids are 2D arrays of integers 0-9 (each = a color)
- "Objects" are connected regions of non-zero cells
- You see training examples: INPUT → OUTPUT

## THE PROPOSED PLAN
{plan}

## YOUR TASK
Analyze if this plan would correctly transform the inputs to match the expected outputs.

Answer in this format:
VALID: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
REASONING: Brief explanation of why this would/wouldn't work
SUGGESTION: (only if NO) What approach would work instead

Example responses:
---
VALID: YES
CONFIDENCE: HIGH
REASONING: The plan correctly identifies that the target object has unique colors compared to the uniform blocks.
---
VALID: NO
CONFIDENCE: MEDIUM
REASONING: Selecting the "largest" object won't work because the target is actually one of the smaller objects.
SUGGESTION: Select by unique color composition instead.
---"""

    def __init__(self, client: Any, config: Any):
        """Initialize critic."""
        self.client = client
        self.config = config
        self.model = config.vlm_model  # Use VLM for visual reasoning
    
    async def validate(self, plan: str, task: Any = None) -> CriticVerdict:
        """Validate if a free-form plan would work.
        
        Args:
            plan: The free-form DSL plan to validate
            task: Optional task for context
            
        Returns:
            CriticVerdict with validation result
        """
        logger.info("[CRITIC] Validating free-form plan...")
        
        try:
            messages = [
                {"role": "system", "content": "You are a precise ARC puzzle critic. Validate solutions."},
                {"role": "user", "content": self.CRITIC_PROMPT.format(plan=plan)}
            ]
            
            response = await self.client.chat(
                self.model,
                messages,
                temperature=0.1
            )
            
            # Parse response
            verdict = self._parse_response(response)
            
            if verdict.valid:
                logger.info(f"[CRITIC] ✅ VALID ({verdict.confidence}): {verdict.reasoning[:80]}...")
            else:
                logger.warning(f"[CRITIC] ❌ INVALID ({verdict.confidence}): {verdict.reasoning[:80]}...")
                if verdict.suggestion:
                    logger.info(f"[CRITIC] 💡 Suggestion: {verdict.suggestion[:80]}...")
            
            return verdict
            
        except Exception as e:
            logger.warning(f"[CRITIC] Validation failed: {e}")
            # Default to valid if critic fails (don't block on errors)
            return CriticVerdict(
                valid=True,
                confidence="low",
                reasoning=f"Critic error: {e}"
            )
    
    def _parse_response(self, response: str) -> CriticVerdict:
        """Parse critic response into structured verdict."""
        response_upper = response.upper()
        
        # Extract VALID
        valid = "VALID: YES" in response_upper or "VALID:YES" in response_upper
        
        # Extract CONFIDENCE
        if "CONFIDENCE: HIGH" in response_upper or "CONFIDENCE:HIGH" in response_upper:
            confidence = "high"
        elif "CONFIDENCE: LOW" in response_upper or "CONFIDENCE:LOW" in response_upper:
            confidence = "low"
        else:
            confidence = "medium"
        
        # Extract REASONING
        reasoning = ""
        if "REASONING:" in response_upper:
            idx = response_upper.find("REASONING:")
            end_idx = response_upper.find("\n", idx + 10)
            if end_idx == -1:
                end_idx = len(response)
            reasoning = response[idx + 10:end_idx].strip()
        
        # Extract SUGGESTION
        suggestion = None
        if "SUGGESTION:" in response_upper:
            idx = response_upper.find("SUGGESTION:")
            end_idx = response_upper.find("\n", idx + 11)
            if end_idx == -1:
                end_idx = len(response)
            suggestion = response[idx + 11:end_idx].strip()
        
        return CriticVerdict(
            valid=valid,
            confidence=confidence,
            reasoning=reasoning or response[:200],
            suggestion=suggestion
        )
