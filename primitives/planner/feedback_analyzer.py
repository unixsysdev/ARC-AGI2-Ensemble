"""Feedback Analyzer - Reasons over failures to generate actionable advice."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """Analyzes raw failure messages and generates reasoned recommendations."""
    
    ANALYSIS_PROMPT = """You are an expert ARC-AGI puzzle debugger. Analyze these failures and provide ONE clear recommendation.

FAILURES FROM PREVIOUS ATTEMPTS:
{failures}

AVAILABLE PRIMITIVES:
- select(criteria="color"|"connected"|"largest"|"smallest"|"enclosed")
- paint(color=N) / replace(source_color=A, target_color=B)
- flood_fill(color=N, start_position="border", target_color=0)
- extract() - CROPS grid to selection bounding box
- transform(action="rotate_90"|"flip_horizontal"|etc)
- gravity(direction="down"|"up"|"left"|"right")

ANALYZE:
1. What pattern do you see in the failures?
2. What's the likely root cause?
3. What specific change would fix it?

OUTPUT FORMAT (be concise, max 3 sentences):
DIAGNOSIS: [What went wrong]
RECOMMENDATION: [Specific action to try, using exact primitive names]
"""

    def __init__(self, client: Any, config: Any):
        """Initialize analyzer.
        
        Args:
            client: ChutesClient for API calls
            config: Configuration with model settings
        """
        self.client = client
        self.config = config
    
    async def analyze(self, failures: list[str]) -> str:
        """Analyze failures and generate actionable recommendation.
        
        Args:
            failures: List of failure messages from previous attempts
            
        Returns:
            Synthesized recommendation string
        """
        if not failures:
            return ""
        
        # Format failures
        failures_text = "\n".join(f"- {f}" for f in failures[:5])  # Limit to 5
        
        prompt = self.ANALYSIS_PROMPT.format(failures=failures_text)
        
        try:
            logger.info(f"[FEEDBACK ANALYZER] Reasoning over {len(failures)} failures...")
            
            response = await self.client.complete(
                prompt=prompt,
                model=self.config.llm_model,  # Use LLM for reasoning
                max_tokens=200,
                temperature=0.3  # More focused
            )
            
            recommendation = response.strip()
            logger.info(f"[FEEDBACK ANALYZER] Recommendation: {recommendation[:100]}...")
            
            return recommendation
            
        except Exception as e:
            logger.warning(f"[FEEDBACK ANALYZER] Analysis failed: {e}")
            return ""  # Return empty if analysis fails
