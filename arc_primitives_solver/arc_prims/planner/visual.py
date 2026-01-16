"""Visual Planner using VLM to generate plans from grid images.

This planner renders the training examples as images and sends them
to a Vision-Language Model, which can better "see" spatial patterns
than text-only models reading ASCII grids.
"""

from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from typing import Any

from ..config import Config, ModelConfig

logger = logging.getLogger(__name__)


class VisualPlanner:
    """Generate step-by-step plans using VLM with rendered grid images."""
    
    VISUAL_PLANNING_PROMPT = """You are an expert at solving ARC-AGI puzzles. 

I'm showing you training examples from an ARC puzzle. Each example shows:
- INPUT grid on the left
- OUTPUT grid on the right

## FIRST: Object-Centric Observation
Before planning, describe what you SEE:
1. What OBJECTS are in the input? (shapes, colors, sizes)
2. What CHANGES between input and output? (position, color, size)
3. What STAYS THE SAME? (invariants)
4. What SPATIAL RELATIONSHIPS exist? (touching, inside, aligned)

## THEN: Generate a Plan
Use ONLY these primitives with EXACT syntax:

PRIMITIVES:
- SELECT all [color] cells         (e.g., "SELECT all green cells")
- SELECT the largest object
- SELECT the smallest object  
- SELECT connected components
- PAINT with [color]               (e.g., "PAINT with yellow")
- PAINT selected cells with [color]
- REPLACE [color1] with [color2]   (e.g., "REPLACE green with yellow")
- FILTER keep only cells touching border
- FILTER keep only cells larger than N
- ROTATE 90/180/270 degrees
- FLIP horizontal/vertical
- GRAVITY down/up/left/right

COLORS: black=0, blue=1, red=2, green=3, yellow=4, grey=5, pink=6, orange=7, cyan=8, brown=9

FORMAT YOUR RESPONSE AS:

## Objects
[List objects you see in INPUT: shapes, colors, positions]

## Changes
[What transforms from INPUT to OUTPUT]

## Invariants
[What stays the same]

## Steps
STEP 1: [primitive]
STEP 2: [primitive]
...

IMPORTANT: Keep steps SIMPLE. Max 5 steps. Use color names or numbers.
"""

    def __init__(self, client: Any, config: Config):
        """Initialize planner.
        
        Args:
            client: ChutesClient for API calls
            config: Configuration with VLM model settings
        """
        self.client = client
        self.config = config
        self.model = config.vlm_model
    
    async def generate_plan(self, task: Any, previous_feedback: list[str] = None) -> str:
        """Generate English plan by showing VLM the rendered training examples.
        
        Args:
            task: ARC Task with training examples
            previous_feedback: Optional list of failures from previous attempt
            
        Returns:
            String containing the step-by-step plan
        """
        # Import visualizer from arc_solver
        from src.critic.visualizer import GridVisualizer
        
        visualizer = GridVisualizer()
        
        # Create a composite image showing all training examples
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Render all training pairs into one image
            comparison_images = []
            for i, pair in enumerate(task.train):
                img_path = tmppath / f"train_{i}.png"
                visualizer.render_comparison(
                    pair.input,
                    pair.output,
                    path=img_path,
                    label=f"Example {i+1}"
                )
                comparison_images.append(img_path)
            
            # For now, use the first example (most VLMs handle single images best)
            primary_image = comparison_images[0]
            
            # Build prompt with context
            prompt = self.VISUAL_PLANNING_PROMPT
            if len(task.train) > 1:
                prompt += f"\n\nNote: There are {len(task.train)} training examples. This image shows Example 1. The same transformation rule applies to all examples."
            
            # Add feedback from previous attempt if available
            if previous_feedback:
                feedback_text = "\n".join(previous_feedback[:3])  # Limit to top 3 failures
                prompt += f"""

IMPORTANT - PREVIOUS ATTEMPT FAILED:
{feedback_text}

Learn from these failures and try a DIFFERENT approach. Avoid the same mistakes.
"""
                logger.info(f"Including feedback from {len(previous_feedback)} previous failures")
            
            logger.info(f"Visual planning with {len(task.train)} training examples")
            logger.debug(f"Using VLM: {self.model.name}")
            
            # Call VLM with graceful fallback
            try:
                response = await self.client.chat_with_image(
                    self.model,
                    prompt,
                    primary_image,
                    temperature=self.model.temperature,
                    max_tokens=self.model.max_tokens
                )
                
                logger.debug(f"VLM response: {len(response)} chars")
                return response
                
            except Exception as e:
                logger.warning(f"Visual planning failed: {e}")
                logger.info("Falling back to text-based planning")
                return self._create_fallback_plan(task)
    
    def _create_fallback_plan(self, task: Any) -> str:
        """Create a simple fallback plan when VLM fails."""
        # Analyze the training examples using simple heuristics
        first_pair = task.train[0]
        in_grid = first_pair.input
        out_grid = first_pair.output
        
        # Simple analysis
        in_colors = set(c for row in in_grid for c in row)
        out_colors = set(c for row in out_grid for c in row)
        new_colors = out_colors - in_colors
        
        plan = """## Pattern Analysis
Analyzing transformation from input to output grids.

## Transformation Rule
Unable to analyze visually - using basic heuristics.

## Step-by-Step Plan
STEP 1: SELECT all non-background objects
STEP 2: PAINT selected cells with the transformation color
STEP 3: OUTPUT the result
"""
        if new_colors:
            plan = plan.replace("transformation color", f"color {list(new_colors)[0]}")
        
        return plan
    
    async def generate_plan_multi_image(self, task: Any) -> str:
        """Generate plan using multiple training example images.
        
        This method tries to combine all training examples for better context.
        Falls back to single image if multi-image fails.
        """
        # Import visualizer and matplotlib
        from src.critic.visualizer import GridVisualizer
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        visualizer = GridVisualizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Render each training pair
            pair_images = []
            for i, pair in enumerate(task.train):
                img_path = tmppath / f"train_{i}.png"
                visualizer.render_comparison(
                    pair.input,
                    pair.output,
                    path=img_path,
                    label=f"Example {i+1}"
                )
                pair_images.append(img_path)
            
            # Combine into a single grid image
            n_examples = len(pair_images)
            fig, axes = plt.subplots(1, n_examples, figsize=(5*n_examples, 5))
            
            if n_examples == 1:
                axes = [axes]
            
            for ax, img_path in zip(axes, pair_images):
                img = mpimg.imread(str(img_path))
                ax.imshow(img)
                ax.axis('off')
            
            combined_path = tmppath / "all_examples.png"
            plt.tight_layout()
            plt.savefig(combined_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            prompt = self.VISUAL_PLANNING_PROMPT
            prompt += f"\n\nThis image shows ALL {n_examples} training examples side by side. Find the pattern that applies to ALL of them."
            
            logger.info(f"Visual planning with combined image of {n_examples} examples")
            
            response = await self.client.chat_with_image(
                self.model,
                prompt,
                combined_path,
                temperature=self.model.temperature,
                max_tokens=self.model.max_tokens
            )
            
            return response
