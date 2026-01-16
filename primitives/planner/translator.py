"""Primitive translator: Convert English steps to DSL Program.

The translator takes English instructions and maps them to
executable primitive operations.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any

from ..primitives.dsl import (
    Primitive,
    Program,
    PrimitiveType,
    SelectCriteria,
    TransformAction,
    PaintAction,
    FilterCondition,
    CompositeMode,
    select,
    transform,
    paint,
    filter_sel,
    composite,
)

logger = logging.getLogger(__name__)


class PrimitiveTranslator:
    """Translate English steps to DSL Program.
    
    Uses LLM to parse English instructions into structured primitives,
    with fallback to pattern matching for common operations.
    """
    
    def __init__(self, client, config):
        """Initialize translator.
        
        Args:
            client: LLM API client
            config: Configuration with model settings
        """
        self.client = client
        self.config = config
        self.model = config.coder_model  # Use coder model for structured output
    
    async def translate(self, english_plan: str) -> Program:
        """Convert English plan to executable primitives.
        
        Args:
            english_plan: Multi-line English plan with steps
            
        Returns:
            Program with list of primitives
        """
        # Extract goal and steps from plan
        goal, steps = self._parse_english_plan(english_plan)
        
        # Translate each step
        primitives = []
        for step in steps:
            try:
                primitive = await self._translate_step(step)
                if primitive:
                    primitives.append(primitive)
            except Exception as e:
                logger.warning(f"Failed to translate step: {step} - {e}")
                # Create a placeholder primitive
                primitives.append(Primitive(
                    type=PrimitiveType.SELECT,
                    params=None,  # type: ignore
                    english=f"FAILED: {step}"
                ))
        
        return Program(goal=goal, steps=primitives)
    
    def _parse_english_plan(self, plan: str) -> tuple[str, list[str]]:
        """Extract goal and steps from English plan.
        
        Handles multiple formats:
        - STEP 1: action
        - 1. action
        - - action
        - SELECT/TRANSFORM/PAINT/etc. action
        """
        lines = plan.strip().split('\n')
        goal = ""
        steps = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers/sections
            if line.startswith('##') or line.startswith('**'):
                continue
            
            # Extract goal
            if line.upper().startswith("GOAL:"):
                goal = line[5:].strip()
                continue
            if "transformation rule" in line.lower() and ":" in line:
                goal = line.split(":", 1)[1].strip()
                continue
                
            # Extract steps in multiple formats
            step_content = None
            
            # Format: "STEP 1: action" or "STEP 1 - action"
            if line.upper().startswith("STEP"):
                if ":" in line:
                    step_content = line.split(":", 1)[1].strip()
                elif "-" in line:
                    step_content = line.split("-", 1)[1].strip()
                    
            # Format: "1. action" or "1) action"
            elif re.match(r'^\d+[\.\)]\s+', line):
                step_content = re.sub(r'^\d+[\.\)]\s+', '', line)
                
            # Format: "- action" or "* action"  
            elif line.startswith('-') or line.startswith('*'):
                step_content = line[1:].strip()
                
            # Format: Direct primitive commands
            elif any(line.upper().startswith(p) for p in 
                     ['SELECT', 'TRANSFORM', 'PAINT', 'FILTER', 'COMPOSITE', 'COPY', 'EXTRACT', 'GRAVITY', 'OUTPUT']):
                step_content = line
            
            if step_content and len(step_content) > 5:  # Minimum length check
                steps.append(step_content)
        
        # If no steps found, try to extract any line with primitive keywords
        if not steps:
            primitives = ['select', 'paint', 'transform', 'filter', 'composite', 
                         'copy', 'extract', 'fill', 'rotate', 'flip', 'shift']
            for line in lines:
                line_lower = line.lower()
                if any(p in line_lower for p in primitives) and len(line) > 10:
                    steps.append(line.strip())
        
        logger.debug(f"Parsed plan: goal='{goal[:30]}...', steps={len(steps)}")
        return goal, steps
    
    async def _translate_step(self, step: str) -> Primitive | None:
        """Translate a single English step to primitive.
        
        First tries pattern matching, then falls back to LLM.
        """
        # Try pattern matching first (fast path)
        primitive = self._pattern_match(step)
        if primitive:
            return primitive
        
        # Fall back to LLM translation
        return await self._llm_translate(step)
    
    def _pattern_match(self, step: str) -> Primitive | None:
        """Pattern matching for common operations."""
        step_lower = step.lower()
        
        # === SELECT patterns ===
        
        # "Select all [color] cells/objects" or "select [color] cells"
        if "select" in step_lower:
            color_match = re.search(r'select (?:all |the )?(\w+)(?: cells| objects| squares| shape)?', step_lower)
            if color_match:
                color_name = color_match.group(1)
                color = self._color_name_to_int(color_name)
                if color is not None:
                    return select(SelectCriteria.COLOR, color, step)
        
        if "find the largest" in step_lower or "select the largest" in step_lower:
            return select(SelectCriteria.LARGEST, english=step)
        
        if "find the smallest" in step_lower or "select the smallest" in step_lower:
            return select(SelectCriteria.SMALLEST, english=step)
        
        if "find connected" in step_lower or "identify connected" in step_lower or "connected component" in step_lower:
            return select(SelectCriteria.CONNECTED, english=step)
        
        if "select all non-" in step_lower and "background" in step_lower:
            return select(SelectCriteria.ALL, english=step)
        
        # Advanced selection patterns
        if "find holes" in step_lower or "enclosed" in step_lower or "inside" in step_lower:
            # Find enclosed regions
            color_match = re.search(r'enclosed by (\w+)|inside (\w+)', step_lower)
            if color_match:
                color_name = color_match.group(1) or color_match.group(2)
                color = self._color_name_to_int(color_name)
                return select(SelectCriteria.ENCLOSED, color, step)
            return select(SelectCriteria.ENCLOSED, english=step)
        
        if "corner" in step_lower:
            return select(SelectCriteria.CORNERS, english=step)
        
        if "boundary" in step_lower or "edge" in step_lower or "border" in step_lower:
            if "select" in step_lower or "find" in step_lower or "identify" in step_lower:
                return select(SelectCriteria.BOUNDARY, english=step)
        
        # === TRANSFORM patterns ===
        if "rotate" in step_lower:
            if "90" in step:
                return transform(TransformAction.ROTATE_90, english=step)
            elif "180" in step:
                return transform(TransformAction.ROTATE_180, english=step)
            elif "270" in step:
                return transform(TransformAction.ROTATE_270, english=step)
        
        if "flip horizontal" in step_lower or "mirror horizontal" in step_lower:
            return transform(TransformAction.FLIP_H, english=step)
        
        if "flip vertical" in step_lower or "mirror vertical" in step_lower:
            return transform(TransformAction.FLIP_V, english=step)
        
        if "shift" in step_lower or "move" in step_lower:
            dx, dy = 0, 0
            if "down" in step_lower:
                match = re.search(r'(\d+)', step)
                dy = int(match.group(1)) if match else 1
            elif "up" in step_lower:
                match = re.search(r'(\d+)', step)
                dy = -(int(match.group(1)) if match else 1)
            elif "right" in step_lower:
                match = re.search(r'(\d+)', step)
                dx = int(match.group(1)) if match else 1
            elif "left" in step_lower:
                match = re.search(r'(\d+)', step)
                dx = -(int(match.group(1)) if match else 1)
            
            if dx != 0 or dy != 0:
                return transform(TransformAction.SHIFT, dx=dx, dy=dy, english=step)
        
        # === PAINT patterns ===
        
        if "paint" in step_lower:
            # Try multiple patterns to extract color
            color = None
            
            # Pattern 1: "paint...with [color]" or "paint...to [color]"
            color_match = re.search(r'(?:with|to|in)\s+(\w+)(?:\s|$|\.|\))', step_lower)
            if color_match:
                color = self._color_name_to_int(color_match.group(1))
            
            # Pattern 2: "paint [color]" at end of line
            if color is None:
                color_match = re.search(r'paint.*?\s+(black|blue|red|green|yellow|grey|gray|fuchsia|pink|orange|teal|cyan|maroon|brown|\d)(?:\s|$|\.|\))', step_lower)
                if color_match:
                    color = self._color_name_to_int(color_match.group(1))
            
            # Pattern 3: "color X" or "(color X)"
            if color is None:
                color_match = re.search(r'color\s*(\d)', step_lower)
                if color_match:
                    color = int(color_match.group(1))
            
            if color is not None:
                logger.debug(f"PAINT pattern matched: color={color} from '{step[:50]}...'")
                return paint(PaintAction.FILL, color, english=step)
        
        if "fill with" in step_lower or "color with" in step_lower:
            color_match = re.search(r'(?:fill|color) with (\w+)', step_lower)
            if color_match:
                color = self._color_name_to_int(color_match.group(1))
                if color is not None:
                    return paint(PaintAction.FILL, color, english=step)
        
        if "replace" in step_lower and "with" in step_lower:
            match = re.search(r'replace (\w+) with (\w+)', step_lower)
            if match:
                src_color = self._color_name_to_int(match.group(1))
                dst_color = self._color_name_to_int(match.group(2))
                if src_color is not None and dst_color is not None:
                    return paint(PaintAction.REPLACE, dst_color, src_color, english=step)
        
        if "outline" in step_lower:
            color_match = re.search(r'outline (?:in |with )?(\w+)', step_lower)
            if color_match:
                color = self._color_name_to_int(color_match.group(1))
                if color is not None:
                    return paint(PaintAction.OUTLINE, color, english=step)
        
        # === FILTER patterns ===
        if "keep only" in step_lower or "filter" in step_lower:
            if "border" in step_lower or "edge" in step_lower:
                return filter_sel(FilterCondition.TOUCHES_BORDER, english=step)
            elif "larger" in step_lower or "area >" in step_lower:
                match = re.search(r'(\d+)', step)
                value = int(match.group(1)) if match else 5
                return filter_sel(FilterCondition.AREA_GT, value, english=step)
            elif "smaller" in step_lower or "area <" in step_lower:
                match = re.search(r'(\d+)', step)
                value = int(match.group(1)) if match else 5
                return filter_sel(FilterCondition.AREA_LT, value, english=step)
        
        if "remove" in step_lower:
            if "small" in step_lower:
                match = re.search(r'(\d+)', step)
                value = int(match.group(1)) if match else 5
                return filter_sel(FilterCondition.AREA_GT, value, english=step)
        
        # === COMPOSITE patterns ===
        if "place on" in step_lower or "overlay" in step_lower:
            return composite(CompositeMode.OVERLAY, english=step)
        
        if "merge" in step_lower:
            return composite(CompositeMode.MERGE, english=step)
        
        return None
    
    async def _llm_translate(self, step: str) -> Primitive | None:
        """Use LLM to translate step to primitive with few-shot examples."""
        messages = [
            {
                "role": "system",
                "content": """You translate English instructions to ARC primitives.

Available primitives:
1. SELECT(criteria, value): Find objects
   - criteria: "color", "shape", "largest", "smallest", "all", "connected", "position"
   - value: color number (0-9), or bounding box [r1,c1,r2,c2]

2. TRANSFORM(action, dx, dy, factor): Geometric changes
   - action: "rotate_90", "rotate_180", "rotate_270", "flip_horizontal", "flip_vertical", "shift", "scale"
   - dx, dy: offset for shift
   - factor: multiplier for scale

3. PAINT(action, color, source_color): Change colors
   - action: "fill", "outline", "replace", "invert"
   - color: target color (0-9)
   - source_color: for replace, color to replace

4. FILTER(condition, value): Refine selection
   - condition: "touches_border", "area_eq", "area_gt", "area_lt", "color_eq"
   - value: threshold or color

5. COMPOSITE(mode): Combine layers
   - mode: "overlay", "underlay", "merge", "replace"

Output ONLY valid JSON: {"type": "...", "params": {...}}"""
            },
            # Few-shot examples for edge cases
            {
                "role": "user",
                "content": "Translate to primitive: Find the blue pixels and fill them with red"
            },
            {
                "role": "assistant", 
                "content": '{"type": "select", "params": {"criteria": "color", "value": 1}}'
            },
            {
                "role": "user",
                "content": "Translate to primitive: Color the selected cells yellow"
            },
            {
                "role": "assistant",
                "content": '{"type": "paint", "params": {"action": "fill", "color": 4}}'
            },
            {
                "role": "user",
                "content": "Translate to primitive: Move everything down by 3 cells"
            },
            {
                "role": "assistant",
                "content": '{"type": "transform", "params": {"action": "shift", "dx": 0, "dy": 3}}'
            },
            {
                "role": "user",
                "content": "Translate to primitive: Keep only objects touching the edge"
            },
            {
                "role": "assistant",
                "content": '{"type": "filter", "params": {"condition": "touches_border"}}'
            },
            # Actual query
            {
                "role": "user",
                "content": f"Translate to primitive: {step}"
            }
        ]
        
        try:
            response = await self.client.chat(
                self.model,
                messages,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Parse JSON response
            data = self._parse_json(response)
            if data:
                return self._json_to_primitive(data, step)
        except Exception as e:
            logger.warning(f"LLM translation failed: {e}")
        
        return None
    
    def _parse_json(self, text: str) -> dict | None:
        """Extract JSON from response."""
        # Try to find JSON object
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        try:
            return json.loads(text)
        except:
            # Try to find JSON object
            match = re.search(r'\{[^{}]+\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return None
    
    def _json_to_primitive(self, data: dict, english: str) -> Primitive | None:
        """Convert JSON to Primitive object."""
        ptype = data.get("type", "").lower()
        params = data.get("params", {})
        
        if ptype == "select":
            criteria = params.get("criteria", "all")
            value = params.get("value")
            criteria_enum = {
                "color": SelectCriteria.COLOR,
                "largest": SelectCriteria.LARGEST,
                "smallest": SelectCriteria.SMALLEST,
                "all": SelectCriteria.ALL,
                "connected": SelectCriteria.CONNECTED,
                "position": SelectCriteria.POSITION,
            }.get(criteria, SelectCriteria.ALL)
            return select(criteria_enum, value, english)
        
        elif ptype == "transform":
            action = params.get("action", "rotate_90")
            action_enum = {
                "rotate_90": TransformAction.ROTATE_90,
                "rotate_180": TransformAction.ROTATE_180,
                "rotate_270": TransformAction.ROTATE_270,
                "flip_horizontal": TransformAction.FLIP_H,
                "flip_vertical": TransformAction.FLIP_V,
                "shift": TransformAction.SHIFT,
                "scale": TransformAction.SCALE,
            }.get(action, TransformAction.ROTATE_90)
            return transform(
                action_enum,
                dx=params.get("dx", 0),
                dy=params.get("dy", 0),
                factor=params.get("factor", 1),
                english=english
            )
        
        elif ptype == "paint":
            action = params.get("action", "fill")
            action_enum = {
                "fill": PaintAction.FILL,
                "outline": PaintAction.OUTLINE,
                "replace": PaintAction.REPLACE,
                "invert": PaintAction.INVERT,
            }.get(action, PaintAction.FILL)
            return paint(
                action_enum,
                color=params.get("color", 0),
                source_color=params.get("source_color"),
                english=english
            )
        
        elif ptype == "filter":
            condition = params.get("condition", "area_gt")
            condition_enum = {
                "touches_border": FilterCondition.TOUCHES_BORDER,
                "area_eq": FilterCondition.AREA_EQ,
                "area_gt": FilterCondition.AREA_GT,
                "area_lt": FilterCondition.AREA_LT,
                "color_eq": FilterCondition.COLOR_EQ,
            }.get(condition, FilterCondition.AREA_GT)
            return filter_sel(condition_enum, params.get("value"), english)
        
        elif ptype == "composite":
            mode = params.get("mode", "overlay")
            mode_enum = {
                "overlay": CompositeMode.OVERLAY,
                "underlay": CompositeMode.UNDERLAY,
                "merge": CompositeMode.MERGE,
                "replace": CompositeMode.REPLACE,
            }.get(mode, CompositeMode.OVERLAY)
            return composite(mode_enum, english=english)
        
        return None
    
    def _color_name_to_int(self, name: str) -> int | None:
        """Convert color name to integer."""
        colors = {
            "black": 0, "0": 0,
            "blue": 1, "1": 1,
            "red": 2, "2": 2,
            "green": 3, "3": 3,
            "yellow": 4, "4": 4,
            "grey": 5, "gray": 5, "5": 5,
            "fuchsia": 6, "pink": 6, "magenta": 6, "6": 6,
            "orange": 7, "7": 7,
            "teal": 8, "cyan": 8, "8": 8,
            "maroon": 9, "brown": 9, "9": 9,
        }
        return colors.get(name.lower())
