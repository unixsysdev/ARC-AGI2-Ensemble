"""Domain Specific Language (DSL) for ARC primitives.

The 8 core primitives:
- SELECT: Find objects by criteria (color, shape, size, position)
- TRANSFORM: Geometric changes (rotate, flip, shift, scale)
- PAINT: Change colors (fill, gradient, pattern)
- FILTER: Refine selection (keep/remove based on conditions)
- COMPOSITE: Combine layers (overlay, place, merge)
- COPY: Duplicate selection to new location
- EXTRACT: Crop grid to selection bounds
- GRAVITY: Drop objects in a direction until blocked
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np


# Type aliases
Grid = list[list[int]]


class PrimitiveType(Enum):
    """Types of DSL primitives."""
    SELECT = "select"
    TRANSFORM = "transform"
    PAINT = "paint"
    FILTER = "filter"
    COMPOSITE = "composite"
    COPY = "copy"           # Duplicate selection
    EXTRACT = "extract"     # Crop to selection
    GRAVITY = "gravity"     # Drop objects in direction
    FLOOD_FILL = "flood_fill"  # Flood fill from position with color


class SelectCriteria(Enum):
    """Criteria for SELECT primitive."""
    # Basic criteria
    COLOR = "color"              # Select by color value
    SHAPE = "shape"              # Select by shape (square, line, etc.)
    SIZE = "size"                # Select by area/size
    POSITION = "position"        # Select by grid position
    LARGEST = "largest"          # Select the largest object
    SMALLEST = "smallest"        # Select the smallest object
    SIZE_RANK = "size_rank"      # Select by size rank (0=smallest, -1=largest, 1=2nd smallest)
    ALL = "all"                  # Select all non-background objects
    CONNECTED = "connected"      # Select connected components
    UNIQUE = "unique"            # Select object with unique property (unique_by in value)
    
    # Advanced pattern matching criteria
    PATTERN = "pattern"          # Match a template pattern (template in value)
    ENCLOSED = "enclosed"        # Find holes/regions enclosed by color
    CORNERS = "corners"          # Find corner cells of objects
    BOUNDARY = "boundary"        # Find edge cells of objects
    NEIGHBORS = "neighbors"      # Find neighbors of selection (4-connected)
    DIAGONAL = "diagonal"        # Find diagonal neighbors


class TransformAction(Enum):
    """Actions for TRANSFORM primitive."""
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_H = "flip_horizontal"
    FLIP_V = "flip_vertical"
    SHIFT = "shift"           # Move by (dx, dy)
    SCALE = "scale"           # Scale by factor
    CROP = "crop"             # Crop to bounding box
    EXPAND = "expand"         # Expand to fill
    MIRROR_H = "mirror_h"     # Mirror horizontally (duplicate)
    MIRROR_V = "mirror_v"     # Mirror vertically (duplicate)
    TRANSPOSE = "transpose"   # Swap rows and columns


class PaintAction(Enum):
    """Actions for PAINT primitive."""
    FILL = "fill"             # Solid color fill
    OUTLINE = "outline"       # Only paint border
    REPLACE = "replace"       # Replace one color with another
    INVERT = "invert"         # Invert colors
    HOLLOW = "hollow"         # Remove interior, keep border
    GRADIENT = "gradient"     # Gradient fill (simple)


class FilterCondition(Enum):
    """Conditions for FILTER primitive."""
    TOUCHES_BORDER = "touches_border"
    INSIDE = "inside"         # Inside another selection
    OUTSIDE = "outside"       # Outside another selection
    OVERLAPS = "overlaps"     # Overlaps with another selection
    AREA_EQ = "area_eq"       # Area equals value
    AREA_GT = "area_gt"       # Area greater than
    AREA_LT = "area_lt"       # Area less than
    COLOR_EQ = "color_eq"     # Has specific color
    HAS_COLORS = "has_colors" # Contains ALL of these colors (value=[1,2,3])


class SelectMode(Enum):
    """Mode for how SELECT combines with existing selections."""
    SET = "set"               # Replace existing (default)
    INTERSECT = "intersect"   # Keep only overlap with existing
    UNION = "union"           # Add to existing
    SUBTRACT = "subtract"     # Remove from existing


class CompositeMode(Enum):
    """Modes for COMPOSITE primitive."""
    OVERLAY = "overlay"       # Place on top
    UNDERLAY = "underlay"     # Place underneath
    MERGE = "merge"           # Combine (non-zero wins)
    REPLACE = "replace"       # Completely replace area
    TILE = "tile"             # Tile pattern across grid


@dataclass
class Selection:
    """A selection (mask) of grid cells.
    
    Selections are the core data structure for primitives.
    They represent a subset of grid cells that can be manipulated.
    """
    mask: np.ndarray  # Boolean mask of selected cells
    source_grid: Grid  # Original grid this was selected from
    bounding_box: tuple[int, int, int, int] | None = None  # (r1, c1, r2, c2)
    
    @classmethod
    def from_grid(cls, grid: Grid) -> "Selection":
        """Create selection of all non-zero cells."""
        arr = np.array(grid)
        mask = arr != 0
        return cls(mask=mask, source_grid=grid)
    
    @classmethod
    def from_color(cls, grid: Grid, color: int) -> "Selection":
        """Create selection of all cells with given color."""
        arr = np.array(grid)
        mask = arr == color
        return cls(mask=mask, source_grid=grid)
    
    @classmethod
    def empty(cls, grid: Grid) -> "Selection":
        """Create empty selection."""
        arr = np.array(grid)
        mask = np.zeros(arr.shape, dtype=bool)
        return cls(mask=mask, source_grid=grid)
    
    @property
    def values(self) -> np.ndarray:
        """Get values in the selection."""
        arr = np.array(self.source_grid)
        return arr[self.mask]
    
    def count(self) -> int:
        """Number of selected cells."""
        return int(np.sum(self.mask))
    
    def get_bounding_box(self) -> tuple[int, int, int, int]:
        """Get bounding box (r1, c1, r2, c2)."""
        if self.bounding_box:
            return self.bounding_box
        
        rows, cols = np.where(self.mask)
        if len(rows) == 0:
            return (0, 0, 0, 0)
        return (int(rows.min()), int(cols.min()), 
                int(rows.max()), int(cols.max()))
    
    def extract(self) -> Grid:
        """Extract selected region as a grid (cropped to bounding box)."""
        r1, c1, r2, c2 = self.get_bounding_box()
        arr = np.array(self.source_grid)
        cropped = arr[r1:r2+1, c1:c2+1].copy()
        # Zero out non-selected cells
        mask_cropped = self.mask[r1:r2+1, c1:c2+1]
        cropped[~mask_cropped] = 0
        return cropped.tolist()


@dataclass
class SelectParams:
    """Parameters for SELECT primitive."""
    criteria: SelectCriteria
    value: Any = None  # Color value, size, etc.
    pattern: Grid | None = None  # Template grid for PATTERN matching
    enclosing_color: int | None = None  # For ENCLOSED: color that encloses
    mode: SelectMode = None  # How to combine with existing (default: SET/replace)


@dataclass
class TransformParams:
    """Parameters for TRANSFORM primitive."""
    action: TransformAction
    dx: int = 0  # For SHIFT
    dy: int = 0  # For SHIFT
    factor: int = 1  # For SCALE


@dataclass
class PaintParams:
    """Parameters for PAINT primitive."""
    action: PaintAction
    color: int = 0  # Target color
    source_color: int | None = None  # For REPLACE: color to replace


@dataclass 
class FilterParams:
    """Parameters for FILTER primitive."""
    condition: FilterCondition
    value: Any = None  # Threshold or reference
    reference_selection: Selection | None = None  # For INSIDE/OVERLAPS


@dataclass
class CompositeParams:
    """Parameters for COMPOSITE primitive."""
    mode: CompositeMode
    target: Grid | None = None  # Background grid


@dataclass
class CopyParams:
    """Parameters for COPY primitive."""
    dx: int = 0  # Offset for copy destination
    dy: int = 0
    count: int = 1  # Number of copies


@dataclass
class ExtractParams:
    """Parameters for EXTRACT primitive."""
    padding: int = 0  # Padding around extracted region


@dataclass
class GravityParams:
    """Parameters for GRAVITY primitive.
    
    Args:
        direction: "up", "down", "left", "right"
        stop_at_color: Stop when hitting this color (None = stop at any non-zero)
        rigid: If True, move connected components as rigid bodies.
               If False, individual pixels fall like sand.
    """
    direction: str = "down"
    stop_at_color: int | None = None
    rigid: bool = True  # Rigid body physics by default


@dataclass
class FloodFillParams:
    """Parameters for FLOOD_FILL primitive.
    
    Flood fill from a starting position or from all border cells.
    
    Args:
        color: Color to fill with
        start_position: (row, col) or "border" to fill from all border cells
        target_color: Only fill cells of this color (None = fill any)
    """
    color: int
    start_position: tuple[int, int] | str = "border"  # (row, col) or "border"
    target_color: int | None = None  # Only fill cells of this color


@dataclass
class Primitive:
    """A single DSL command.
    
    Each primitive has:
    - type: What kind of operation
    - params: Parameters for the operation
    - english: Human-readable description (for the judge)
    """
    type: PrimitiveType
    params: SelectParams | TransformParams | PaintParams | FilterParams | CompositeParams | CopyParams | ExtractParams | GravityParams
    english: str  # Human-readable description
    
    def __repr__(self) -> str:
        return f"Primitive({self.type.value}: {self.english})"


@dataclass
class Program:
    """A sequence of primitives that solves an ARC task.
    
    A program consists of:
    - goal: High-level English description of what we're trying to do
    - steps: Ordered list of primitives to execute
    """
    goal: str
    steps: list[Primitive] = field(default_factory=list)
    
    def add_step(self, primitive: Primitive) -> None:
        """Add a step to the program."""
        self.steps.append(primitive)
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __repr__(self) -> str:
        return f"Program({len(self.steps)} steps: {self.goal[:50]}...)"


# Convenience constructors
def select(criteria: SelectCriteria, value: Any = None, english: str = "") -> Primitive:
    """Create a SELECT primitive."""
    return Primitive(
        type=PrimitiveType.SELECT,
        params=SelectParams(criteria=criteria, value=value),
        english=english or f"Select by {criteria.value}"
    )


def transform(action: TransformAction, dx: int = 0, dy: int = 0, 
              factor: int = 1, english: str = "") -> Primitive:
    """Create a TRANSFORM primitive."""
    return Primitive(
        type=PrimitiveType.TRANSFORM,
        params=TransformParams(action=action, dx=dx, dy=dy, factor=factor),
        english=english or f"Transform: {action.value}"
    )


def paint(action: PaintAction, color: int = 0, 
          source_color: int | None = None, english: str = "") -> Primitive:
    """Create a PAINT primitive."""
    return Primitive(
        type=PrimitiveType.PAINT,
        params=PaintParams(action=action, color=color, source_color=source_color),
        english=english or f"Paint: {action.value} with color {color}"
    )


def filter_sel(condition: FilterCondition, value: Any = None, 
               english: str = "") -> Primitive:
    """Create a FILTER primitive."""
    return Primitive(
        type=PrimitiveType.FILTER,
        params=FilterParams(condition=condition, value=value),
        english=english or f"Filter: {condition.value}"
    )


def composite(mode: CompositeMode, target: Grid | None = None, 
              english: str = "") -> Primitive:
    """Create a COMPOSITE primitive."""
    return Primitive(
        type=PrimitiveType.COMPOSITE,
        params=CompositeParams(mode=mode, target=target),
        english=english or f"Composite: {mode.value}"
    )


def copy(dx: int = 0, dy: int = 0, count: int = 1, english: str = "") -> Primitive:
    """Create a COPY primitive."""
    return Primitive(
        type=PrimitiveType.COPY,
        params=CopyParams(dx=dx, dy=dy, count=count),
        english=english or f"Copy to offset ({dx}, {dy})"
    )


def extract(padding: int = 0, english: str = "") -> Primitive:
    """Create an EXTRACT primitive."""
    return Primitive(
        type=PrimitiveType.EXTRACT,
        params=ExtractParams(padding=padding),
        english=english or "Extract selection to new grid"
    )


def gravity(direction: str = "down", stop_at_color: int | None = None, 
            rigid: bool = True, english: str = "") -> Primitive:
    """Create a GRAVITY primitive.
    
    Args:
        direction: "up", "down", "left", "right"
        stop_at_color: Stop when hitting this color (None = any non-zero)
        rigid: If True, move connected components as units. If False, sand physics.
        english: Human-readable description
    """
    return Primitive(
        type=PrimitiveType.GRAVITY,
        params=GravityParams(direction=direction, stop_at_color=stop_at_color, rigid=rigid),
        english=english or f"Apply gravity: {direction}{' (sand)' if not rigid else ''}"
    )
