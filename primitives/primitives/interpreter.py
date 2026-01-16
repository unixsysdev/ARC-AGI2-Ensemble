"""Primitive interpreter with state tracking.

The interpreter executes primitives step-by-step, maintaining:
- Current grid state
- Active selections (object masks)
- History of all states (for filmstrip rendering)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import logging

import numpy as np
from scipy import ndimage

from .dsl import (
    Grid,
    PrimitiveType,
    Primitive,
    Program,
    Selection,
    SelectParams,
    TransformParams,
    PaintParams,
    FilterParams,
    CompositeParams,
    CopyParams,
    ExtractParams,
    GravityParams,
    SelectCriteria,
    TransformAction,
    PaintAction,
    FilterCondition,
    CompositeMode,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionState:
    """State after each primitive step.
    
    Tracks:
    - The current grid
    - Active selections (masks from SELECT operations)
    - Which step we're on
    - The primitive that produced this state
    """
    grid: Grid
    selections: list[Selection] = field(default_factory=list)
    step_index: int = 0
    primitive: Primitive | None = None
    error: str | None = None
    
    @classmethod
    def initial(cls, grid: Grid) -> "ExecutionState":
        """Create initial state from input grid."""
        return cls(grid=grid, selections=[], step_index=0, primitive=None)
    
    def copy(self) -> "ExecutionState":
        """Create a copy of this state."""
        import copy
        return ExecutionState(
            grid=[row[:] for row in self.grid],
            selections=[Selection(
                mask=s.mask.copy(),
                source_grid=[row[:] for row in s.source_grid]
            ) for s in self.selections],
            step_index=self.step_index,
            primitive=self.primitive,
            error=self.error
        )


class PrimitiveInterpreter:
    """Execute DSL commands with state tracking.
    
    The interpreter provides:
    - Step-by-step execution
    - State history for filmstrip rendering
    - Error handling with meaningful messages
    """
    
    def __init__(self):
        self._handlers = {
            PrimitiveType.SELECT: self._execute_select,
            PrimitiveType.TRANSFORM: self._execute_transform,
            PrimitiveType.PAINT: self._execute_paint,
            PrimitiveType.FILTER: self._execute_filter,
            PrimitiveType.COMPOSITE: self._execute_composite,
            PrimitiveType.COPY: self._execute_copy,
            PrimitiveType.EXTRACT: self._execute_extract,
            PrimitiveType.GRAVITY: self._execute_gravity,
            PrimitiveType.FLOOD_FILL: self._execute_flood_fill,
        }
    
    def execute_step(
        self, 
        state: ExecutionState, 
        primitive: Primitive
    ) -> ExecutionState:
        """Execute one primitive and return new state.
        
        Args:
            state: Current execution state
            primitive: Primitive to execute
            
        Returns:
            New state after execution
        """
        handler = self._handlers.get(primitive.type)
        if not handler:
            return ExecutionState(
                grid=state.grid,
                selections=state.selections,
                step_index=state.step_index + 1,
                primitive=primitive,
                error=f"Unknown primitive type: {primitive.type}"
            )
        
        try:
            new_state = handler(state, primitive)
            new_state.step_index = state.step_index + 1
            new_state.primitive = primitive
            return new_state
        except Exception as e:
            logger.exception(f"Error executing {primitive.type}: {e}")
            return ExecutionState(
                grid=state.grid,
                selections=state.selections,
                step_index=state.step_index + 1,
                primitive=primitive,
                error=str(e)
            )
    
    def execute_program(
        self, 
        grid: Grid, 
        program: Program
    ) -> list[ExecutionState]:
        """Execute full program, returning all intermediate states.
        
        This is the "filmstrip" - a sequence of all states for visualization.
        
        Args:
            grid: Input grid
            program: Program to execute
            
        Returns:
            List of all states (including initial state)
        """
        states = [ExecutionState.initial(grid)]
        
        for primitive in program.steps:
            prev_state = states[-1]
            
            # Stop on error
            if prev_state.error:
                break
            
            new_state = self.execute_step(prev_state, primitive)
            states.append(new_state)
        
        return states
    
    # ==================== Primitive Handlers ====================
    
    def _execute_select(
        self, 
        state: ExecutionState, 
        primitive: Primitive
    ) -> ExecutionState:
        """Execute SELECT primitive."""
        params: SelectParams = primitive.params
        grid = state.grid
        arr = np.array(grid)
        
        if params.criteria == SelectCriteria.COLOR:
            # Select by color
            color = params.value
            selection = Selection.from_color(grid, color)
            
        elif params.criteria == SelectCriteria.ALL:
            # Select all non-background (non-zero) cells
            selection = Selection.from_grid(grid)
            
        elif params.criteria == SelectCriteria.CONNECTED:
            # Find connected components
            mask = arr != 0
            labeled, num_features = ndimage.label(mask)
            
            selections = []
            for i in range(1, num_features + 1):
                component_mask = labeled == i
                selections.append(Selection(mask=component_mask, source_grid=grid))
            
            return ExecutionState(
                grid=grid,
                selections=selections
            )
            
        elif params.criteria == SelectCriteria.LARGEST:
            # Find largest connected component
            mask = arr != 0
            labeled, num_features = ndimage.label(mask)
            
            if num_features == 0:
                return ExecutionState(grid=grid, selections=[Selection.empty(grid)])
            
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_idx = np.argmax(sizes) + 1
            selection = Selection(mask=labeled == largest_idx, source_grid=grid)
            
        elif params.criteria == SelectCriteria.SMALLEST:
            # Find smallest connected component
            mask = arr != 0
            labeled, num_features = ndimage.label(mask)
            
            if num_features == 0:
                return ExecutionState(grid=grid, selections=[Selection.empty(grid)])
            
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            smallest_idx = np.argmin(sizes) + 1
            selection = Selection(mask=labeled == smallest_idx, source_grid=grid)
            
        elif params.criteria == SelectCriteria.POSITION:
            # Select by position (value should be (r1, c1, r2, c2))
            r1, c1, r2, c2 = params.value
            mask = np.zeros(arr.shape, dtype=bool)
            mask[r1:r2+1, c1:c2+1] = True
            selection = Selection(mask=mask, source_grid=grid)
        
        # ========== Advanced Pattern Matching ==========
        
        elif params.criteria == SelectCriteria.PATTERN:
            # Match a template pattern anywhere in the grid
            pattern = np.array(params.pattern) if params.pattern else np.array([[1]])
            ph, pw = pattern.shape
            gh, gw = arr.shape
            
            mask = np.zeros(arr.shape, dtype=bool)
            
            # Slide pattern over grid
            for r in range(gh - ph + 1):
                for c in range(gw - pw + 1):
                    window = arr[r:r+ph, c:c+pw]
                    # Check if pattern matches (treat -1 as wildcard)
                    matches = (pattern == -1) | (window == pattern)
                    if np.all(matches):
                        mask[r:r+ph, c:c+pw] = True
            
            selection = Selection(mask=mask, source_grid=grid)
        
        elif params.criteria == SelectCriteria.ENCLOSED:
            # Find regions enclosed by a specific color
            enclosing_color = params.enclosing_color if params.enclosing_color is not None else params.value
            if enclosing_color is None:
                enclosing_color = 0  # Default: find regions enclosed by non-zero
            
            # Create binary mask of "walls"
            if enclosing_color == 0:
                walls = arr != 0  # Anything non-zero is a wall
            else:
                walls = arr == enclosing_color
            
            # Find connected components of non-wall regions
            non_walls = ~walls
            labeled, num_features = ndimage.label(non_walls)
            
            # Find which regions touch the border (these are not enclosed)
            h, w = arr.shape
            border_labels = set()
            border_labels.update(labeled[0, :].flatten())    # Top
            border_labels.update(labeled[-1, :].flatten())   # Bottom
            border_labels.update(labeled[:, 0].flatten())    # Left
            border_labels.update(labeled[:, -1].flatten())   # Right
            border_labels.discard(0)  # Background
            
            # Enclosed = regions that don't touch border
            enclosed_mask = np.zeros_like(arr, dtype=bool)
            for i in range(1, num_features + 1):
                if i not in border_labels:
                    enclosed_mask |= (labeled == i)
            
            selection = Selection(mask=enclosed_mask, source_grid=grid)
        
        elif params.criteria == SelectCriteria.CORNERS:
            # Find corner cells of selected objects
            if state.selections:
                # Find corners of existing selection
                mask = np.zeros_like(arr, dtype=bool)
                for sel in state.selections:
                    # A corner has exactly 2 orthogonal neighbors within the selection
                    padded = np.pad(sel.mask, 1, mode='constant', constant_values=False)
                    # Count 4-connected neighbors
                    neighbor_count = (
                        padded[:-2, 1:-1].astype(int) +   # Above
                        padded[2:, 1:-1].astype(int) +    # Below
                        padded[1:-1, :-2].astype(int) +   # Left
                        padded[1:-1, 2:].astype(int)      # Right
                    )
                    # Corners: cells in selection with exactly 2 neighbors
                    corners = sel.mask & (neighbor_count == 2)
                    
                    # Refine: check diagonal continuity for true corners
                    mask |= corners
            else:
                # Find corners of all non-zero regions
                obj_mask = arr != 0
                padded = np.pad(obj_mask, 1, mode='constant', constant_values=False)
                neighbor_count = (
                    padded[:-2, 1:-1].astype(int) +
                    padded[2:, 1:-1].astype(int) +
                    padded[1:-1, :-2].astype(int) +
                    padded[1:-1, 2:].astype(int)
                )
                mask = obj_mask & (neighbor_count == 2)
            
            selection = Selection(mask=mask, source_grid=grid)
        
        elif params.criteria == SelectCriteria.BOUNDARY:
            # Find edge cells of objects (cells with at least one empty neighbor)
            if state.selections:
                mask = np.zeros_like(arr, dtype=bool)
                for sel in state.selections:
                    eroded = ndimage.binary_erosion(sel.mask)
                    boundary = sel.mask & ~eroded
                    mask |= boundary
            else:
                obj_mask = arr != 0
                eroded = ndimage.binary_erosion(obj_mask)
                mask = obj_mask & ~eroded
            
            selection = Selection(mask=mask, source_grid=grid)
        
        elif params.criteria == SelectCriteria.NEIGHBORS:
            # Find 4-connected neighbors of existing selection
            if state.selections:
                mask = np.zeros_like(arr, dtype=bool)
                for sel in state.selections:
                    dilated = ndimage.binary_dilation(sel.mask)
                    neighbors = dilated & ~sel.mask  # Exclude self
                    mask |= neighbors
            else:
                mask = np.zeros_like(arr, dtype=bool)
            
            selection = Selection(mask=mask, source_grid=grid)
        
        elif params.criteria == SelectCriteria.DIAGONAL:
            # Find 8-connected neighbors (excluding 4-connected)
            if state.selections:
                mask = np.zeros_like(arr, dtype=bool)
                struct_8 = np.ones((3, 3), dtype=bool)  # 8-connectivity
                struct_4 = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool)  # 4-connectivity
                for sel in state.selections:
                    dilated_8 = ndimage.binary_dilation(sel.mask, structure=struct_8)
                    dilated_4 = ndimage.binary_dilation(sel.mask, structure=struct_4)
                    diagonals = dilated_8 & ~dilated_4 & ~sel.mask
                    mask |= diagonals
            else:
                mask = np.zeros_like(arr, dtype=bool)
            
            selection = Selection(mask=mask, source_grid=grid)
            
        else:
            selection = Selection.empty(grid)
        
        return ExecutionState(
            grid=grid,
            selections=[selection]
        )
    
    def _execute_transform(
        self, 
        state: ExecutionState, 
        primitive: Primitive
    ) -> ExecutionState:
        """Execute TRANSFORM primitive on active selections or whole grid."""
        params: TransformParams = primitive.params
        grid = state.grid
        arr = np.array(grid)
        
        # Case 1: No selection - transform entire grid
        if not state.selections:
            if params.action == TransformAction.ROTATE_90:
                new_arr = np.rot90(arr, k=-1)  # k=-1 for clockwise
            elif params.action == TransformAction.ROTATE_180:
                new_arr = np.rot90(arr, k=2)
            elif params.action == TransformAction.ROTATE_270:
                new_arr = np.rot90(arr, k=1)  # k=1 for counter-clockwise (270 CW = 90 CCW)
            elif params.action == TransformAction.FLIP_H:
                new_arr = np.fliplr(arr)
            elif params.action == TransformAction.FLIP_V:
                new_arr = np.flipud(arr)
            elif params.action == TransformAction.SCALE:
                new_arr = np.repeat(np.repeat(arr, params.factor, axis=0), 
                                    params.factor, axis=1)
            elif params.action == TransformAction.SHIFT:
                # Shift entire grid
                new_arr = np.zeros_like(arr)
                h, w = arr.shape
                dy, dx = params.dy, params.dx
                # Calculate valid source and destination ranges
                src_r1 = max(0, -dy)
                src_r2 = min(h, h - dy)
                src_c1 = max(0, -dx)
                src_c2 = min(w, w - dx)
                dst_r1 = max(0, dy)
                dst_r2 = min(h, h + dy)
                dst_c1 = max(0, dx)
                dst_c2 = min(w, w + dx)
                if src_r2 > src_r1 and src_c2 > src_c1:
                    new_arr[dst_r1:dst_r2, dst_c1:dst_c2] = arr[src_r1:src_r2, src_c1:src_c2]
            elif params.action == TransformAction.CROP:
                # Crop to non-zero bounding box
                rows, cols = np.where(arr != 0)
                if len(rows) > 0:
                    r1, r2 = rows.min(), rows.max()
                    c1, c2 = cols.min(), cols.max()
                    new_arr = arr[r1:r2+1, c1:c2+1]
                else:
                    new_arr = arr
            else:
                new_arr = arr
            
            return ExecutionState(
                grid=new_arr.tolist(),
                selections=[]
            )
        
        # Case 2: Transform selections in place (preserve background)
        new_grid = arr.copy()
        new_selections = []
        
        for selection in state.selections:
            r1, c1, r2, c2 = selection.get_bounding_box()
            if r1 == r2 == c1 == c2 == 0 and selection.count() == 0:
                continue
            
            # Extract the selected region with its values
            region = arr[r1:r2+1, c1:c2+1].copy()
            mask_region = selection.mask[r1:r2+1, c1:c2+1].copy()
            
            # Clear the original location first
            new_grid[selection.mask] = 0
            
            # Apply transformation to region
            if params.action == TransformAction.ROTATE_90:
                region = np.rot90(region, k=1)
                mask_region = np.rot90(mask_region, k=1)
            elif params.action == TransformAction.ROTATE_180:
                region = np.rot90(region, k=2)
                mask_region = np.rot90(mask_region, k=2)
            elif params.action == TransformAction.ROTATE_270:
                region = np.rot90(region, k=3)
                mask_region = np.rot90(mask_region, k=3)
            elif params.action == TransformAction.FLIP_H:
                region = np.fliplr(region)
                mask_region = np.fliplr(mask_region)
            elif params.action == TransformAction.FLIP_V:
                region = np.flipud(region)
                mask_region = np.flipud(mask_region)
            elif params.action == TransformAction.SHIFT:
                # Shift position
                r1 += params.dy
                c1 += params.dx
            elif params.action == TransformAction.SCALE:
                region = np.repeat(np.repeat(region, params.factor, axis=0), 
                                   params.factor, axis=1)
                mask_region = np.repeat(np.repeat(mask_region, params.factor, axis=0),
                                        params.factor, axis=1)
            
            # Place transformed region back (only where mask is True)
            h, w = region.shape
            r2_new = r1 + h
            c2_new = c1 + w
            
            # Ensure we fit within grid bounds
            if r1 < 0:
                region = region[-r1:]
                mask_region = mask_region[-r1:]
                r1 = 0
            if c1 < 0:
                region = region[:, -c1:]
                mask_region = mask_region[:, -c1:]
                c1 = 0
            
            r2_new = min(r1 + region.shape[0], new_grid.shape[0])
            c2_new = min(c1 + region.shape[1], new_grid.shape[1])
            
            fit_h = r2_new - r1
            fit_w = c2_new - c1
            
            if fit_h > 0 and fit_w > 0:
                # Only place where mask is True
                place_mask = mask_region[:fit_h, :fit_w]
                target_slice = new_grid[r1:r2_new, c1:c2_new]
                target_slice[place_mask] = region[:fit_h, :fit_w][place_mask]
                
                # Create new selection for the transformed region
                new_full_mask = np.zeros(new_grid.shape, dtype=bool)
                new_full_mask[r1:r2_new, c1:c2_new] = place_mask
                new_selections.append(Selection(
                    mask=new_full_mask,
                    source_grid=new_grid.tolist()
                ))
        
        return ExecutionState(
            grid=new_grid.tolist(),
            selections=new_selections  # Preserve transformed selections
        )
    
    def _execute_paint(
        self, 
        state: ExecutionState, 
        primitive: Primitive
    ) -> ExecutionState:
        """Execute PAINT primitive on active selections."""
        params: PaintParams = primitive.params
        arr = np.array(state.grid)
        
        # Handle None params (from failed translation)
        if params is None:
            logger.warning("PAINT: params is None, skipping")
            return ExecutionState(grid=state.grid, selections=state.selections)
        
        logger.debug(f"PAINT: action={params.action}, color={params.color}, "
                    f"selections={len(state.selections)}")
        
        if not state.selections:
            # Paint entire grid if no selection
            logger.debug("PAINT: No selections, painting entire grid")
            if params.action == PaintAction.FILL:
                arr.fill(params.color)
            return ExecutionState(grid=arr.tolist(), selections=[])
        
        for i, selection in enumerate(state.selections):
            cell_count = selection.count()
            logger.debug(f"PAINT: Selection {i} has {cell_count} cells")
            
            if cell_count == 0:
                logger.debug("PAINT: Empty selection, skipping")
                continue
            
            if params.action == PaintAction.FILL:
                arr[selection.mask] = params.color
                logger.debug(f"PAINT: Filled {cell_count} cells with color {params.color}")
                
            elif params.action == PaintAction.REPLACE:
                # Replace source_color with color in selection
                if params.source_color is not None:
                    replace_mask = (arr == params.source_color) & selection.mask
                    count = np.sum(replace_mask)
                    arr[replace_mask] = params.color
                    logger.debug(f"PAINT: Replaced {count} cells from {params.source_color} to {params.color}")
                    
            elif params.action == PaintAction.OUTLINE:
                # Only paint border of selection
                eroded = ndimage.binary_erosion(selection.mask)
                border = selection.mask & ~eroded
                arr[border] = params.color
                logger.debug(f"PAINT: Outlined {np.sum(border)} border cells")
                
            elif params.action == PaintAction.INVERT:
                # Invert colors (9 - color) within selection
                arr[selection.mask] = 9 - arr[selection.mask]
        
        return ExecutionState(
            grid=arr.tolist(),
            selections=state.selections  # Preserve selections for chaining
        )
    
    def _execute_filter(
        self, 
        state: ExecutionState, 
        primitive: Primitive
    ) -> ExecutionState:
        """Execute FILTER primitive to refine selections."""
        params: FilterParams = primitive.params
        arr = np.array(state.grid)
        
        if not state.selections:
            return state.copy()
        
        filtered = []
        
        for selection in state.selections:
            keep = True
            
            if params.condition == FilterCondition.TOUCHES_BORDER:
                # Check if selection touches any edge
                rows, cols = np.where(selection.mask)
                if len(rows) == 0:
                    keep = False
                else:
                    h, w = arr.shape
                    touches = (rows.min() == 0 or rows.max() == h-1 or
                               cols.min() == 0 or cols.max() == w-1)
                    keep = touches
                    
            elif params.condition == FilterCondition.AREA_EQ:
                keep = selection.count() == params.value
                
            elif params.condition == FilterCondition.AREA_GT:
                keep = selection.count() > params.value
                
            elif params.condition == FilterCondition.AREA_LT:
                keep = selection.count() < params.value
                
            elif params.condition == FilterCondition.COLOR_EQ:
                # Check if selection contains only one color
                values = selection.values
                unique = np.unique(values)
                keep = len(unique) == 1 and unique[0] == params.value
            
            if keep:
                filtered.append(selection)
        
        return ExecutionState(
            grid=state.grid,
            selections=filtered
        )
    
    def _execute_composite(
        self, 
        state: ExecutionState, 
        primitive: Primitive
    ) -> ExecutionState:
        """Execute COMPOSITE primitive to combine layers."""
        params: CompositeParams = primitive.params
        
        # Start with background or current grid
        if params.target is not None:
            result = np.array(params.target)
        else:
            result = np.zeros_like(np.array(state.grid))
        
        if not state.selections:
            return ExecutionState(grid=result.tolist(), selections=[])
        
        for selection in state.selections:
            arr = np.array(selection.source_grid)
            
            if params.mode == CompositeMode.OVERLAY:
                # Place selection on top (non-zero wins)
                result[selection.mask] = arr[selection.mask]
                
            elif params.mode == CompositeMode.UNDERLAY:
                # Place selection underneath (existing non-zero preserved)
                underlay_mask = selection.mask & (result == 0)
                result[underlay_mask] = arr[underlay_mask]
                
            elif params.mode == CompositeMode.MERGE:
                # Combine (non-zero wins, selection priority)
                result[selection.mask] = arr[selection.mask]
                
            elif params.mode == CompositeMode.REPLACE:
                # Completely replace area
                r1, c1, r2, c2 = selection.get_bounding_box()
                extracted = selection.extract()
                extracted_arr = np.array(extracted)
                h, w = extracted_arr.shape
                if r1 + h <= result.shape[0] and c1 + w <= result.shape[1]:
                    result[r1:r1+h, c1:c1+w] = extracted_arr
        
        return ExecutionState(
            grid=result.tolist(),
            selections=[]
        )
    
    def _execute_copy(
        self,
        state: ExecutionState,
        primitive: Primitive
    ) -> ExecutionState:
        """Execute COPY primitive to duplicate selection at offset."""
        params: CopyParams = primitive.params
        arr = np.array(state.grid)
        
        if not state.selections:
            return state.copy()
        
        new_grid = arr.copy()
        new_selections = []
        
        for selection in state.selections:
            # Extract values from selection
            values = arr.copy()
            values[~selection.mask] = 0
            
            # Copy at offset(s)
            for i in range(params.count):
                offset_dy = params.dy * (i + 1)
                offset_dx = params.dx * (i + 1)
                
                # Create offset mask
                h, w = arr.shape
                for r in range(h):
                    for c in range(w):
                        if selection.mask[r, c]:
                            new_r = r + offset_dy
                            new_c = c + offset_dx
                            if 0 <= new_r < h and 0 <= new_c < w:
                                new_grid[new_r, new_c] = arr[r, c]
                
                # Create new selection for the copy
                new_mask = np.zeros_like(selection.mask)
                rows, cols = np.where(selection.mask)
                for r, c in zip(rows, cols):
                    new_r, new_c = r + offset_dy, c + offset_dx
                    if 0 <= new_r < h and 0 <= new_c < w:
                        new_mask[new_r, new_c] = True
                
                if np.any(new_mask):
                    new_selections.append(Selection(
                        mask=new_mask,
                        source_grid=new_grid.tolist()
                    ))
        
        # Keep original selection too
        new_selections.extend(state.selections)
        
        return ExecutionState(
            grid=new_grid.tolist(),
            selections=new_selections
        )
    
    def _execute_extract(
        self,
        state: ExecutionState,
        primitive: Primitive
    ) -> ExecutionState:
        """Execute EXTRACT primitive to crop grid to selection bounds."""
        params: ExtractParams = primitive.params
        arr = np.array(state.grid)
        
        if not state.selections:
            # Extract all non-zero
            rows, cols = np.where(arr != 0)
            if len(rows) == 0:
                return state.copy()
            r1, r2 = rows.min(), rows.max()
            c1, c2 = cols.min(), cols.max()
        else:
            # Use selection bounding box
            all_rows, all_cols = [], []
            for sel in state.selections:
                rows, cols = np.where(sel.mask)
                all_rows.extend(rows)
                all_cols.extend(cols)
            
            if not all_rows:
                return state.copy()
            
            r1, r2 = min(all_rows), max(all_rows)
            c1, c2 = min(all_cols), max(all_cols)
        
        # Add padding
        r1 = max(0, r1 - params.padding)
        c1 = max(0, c1 - params.padding)
        r2 = min(arr.shape[0] - 1, r2 + params.padding)
        c2 = min(arr.shape[1] - 1, c2 + params.padding)
        
        # Extract region
        extracted = arr[r1:r2+1, c1:c2+1]
        
        return ExecutionState(
            grid=extracted.tolist(),
            selections=[]  # Selections don't transfer to cropped grid
        )
    
    def _execute_gravity(
        self,
        state: ExecutionState,
        primitive: Primitive
    ) -> ExecutionState:
        """Execute GRAVITY primitive to drop objects in direction.
        
        Two physics modes:
        - rigid=True: Connected components move as units (objects stay intact)
        - rigid=False: Individual pixels fall like sand
        """
        params: GravityParams = primitive.params
        arr = np.array(state.grid)
        h, w = arr.shape
        
        # Determine which cells to move
        if state.selections:
            move_mask = np.zeros_like(arr, dtype=bool)
            for sel in state.selections:
                move_mask |= sel.mask
        else:
            move_mask = arr != 0
        
        new_grid = arr.copy()
        
        # Get direction delta
        if params.direction == "down":
            delta = (1, 0)
        elif params.direction == "up":
            delta = (-1, 0)
        elif params.direction == "right":
            delta = (0, 1)
        elif params.direction == "left":
            delta = (0, -1)
        else:
            delta = (1, 0)  # Default down
        
        if params.rigid:
            # === RIGID BODY PHYSICS ===
            # Move connected components as single units
            
            # Find connected components within the movable mask
            labeled, num_features = ndimage.label(move_mask)
            
            if num_features == 0:
                return state.copy()
            
            # Sort components by proximity to "floor" (furthest in gravity direction first)
            component_order = []
            for i in range(1, num_features + 1):
                comp_mask = labeled == i
                rows, cols = np.where(comp_mask)
                
                if params.direction == "down":
                    sort_key = rows.max()  # Lowest row first
                elif params.direction == "up":
                    sort_key = -rows.min()  # Highest row first
                elif params.direction == "right":
                    sort_key = cols.max()  # Rightmost first
                elif params.direction == "left":
                    sort_key = -cols.min()  # Leftmost first
                else:
                    sort_key = 0
                
                component_order.append((i, sort_key))
            
            component_order.sort(key=lambda x: -x[1])  # Process furthest first
            
            # Clear movable cells from grid before moving
            new_grid[move_mask] = 0
            
            # Move each connected component as a unit
            for comp_id, _ in component_order:
                comp_mask = labeled == comp_id
                rows, cols = np.where(comp_mask)
                
                if len(rows) == 0:
                    continue
                
                # Get the component's values
                comp_values = arr[comp_mask]
                
                # Find max distance this component can move before ANY pixel hits obstacle
                max_dist = float('inf')
                
                for r, c in zip(rows, cols):
                    dist = 0
                    test_r, test_c = r + delta[0], c + delta[1]
                    
                    while True:
                        # Check bounds
                        if not (0 <= test_r < h and 0 <= test_c < w):
                            break
                        
                        # Check if blocked (by something not in this component)
                        blocking = new_grid[test_r, test_c]
                        if blocking != 0:
                            if params.stop_at_color is None or blocking == params.stop_at_color:
                                break
                        
                        dist += 1
                        test_r += delta[0]
                        test_c += delta[1]
                    
                    max_dist = min(max_dist, dist)
                
                if max_dist == float('inf'):
                    max_dist = 0
                
                # Move entire component by max_dist
                for (r, c), value in zip(zip(rows, cols), comp_values):
                    new_r = r + delta[0] * max_dist
                    new_c = c + delta[1] * max_dist
                    
                    if 0 <= new_r < h and 0 <= new_c < w:
                        new_grid[new_r, new_c] = value
        
        else:
            # === SAND PHYSICS ===
            # Individual pixels fall independently
            
            movable_cells = []
            rows, cols = np.where(move_mask)
            for r, c in zip(rows, cols):
                movable_cells.append(((r, c), arr[r, c]))
            
            # Clear movable cells
            new_grid[move_mask] = 0
            
            # Sort by direction (process cells furthest in direction first)
            if params.direction == "down":
                movable_cells.sort(key=lambda x: -x[0][0])
            elif params.direction == "up":
                movable_cells.sort(key=lambda x: x[0][0])
            elif params.direction == "right":
                movable_cells.sort(key=lambda x: -x[0][1])
            elif params.direction == "left":
                movable_cells.sort(key=lambda x: x[0][1])
            
            # Move each pixel
            for (r, c), value in movable_cells:
                new_r, new_c = r, c
                
                while True:
                    next_r = new_r + delta[0]
                    next_c = new_c + delta[1]
                    
                    if not (0 <= next_r < h and 0 <= next_c < w):
                        break
                    
                    blocking = new_grid[next_r, next_c]
                    if params.stop_at_color is not None:
                        if blocking == params.stop_at_color:
                            break
                    else:
                        if blocking != 0:
                            break
                    
                    new_r, new_c = next_r, next_c
                
                new_grid[new_r, new_c] = value
        
        return ExecutionState(
            grid=new_grid.tolist(),
            selections=[]
        )
    
    def _execute_flood_fill(
        self,
        state: ExecutionState,
        primitive: Primitive
    ) -> ExecutionState:
        """Execute FLOOD_FILL primitive.
        
        Flood fill from border cells or a specific position.
        Used to mark exterior cells (for finding enclosed holes).
        """
        from .dsl import FloodFillParams
        
        params: FloodFillParams = primitive.params
        arr = np.array(state.grid)
        h, w = arr.shape
        new_grid = arr.copy()
        
        fill_color = params.color
        target = params.target_color
        
        # Get starting positions
        if params.start_position == "border":
            # All border cells
            starts = []
            for c in range(w):
                starts.append((0, c))
                starts.append((h-1, c))
            for r in range(1, h-1):
                starts.append((r, 0))
                starts.append((r, w-1))
        else:
            starts = [params.start_position]
        
        # BFS flood fill
        visited = np.zeros_like(arr, dtype=bool)
        queue = []
        
        for r, c in starts:
            if 0 <= r < h and 0 <= c < w:
                if target is None or arr[r, c] == target:
                    queue.append((r, c))
                    visited[r, c] = True
        
        while queue:
            r, c = queue.pop(0)
            
            # Only fill if matches target
            if target is None or arr[r, c] == target:
                new_grid[r, c] = fill_color
            
            # Check 4 neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if target is None or arr[nr, nc] == target:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        
        return ExecutionState(
            grid=new_grid.tolist(),
            selections=state.selections
        )

