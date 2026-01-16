"""Filmstrip renderer for visualizing execution trajectory.

Renders a sequence of grid states as a horizontal filmstrip image,
allowing the VLM to see the progression of transformations.
"""

from __future__ import annotations
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

if TYPE_CHECKING:
    from ..primitives.interpreter import ExecutionState

# Official ARC color map (same as arc_solver)
ARC_COLORS = {
    0: '#000000',  # Black
    1: '#0074D9',  # Blue
    2: '#FF4136',  # Red
    3: '#2ECC40',  # Green
    4: '#FFDC00',  # Yellow
    5: '#AAAAAA',  # Grey
    6: '#F012BE',  # Fuchsia
    7: '#FF851B',  # Orange
    8: '#7FDBFF',  # Teal
    9: '#870C25',  # Maroon
}


class FilmstripRenderer:
    """Render sequence of grid states as horizontal filmstrip.
    
    Features:
    - Side-by-side states with step labels
    - Change highlighting (diff overlay)
    - Selection visualization (semi-transparent overlay)
    - Organized output: logs/filmstrips/{task_id}/{run_timestamp}/
    """
    
    def __init__(self, output_dir: Path | None = None):
        self.base_dir = output_dir or Path(tempfile.gettempdir()) / "filmstrips"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Current run context (set by solver)
        self.task_id: str | None = None
        self.run_id: str | None = None
        self.attempt: int = 0
        self._run_dir: Path | None = None
        
        # Create colormap
        color_list = [ARC_COLORS[i] for i in range(10)]
        self.cmap = colors.ListedColormap(color_list)
        self.norm = colors.Normalize(vmin=0, vmax=9)
    
    def set_run_context(self, task_id: str, run_id: str | None = None):
        """Set context for current run (creates folder structure).
        
        Creates: logs/filmstrips/{task_id}/{run_timestamp}/
        
        Args:
            task_id: Task ID (e.g., "00d62c1b")
            run_id: Run timestamp (auto-generated if None)
        """
        from datetime import datetime
        
        self.task_id = task_id
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.attempt = 0
        
        # Create run directory
        self._run_dir = self.base_dir / task_id / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
    
    def next_attempt(self):
        """Increment attempt counter for current run."""
        self.attempt += 1
        self._last_filmstrip: Path | None = None  # Track last rendered filmstrip
    
    def mark_winner(self):
        """Rename the last filmstrip to include _WINNER suffix."""
        if self._last_filmstrip and self._last_filmstrip.exists():
            winner_path = self._last_filmstrip.with_stem(
                self._last_filmstrip.stem + "_WINNER"
            )
            self._last_filmstrip.rename(winner_path)
    
    @property
    def output_dir(self) -> Path:
        """Get current output directory (run-specific or base)."""
        return self._run_dir if self._run_dir else self.base_dir
    
    def render(
        self,
        states: list["ExecutionState"],
        highlight_changes: bool = True,
        max_frames: int = 8,
        path: Path | None = None
    ) -> Path:
        """Create horizontal filmstrip showing progression.
        
        Args:
            states: List of execution states
            highlight_changes: Show diff between consecutive frames
            max_frames: Maximum frames to show (subsamples if more)
            path: Output path (auto-generated if None)
            
        Returns:
            Path to the rendered image
        """
        if path is None:
            # Use attempt-based naming: attempt_1_5_steps.png
            if self.attempt > 0:
                path = self.output_dir / f"attempt_{self.attempt}_{len(states)}_steps.png"
            else:
                path = self.output_dir / f"filmstrip_{len(states)}_steps.png"
        
        # Subsample if too many frames
        if len(states) > max_frames:
            indices = np.linspace(0, len(states) - 1, max_frames, dtype=int)
            states = [states[i] for i in indices]
        
        n_frames = len(states)
        if n_frames == 0:
            return path
        
        # Calculate figure size
        max_h = max(len(s.grid) for s in states)
        max_w = max(len(s.grid[0]) if s.grid else 0 for s in states)
        
        fig_width = max(12, n_frames * 2.5)
        fig_height = max(3, max_h * 0.3)
        
        fig, axes = plt.subplots(1, n_frames, figsize=(fig_width, fig_height))
        
        if n_frames == 1:
            axes = [axes]
        
        prev_arr = None
        
        for i, (ax, state) in enumerate(zip(axes, states)):
            arr = np.array(state.grid)
            
            # Draw grid
            ax.imshow(arr, cmap=self.cmap, norm=self.norm)
            
            # Highlight changes from previous frame
            if highlight_changes and prev_arr is not None and prev_arr.shape == arr.shape:
                diff_mask = arr != prev_arr
                if np.any(diff_mask):
                    # Create overlay for changed cells
                    overlay = np.zeros((*arr.shape, 4))
                    overlay[diff_mask] = [1, 1, 0, 0.3]  # Yellow with alpha
                    ax.imshow(overlay)
            
            # Draw selection overlay if present
            if state.selections:
                for sel in state.selections:
                    if sel.mask.shape == arr.shape:
                        overlay = np.zeros((*arr.shape, 4))
                        overlay[sel.mask] = [0, 1, 1, 0.2]  # Cyan with alpha
                        ax.imshow(overlay)
            
            # Add gridlines
            h, w = arr.shape
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(which='minor', color='#555555', linewidth=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)
            
            # Title
            if i == 0:
                title = "INPUT"
            elif state.primitive:
                title = f"Step {i}: {state.primitive.type.value}"
            else:
                title = f"Step {i}"
            
            ax.set_title(title, fontsize=8, fontweight='bold')
            
            # Add primitive description
            if state.primitive and state.primitive.english:
                # Truncate long descriptions
                desc = state.primitive.english[:30]
                if len(state.primitive.english) > 30:
                    desc += "..."
                ax.set_xlabel(desc, fontsize=6)
            
            prev_arr = arr
        
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Track last rendered path for mark_winner()
        self._last_filmstrip = path
        
        return path
    
    def render_comparison(
        self,
        state: "ExecutionState",
        expected_output: list[list[int]] | None = None,
        path: Path | None = None
    ) -> Path:
        """Render single state with optional expected output comparison.
        
        Args:
            state: Current execution state
            expected_output: Expected output grid for comparison
            path: Output path
            
        Returns:
            Path to the rendered image
        """
        if path is None:
            path = self.output_dir / f"comparison_step_{state.step_index}.png"
        
        n_panels = 2 if expected_output else 1
        if expected_output:
            n_panels = 3  # Input, Current, Expected
        
        arr = np.array(state.grid)
        h, w = arr.shape
        
        fig_width = n_panels * 3
        fig_height = max(3, h * 0.3)
        
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
        
        if n_panels == 1:
            axes = [axes]
        
        # Current state
        axes[0].imshow(arr, cmap=self.cmap, norm=self.norm)
        title = f"Step {state.step_index}"
        if state.primitive:
            title += f": {state.primitive.type.value}"
        axes[0].set_title(title, fontsize=9, fontweight='bold')
        self._add_gridlines(axes[0], h, w)
        
        # Expected output
        if expected_output and len(axes) > 1:
            exp_arr = np.array(expected_output)
            axes[1].imshow(exp_arr, cmap=self.cmap, norm=self.norm)
            axes[1].set_title("EXPECTED", fontsize=9, fontweight='bold')
            self._add_gridlines(axes[1], exp_arr.shape[0], exp_arr.shape[1])
            
            # Diff panel
            if len(axes) > 2:
                diff = np.zeros((h, w, 3))
                min_h = min(h, exp_arr.shape[0])
                min_w = min(w, exp_arr.shape[1])
                
                for i in range(min_h):
                    for j in range(min_w):
                        if arr[i, j] == exp_arr[i, j]:
                            diff[i, j] = [0.2, 0.8, 0.2]  # Green
                        else:
                            diff[i, j] = [0.8, 0.2, 0.2]  # Red
                
                axes[2].imshow(diff)
                axes[2].set_title("DIFF", fontsize=9, fontweight='bold')
                self._add_gridlines(axes[2], h, w)
        
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        return path
    
    def _add_gridlines(self, ax, h: int, w: int):
        """Add gridlines and remove tick labels."""
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='#555555', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
