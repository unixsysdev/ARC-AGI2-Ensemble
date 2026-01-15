"""Grid visualization for VLM critic."""

from __future__ import annotations
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from ..models.task import Grid


# Official ARC color map
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


class GridVisualizer:
    """Renders ARC grids as images for VLM processing."""
    
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create colormap
        color_list = [ARC_COLORS[i] for i in range(10)]
        self.cmap = colors.ListedColormap(color_list)
        self.norm = colors.Normalize(vmin=0, vmax=9)
    
    def render_grid(self, grid: Grid, path: str | Path | None = None) -> Path:
        """Render a single grid to PNG."""
        if path is None:
            path = self.output_dir / "temp_grid.png"
        path = Path(path)
        
        arr = np.array(grid)
        height, width = arr.shape
        
        # Calculate figure size (scale by grid dimensions)
        fig_width = max(2, width * 0.5)
        fig_height = max(2, height * 0.5)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Draw grid
        ax.imshow(arr, cmap=self.cmap, norm=self.norm)
        
        # Add gridlines
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which='minor', color='#555555', linewidth=1)
        
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        return path
    
    def render_comparison(
        self,
        input_grid: Grid,
        output_grid: Grid,
        path: str | Path | None = None,
        label: str = ""
    ) -> Path:
        """Render input and output side by side for comparison."""
        if path is None:
            path = self.output_dir / "temp_comparison.png"
        path = Path(path)
        
        input_arr = np.array(input_grid)
        output_arr = np.array(output_grid)
        
        # Calculate figure size
        h1, w1 = input_arr.shape
        h2, w2 = output_arr.shape
        
        max_h = max(h1, h2)
        total_w = w1 + w2 + 2  # +2 for gap
        
        fig_width = max(4, total_w * 0.4)
        fig_height = max(3, max_h * 0.4)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        
        # Input grid
        ax1.imshow(input_arr, cmap=self.cmap, norm=self.norm)
        ax1.set_title("INPUT", fontsize=10, fontweight='bold')
        ax1.set_xticks(np.arange(-0.5, w1, 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, h1, 1), minor=True)
        ax1.grid(which='minor', color='#555555', linewidth=1)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.tick_params(axis='both', which='both', length=0)
        
        # Output grid
        ax2.imshow(output_arr, cmap=self.cmap, norm=self.norm)
        title = "OUTPUT" if not label else f"OUTPUT ({label})"
        ax2.set_title(title, fontsize=10, fontweight='bold')
        ax2.set_xticks(np.arange(-0.5, w2, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, h2, 1), minor=True)
        ax2.grid(which='minor', color='#555555', linewidth=1)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(axis='both', which='both', length=0)
        
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        return path
    
    def render_diff(
        self,
        expected: Grid,
        actual: Grid,
        path: str | Path | None = None
    ) -> Path:
        """Render expected vs actual with diff highlighting."""
        if path is None:
            path = self.output_dir / "temp_diff.png"
        path = Path(path)
        
        exp_arr = np.array(expected)
        act_arr = np.array(actual)
        
        h, w = exp_arr.shape
        
        fig_width = max(4, w * 0.8)
        fig_height = max(3, h * 0.4)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        
        # Expected
        ax1.imshow(exp_arr, cmap=self.cmap, norm=self.norm)
        ax1.set_title("EXPECTED", fontsize=9, fontweight='bold')
        self._add_grid_lines(ax1, h, w)
        
        # Actual
        ax2.imshow(act_arr, cmap=self.cmap, norm=self.norm)
        ax2.set_title("ACTUAL", fontsize=9, fontweight='bold')
        self._add_grid_lines(ax2, h, w)
        
        # Diff (red for wrong, green for correct)
        diff = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                if i < exp_arr.shape[0] and j < exp_arr.shape[1]:
                    if i < act_arr.shape[0] and j < act_arr.shape[1]:
                        if exp_arr[i, j] == act_arr[i, j]:
                            diff[i, j] = [0.2, 0.8, 0.2]  # Green
                        else:
                            diff[i, j] = [0.8, 0.2, 0.2]  # Red
                    else:
                        diff[i, j] = [0.8, 0.2, 0.2]  # Missing = Red
        
        ax3.imshow(diff)
        ax3.set_title("DIFF", fontsize=9, fontweight='bold')
        self._add_grid_lines(ax3, h, w)
        
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        return path
    
    def _add_grid_lines(self, ax, h, w):
        """Add gridlines and remove ticks."""
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='#555555', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
