"""Configuration for primitives-based solver.

Extends arc_solver configuration with primitives-specific settings.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Add arc_solver to path for imports
# ARC_ROOT should always point to /home/marcel/Work/ARC/
# Handle both /ARC/arc_primitives_solver and /ARC/arc_solver/arc_primitives_solver
_here = Path(__file__).resolve()
# Walk up until we find "ARC" folder (the workspace root)
ARC_ROOT = _here
while ARC_ROOT.name != "ARC" and ARC_ROOT.parent != ARC_ROOT:
    ARC_ROOT = ARC_ROOT.parent
sys.path.insert(0, str(ARC_ROOT / "arc_solver"))

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    max_tokens: int = 4096
    temperature: float = 0.7


# Available models - verified from Chutes AI
AVAILABLE_MODELS = {
    # === TEXT (for planning & translation) ===
    # Fast 
    "nemotron-30b": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "qwen-30b": "Qwen/Qwen3-30B-A3B-Instruct",
    
    # Balanced
    "kimi": "moonshotai/Kimi-K2-Instruct",
    "glm": "zai-org/GLM-4.7-TEE",
    
    # Quality
    "deepseek": "deepseek-ai/DeepSeek-V3.2-Speciale-TEE",
    "qwen-coder": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
    
    # === VISION (VLM) ===
    # Large
    "qwen-vl": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "internvl": "OpenGVLab/InternVL3-78B-TEE",
    
    # Smaller
    "qwen-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct-TEE",
    "qwen-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
}

# Model presets for different use cases
MODEL_PRESETS = {
    "fast": {
        "reasoner": "nemotron-30b",
        "coder": "nemotron-30b",
        "vlm": "qwen-vl-32b",
    },
    "balanced": {
        "reasoner": "kimi",
        "coder": "qwen-30b",
        "vlm": "internvl",
    },
    "quality": {
        "reasoner": "deepseek",
        "coder": "qwen-coder",
        "vlm": "qwen-vl",
    },
}


@dataclass
class Config:
    """Main configuration for primitives solver."""
    
    # API Settings
    api_key: str = field(default_factory=lambda: os.environ.get("CHUTES_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1"))
    max_concurrency: int = 10
    
    # Models (defaults to "balanced" preset)
    reasoner_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name=AVAILABLE_MODELS["kimi"],
        max_tokens=8192,
        temperature=0.4
    ))
    coder_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name=AVAILABLE_MODELS["qwen-30b"],
        max_tokens=4096,
        temperature=0.2
    ))
    vlm_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name=AVAILABLE_MODELS["qwen-vl"],
        max_tokens=4096,
        temperature=0.3
    ))
    
    # Paths
    data_dir: Path = field(default_factory=lambda: ARC_ROOT / "ARC-AGI-2" / "data")
    logs_dir: Path = field(default_factory=lambda: Path("./logs"))
    filmstrips_dir: Path = field(default_factory=lambda: Path("./logs/filmstrips"))
    
    # Solver Settings
    max_retries_per_step: int = 3
    use_vlm_verification: bool = True
    use_visual_planning: bool = True  # Use VLM for initial planning
    use_ensemble_planning: bool = False  # NEW: Use VLM+LLM dual-path ensemble
    max_program_length: int = 20
    
    def __post_init__(self):
        """Create directories."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.filmstrips_dir.mkdir(parents=True, exist_ok=True)
    
    def set_preset(self, preset: str) -> None:
        """Apply a model preset."""
        if preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(MODEL_PRESETS.keys())}")
        
        p = MODEL_PRESETS[preset]
        self.reasoner_model = ModelConfig(
            name=AVAILABLE_MODELS[p["reasoner"]],
            max_tokens=8192,
            temperature=0.4
        )
        self.coder_model = ModelConfig(
            name=AVAILABLE_MODELS[p["coder"]],
            max_tokens=4096,
            temperature=0.2
        )
        self.vlm_model = ModelConfig(
            name=AVAILABLE_MODELS[p["vlm"]],
            max_tokens=4096,
            temperature=0.3
        )


def load_config(preset: str = "balanced") -> Config:
    """Load configuration with optional preset."""
    config = Config()
    config.set_preset(preset)
    return config

