"""Configuration for ARC Solver."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Available models on Chutes API (verified working)
AVAILABLE_MODELS = {
    # Reasoners (for instruction generation)
    "deepseek": "deepseek-ai/DeepSeek-V3.2-Speciale-TEE",
    "kimi": "moonshotai/Kimi-K2-Instruct",  # Fallback reasoner
    "qwen-reasoning": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
    
    # Coders (for Python generation)
    "qwen-coder": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
    "qwen-coder-small": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    
    # Vision (for VLM critic)
    "qwen-vl": "Qwen/Qwen3-VL-235B-A22B-Instruct",
}

# Fallback chains (try in order)
REASONER_FALLBACK_CHAIN = ["deepseek", "kimi", "qwen-reasoning"]
CODER_FALLBACK_CHAIN = ["qwen-coder", "qwen-coder-small"]

# Default model selections
DEFAULT_REASONER = "deepseek"  # DeepSeek for reasoning/instruction execution
DEFAULT_CODER = "qwen-coder"   # Qwen for code generation
DEFAULT_VLM = "qwen-vl"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class Config:
    """Main configuration."""
    
    # API Settings
    api_key: str = field(default_factory=lambda: os.environ.get("CHUTES_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1"))
    max_concurrency: int = field(default_factory=lambda: int(os.environ.get("MAX_CONCURRENCY", "20")))
    
    # Local LLM Settings
    local_url: str = field(default_factory=lambda: os.environ.get("LOCAL_LLM_URL", "http://localhost:8000"))
    use_local: bool = False
    local_concurrency: int = 32  # Parallel requests to local LLM (match vLLM max-num-seqs)
    local_batch_size: int = 32   # Number of candidates per generation
    
    # Hybrid Mode Settings (local exploration + remote refinement)
    hybrid_mode: bool = False
    hybrid_local_candidates: int = 32  # Generate this many with local (1 batch of 32)
    hybrid_top_k: int = 5  # Keep top K for remote refinement
    hybrid_remote_revisions: int = 2  # Revisions per top candidate
    
    # Feedback Loop Settings (teach local from remote's fixes)
    feedback_to_local: bool = True  # Send remote's fix feedback to local for learning
    feedback_local_attempts: int = 16  # How many local attempts after feedback
    
    # Models
    reasoner_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name=AVAILABLE_MODELS[DEFAULT_REASONER],
        max_tokens=8192,
        temperature=0.7
    ))
    coder_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name=AVAILABLE_MODELS[DEFAULT_CODER],
        max_tokens=8192,
        temperature=0.6
    ))
    vlm_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name=AVAILABLE_MODELS[DEFAULT_VLM],
        max_tokens=2048,
        temperature=0.3
    ))
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    attempts_dir: Path = field(default_factory=lambda: Path("./attempts"))
    logs_dir: Path = field(default_factory=lambda: Path("./logs"))
    
    # Solver Settings
    initial_candidates: int = 30  # Number of initial candidates to generate
    top_k_for_revision: int = 5   # Top candidates to revise
    revision_per_candidate: int = 3  # Revisions per top candidate
    max_attempts_per_task: int = 100  # Max total attempts before giving up
    code_execution_timeout: float = 5.0  # Seconds
    
    def __post_init__(self):
        # Create directories
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def set_reasoner(self, model_key: str):
        """Set reasoner model by key."""
        if model_key in AVAILABLE_MODELS:
            self.reasoner_model = ModelConfig(
                name=AVAILABLE_MODELS[model_key],
                max_tokens=8192,
                temperature=0.7
            )
    
    def set_coder(self, model_key: str):
        """Set coder model by key."""
        if model_key in AVAILABLE_MODELS:
            self.coder_model = ModelConfig(
                name=AVAILABLE_MODELS[model_key],
                max_tokens=8192,
                temperature=0.6
            )


def load_config(
    reasoner: str | None = None,
    coder: str | None = None,
    use_local: bool = False,
    local_url: str | None = None,
    hybrid_mode: bool = False,
    local_concurrency: int = 32,
    hybrid_candidates: int = 64,
    feedback_to_local: bool = True  # Enable feedback loop by default
) -> Config:
    """Load configuration with optional model overrides."""
    config = Config()
    
    if reasoner:
        config.set_reasoner(reasoner)
    if coder:
        config.set_coder(coder)
    if use_local:
        config.use_local = True
    if local_url:
        config.local_url = local_url
    if hybrid_mode:
        config.hybrid_mode = True
        config.use_local = True  # Hybrid requires local
    
    # Set concurrency/batching
    config.local_concurrency = local_concurrency
    config.local_batch_size = local_concurrency  # Match batch to concurrency
    config.hybrid_local_candidates = hybrid_candidates
    
    # Set feedback loop
    config.feedback_to_local = feedback_to_local
    
    return config
