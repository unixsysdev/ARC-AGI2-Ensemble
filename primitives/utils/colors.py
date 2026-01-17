"""Colored logging utilities for ARC solver."""

import logging
import sys


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Standard colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    
    # Backgrounds
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors based on log level and content."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM,
        logging.INFO: Colors.RESET,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE,
    }
    
    KEYWORD_COLORS = {
        # Success indicators
        "✓": Colors.BRIGHT_GREEN,
        "✅": Colors.BRIGHT_GREEN,
        "PASSED": Colors.BRIGHT_GREEN,
        "SUCCESS": Colors.BRIGHT_GREEN,
        "VALID": Colors.BRIGHT_GREEN,
        "WINNER": Colors.BRIGHT_GREEN,
        
        # Warning indicators
        "⚠️": Colors.BRIGHT_YELLOW,
        "WARNING": Colors.YELLOW,
        "FAILED": Colors.BRIGHT_YELLOW,
        
        # Error indicators
        "❌": Colors.BRIGHT_RED,
        "ERROR": Colors.RED,
        "INVALID": Colors.RED,
        "CRASHED": Colors.RED,
        
        # Info indicators
        "💡": Colors.BRIGHT_CYAN,
        "PARTIAL SUCCESS": Colors.BRIGHT_CYAN,
        "INTERPRETER": Colors.CYAN,
        "CRITIC": Colors.MAGENTA,
        "ENSEMBLE": Colors.BLUE,
        "VLM": Colors.BRIGHT_MAGENTA,
        "LLM": Colors.BRIGHT_BLUE,
        "META-REVIEWER": Colors.BRIGHT_YELLOW,
        "FREE-FORM": Colors.BRIGHT_CYAN,
        "FEEDBACK": Colors.YELLOW,
        
        # Step indicators
        "Step 1": Colors.CYAN,
        "Step 2": Colors.CYAN,
        "Step 3": Colors.CYAN,
        "Attempt": Colors.BOLD,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Get base format
        message = super().format(record)
        
        # Apply level color
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        
        # Apply keyword colors
        for keyword, color in self.KEYWORD_COLORS.items():
            if keyword in message:
                message = message.replace(keyword, f"{color}{keyword}{Colors.RESET}")
        
        # Color the log level
        level_name = record.levelname
        if record.levelno == logging.WARNING:
            message = message.replace(f"[{level_name}]", f"[{Colors.YELLOW}{level_name}{Colors.RESET}]")
        elif record.levelno == logging.ERROR:
            message = message.replace(f"[{level_name}]", f"[{Colors.RED}{level_name}{Colors.RESET}]")
        elif record.levelno == logging.INFO:
            message = message.replace(f"[INFO]", f"[{Colors.DIM}INFO{Colors.RESET}]")
        
        return message


def setup_colored_logging(level: int = logging.INFO):
    """Set up colored logging for the entire application."""
    # Create colored formatter
    formatter = ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create handler with colored formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    return root_logger
