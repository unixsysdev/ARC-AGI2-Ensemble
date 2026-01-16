"""Local LLM client for llama.cpp server or Ollama."""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """
    Client for local LLM servers (llama.cpp server, Ollama, vLLM, etc.)
    
    These all use OpenAI-compatible API format, so this is similar to ChutesClient
    but optimized for local inference (no auth, faster timeouts, higher concurrency).
    
    Usage:
        # Start vLLM server:
        # vllm serve Qwen/Qwen3-30B-A3B-GPTQ-Int4 --host 0.0.0.0 --port 8000
        
        client = LocalLLMClient("http://localhost:8000")
        response = await client.chat(messages, temperature=0.7)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "auto",  # "auto" = detect from /v1/models
        max_concurrency: int = 4,
        timeout: float = 600.0  # 10 min timeout for large models like GPT-OSS-120B
    ):
        self.base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._detected_model: str | None = None
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client: httpx.AsyncClient | None = None
    
    def _ensure_client(self):
        """Lazily create HTTP client on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
    
    async def _detect_model(self) -> str:
        """Auto-detect model name from vLLM /v1/models endpoint."""
        if self._detected_model:
            return self._detected_model
        
        if self._model_name != "auto":
            return self._model_name
        
        try:
            self._ensure_client()
            resp = await self._client.get(f"{self.base_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data") and len(data["data"]) > 0:
                    self._detected_model = data["data"][0]["id"]
                    logger.info(f"Auto-detected local model: {self._detected_model}")
                    return self._detected_model
        except Exception as e:
            logger.warning(f"Failed to detect model: {e}")
        
        # Fallback
        return "local"
    
    async def health_check(self) -> bool:
        """Check if local server is running."""
        try:
            self._ensure_client()
            resp = await self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: list[str] | None = None
    ) -> str:
        """Send chat completion request to local server."""
        async with self._semaphore:
            self._ensure_client()  # Auto-initialize on first use
            
            # Auto-detect model name on first request
            model = await self._detect_model()
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "reasoning_effort": "low"  # For reasoning models like GPT-OSS
            }
            
            if stop:
                payload["stop"] = stop
            
            logger.debug(f"Local LLM request: {len(messages)} messages")
            
            response = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            # Log raw response for debugging
            raw_text = response.text
            logger.debug(f"Raw response ({len(raw_text)} bytes): {raw_text[:500]}...")
            
            data = response.json()
            
            # Safely extract content with detailed error logging
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to parse vLLM response: {e}")
                logger.error(f"Response data: {str(data)[:500]}")
                return ""
            
            # Handle null content from vLLM - this model puts reasoning in separate field
            if content is None:
                message = data.get("choices", [{}])[0].get("message", {})
                logger.warning(f"vLLM null content. Message keys: {list(message.keys())}")
                
                # Check both reasoning fields for embedded code
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                
                # Try to extract code from reasoning if it contains def transform
                import re
                if "def transform" in reasoning or "def " in reasoning:
                    logger.info("Found potential code in reasoning, extracting...")
                    content = reasoning
                else:
                    logger.warning(f"No code in reasoning. First 200 chars: {reasoning[:200]}")
                    return ""
            
            logger.debug(f"Local LLM response: {len(content)} chars - {content[:200]}...")
            return content
    
    async def batch_chat(
        self,
        prompts: list[list[dict[str, Any]]],
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> list[str | Exception]:
        """Send multiple chat requests in parallel."""
        tasks = [
            self.chat(messages, temperature, max_tokens)
            for messages in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def parse_json_response(self, text: str) -> dict | list | None:
        """Extract JSON from response text."""
        # Try to find JSON block in markdown
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
        except json.JSONDecodeError:
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start != -1 and end > start:
                    try:
                        return json.loads(text[start:end+1])
                    except json.JSONDecodeError:
                        continue
            return None


class LocalCodeGenerator:
    """
    Code generator using local LLM for high-volume exploration.
    
    Same interface as CodeGenerator but uses local model.
    """
    
    def __init__(self, client: LocalLLMClient):
        self.client = client
    
    def _format_task(self, task) -> str:
        """Format task examples as string."""
        from ..models.task import grid_to_ascii
        
        lines = []
        for i, pair in enumerate(task.train):
            lines.append(f"Example {i+1}:")
            lines.append("Input:")
            lines.append(grid_to_ascii(pair.input))
            lines.append("Output:")
            lines.append(grid_to_ascii(pair.output))
            lines.append("")
        return "\n".join(lines)
    
    async def generate(
        self,
        task,  # Task object
        n: int = 5,
        temperature: float = 0.7
    ) -> list[str]:
        """Generate n Python code candidates."""
        
        task_examples = self._format_task(task)
        
        prompt = f"""You are an expert Python programmer solving ARC-AGI puzzles.

Analyze these input/output examples and write a transform function:

{task_examples}

Write a Python function that transforms inputs to outputs.
The function signature MUST be: def transform(grid: list[list[int]]) -> list[list[int]]
Use numpy if helpful (imported as np).

```python
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    # Your code here
```"""

        messages = [{"role": "user", "content": prompt}]
        
        # Generate n codes with varying temperatures for diversity
        logger.info(f"LocalCodeGenerator: Starting {n} parallel code generations...")
        tasks = []
        for i in range(n):
            t = temperature + (i * 0.05)
            t = min(t, 1.2)
            tasks.append(self.client.chat(messages, temperature=t, max_tokens=4096))
        
        # Wait for all requests with progress tracking
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"LocalCodeGenerator: All {n} requests completed")
        
        # Extract valid codes
        codes = []
        extraction_failed = 0
        for i, r in enumerate(results):
            if isinstance(r, str):
                code = self._extract_code(r)
                if code:
                    codes.append(code)
                else:
                    extraction_failed += 1
            elif isinstance(r, Exception):
                logger.warning(f"Local code generation {i+1}/{n} failed: {type(r).__name__}: {r}")
        
        logger.info(f"LocalCodeGenerator: Extracted {len(codes)} valid codes ({extraction_failed} extraction failures)")
        return codes
    
    def _extract_code(self, response: str) -> str | None:
        """Extract Python code from response."""
        import re
        
        if not response:
            return None
        
        # Try to extract from markdown code blocks first
        patterns = [
            r"```python\s*(.*?)```",
            r"```\s*(.*?)```",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                # Accept any function definition
                if re.search(r"def\s+\w+\s*\(", code):
                    # Normalize function name to 'transform'
                    code = re.sub(r"def\s+\w+\s*\(", "def transform(", code, count=1)
                    if "import numpy" not in code:
                        code = "import numpy as np\n\n" + code
                    return code
        
        # Try raw response if it looks like code (any def statement)
        if re.search(r"def\s+\w+\s*\(", response):
            code = response
            # Normalize function name
            code = re.sub(r"def\s+\w+\s*\(", "def transform(", code, count=1)
            if "import numpy" not in code:
                code = "import numpy as np\n\n" + code
            return code
        
        return None
    
    async def revise_with_feedback(
        self,
        task,  # Task object
        failed_code: str,
        errors: list[str],
        successful_code: str,
        n: int = 16,
        temperature: float = 0.7
    ) -> list[str]:
        """
        Generate revised code using feedback from remote's successful solution.
        
        This enables the local model to learn from the remote model's fix
        without just copying the solution.
        """
        task_examples = self._format_task(task)
        
        # Build feedback prompt
        prompt = f"""You are learning to solve ARC-AGI puzzles. 

Here's a puzzle and examples:

{task_examples}

Your previous attempt FAILED:
```python
{failed_code}
```

The errors were:
{chr(10).join(f'- {e}' for e in errors[:3])}

Here's a WORKING solution for reference:
```python
{successful_code}
```

Study what the working solution does differently. Then write your OWN improved version.
Do NOT just copy the solution - understand the pattern and implement it your way.

```python
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    # Your improved code here
```"""

        messages = [{"role": "user", "content": prompt}]
        
        logger.info(f"LocalCodeGenerator: Revising with feedback ({n} attempts)...")
        tasks = []
        for i in range(n):
            t = temperature + (i * 0.05)
            t = min(t, 1.2)
            tasks.append(self.client.chat(messages, temperature=t, max_tokens=4096))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        codes = []
        for i, r in enumerate(results):
            if isinstance(r, str):
                code = self._extract_code(r)
                if code:
                    codes.append(code)
        
        logger.info(f"LocalCodeGenerator: Feedback revision extracted {len(codes)} codes")
        return codes
