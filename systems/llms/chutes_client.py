"""OpenAI-compatible async client for Chutes API."""

from __future__ import annotations
import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Config, ModelConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter that ensures minimum time between requests."""
    
    def __init__(self, min_interval: float = 1.0):
        """Initialize rate limiter.
        
        Args:
            min_interval: Minimum seconds between requests (default: 1.0)
        """
        self.min_interval = min_interval
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until we can make a request (respecting rate limit)."""
        async with self._lock:
            now = time.time()
            time_since_last = now - self._last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            self._last_request_time = time.time()


class ChutesClient:
    """Async client for Chutes LLM API (OpenAI-compatible)."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._rate_limiter = RateLimiter(min_interval=getattr(config, 'min_request_interval', 1.0))
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "ChutesClient":
        self._client = httpx.AsyncClient(timeout=300.0)
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    async def chat(
        self,
        model: ModelConfig,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None
    ) -> str:
        """Send chat completion request."""
        async with self._semaphore:
            # Rate limit - wait for minimum interval between requests
            await self._rate_limiter.acquire()
            
            if not self._client:
                raise RuntimeError("Client not initialized. Use async context manager.")
            
            payload = {
                "model": model.name,
                "messages": messages,
                "temperature": temperature or model.temperature,
                "max_tokens": max_tokens or model.max_tokens,
            }
            
            if response_format:
                payload["response_format"] = response_format
            
            logger.debug(f"Requesting {model.name}: {len(messages)} messages")
            
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            logger.debug(f"Response from {model.name}: {len(content)} chars")
            return content
    
    async def chat_with_fallback(
        self,
        model: ModelConfig,
        messages: list[dict[str, Any]],
        fallback_models: list[str],
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> str:
        """Send chat with fallback to alternate models on failure."""
        from ..config import AVAILABLE_MODELS
        
        # Try primary model first
        try:
            return await self.chat(model, messages, temperature, max_tokens)
        except Exception as e:
            logger.warning(f"Primary model {model.name} failed: {e}")
        
        # Try fallback models
        for fallback_key in fallback_models:
            if fallback_key not in AVAILABLE_MODELS:
                continue
            fallback_name = AVAILABLE_MODELS[fallback_key]
            if fallback_name == model.name:
                continue  # Skip if same as primary
            
            logger.info(f"Trying fallback model: {fallback_name}")
            fallback_model = ModelConfig(
                name=fallback_name,
                max_tokens=max_tokens or model.max_tokens,
                temperature=temperature or model.temperature
            )
            
            try:
                return await self.chat(fallback_model, messages)
            except Exception as e:
                logger.warning(f"Fallback model {fallback_name} failed: {e}")
        
        raise RuntimeError(f"All models failed for request")
    
    async def chat_with_image(
        self,
        model: ModelConfig,
        text_prompt: str,
        image_path: Path | str,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> str:
        """Send chat completion with an image (for VLM)."""
        # Read and base64 encode the image
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp"
        }.get(suffix, "image/png")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ]
        
        return await self.chat(model, messages, temperature, max_tokens)
    
    async def batch_chat(
        self,
        model: ModelConfig,
        prompts: list[list[dict[str, Any]]],
        temperature: float | None = None
    ) -> list[str]:
        """Send multiple chat requests in parallel."""
        tasks = [
            self.chat(model, messages, temperature)
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
            # Try to find JSON object or array
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start != -1 and end > start:
                    try:
                        return json.loads(text[start:end+1])
                    except json.JSONDecodeError:
                        continue
            return None
