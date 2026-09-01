"""
core/model_discovery.py — Dynamic LLM model discovery and fallback selector.

Automatically discovers live, warm, and free models from the active LLM provider
(Hugging Face, NVIDIA NIM, Groq, OpenRouter, Featherless) so that models are never
hard-coded or fail due to provider pool changes.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional

from openai import OpenAI

log = logging.getLogger(__name__)


def discover_available_models(
    client: Optional[OpenAI] = None,
    base_url: str = "",
    api_key: str = "",
) -> list[str]:
    """
    Dynamically discover all working models available on the active provider.
    Returns an ordered list of model IDs, prioritizing established instruct models.
    """
    base_lower = base_url.lower()
    discovered: list[str] = []

    # 1. Hugging Face dynamic serverless warm models discovery
    if "huggingface.co" in base_lower or api_key.startswith("hf_"):
        try:
            url = "https://huggingface.co/api/models?inference=warm&pipeline_tag=text-generation"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            res = urllib.request.urlopen(req, timeout=4)
            data = json.loads(res.read().decode("utf-8"))


            priority_names = [
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "google/gemma-2-9b-it",
            ]
            warm_ids = {m.get("id") for m in data if m.get("id")}


            for p in priority_names:
                if p in warm_ids and p not in discovered:
                    discovered.append(p)


            for m in data:
                mid = m.get("id", "")
                if mid and mid not in discovered:
                    m_low = mid.lower()
                    if ("instruct" in m_low or "chat" in m_low) and any(
                        fam in m_low for fam in ["llama", "mistral", "qwen", "deepseek", "gemma", "phi"]
                    ):
                        discovered.append(mid)
        except Exception as exc:
            log.warning(f"Dynamic HF model discovery failed: {exc}")


    # 2. Query client.models.list() via standard OpenAI API endpoint
    if client is not None and not discovered:
        try:
            model_list = client.models.list()
            for m in getattr(model_list, "data", []):
                mid = getattr(m, "id", None)
                if mid and mid not in discovered:
                    discovered.append(mid)
        except Exception as exc:
            log.debug(f"client.models.list() query failed: {exc}")


    # 3. Fallback defaults if discovery network request fails
    if not discovered:
        if "nvidia.com" in base_lower or api_key.startswith("nvapi-"):
            discovered = [
                "meta/llama-3.1-8b-instruct",
                "mistralai/mistral-7b-instruct-v0.3",
                "deepseek-ai/deepseek-r1",
            ]
        elif "groq.com" in base_lower or api_key.startswith("gsk_"):
            discovered = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "deepseek-r1-distill-llama-70b",
            ]
        elif "openrouter.ai" in base_lower or api_key.startswith("sk-or-"):
            discovered = [
                "meta-llama/llama-3.1-8b-instruct:free",
                "mistralai/mistral-7b-instruct:free",
                "qwen/qwen-2.5-7b-instruct:free",
            ]
        elif "huggingface.co" in base_lower or api_key.startswith("hf_"):
            discovered = [
                "meta-llama/Llama-3.1-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            ]
        else:
            discovered = [
                "meta-llama/Llama-3.1-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            ]

    return discovered
