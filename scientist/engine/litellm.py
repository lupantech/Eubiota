# TODO: The current implementation is not based on textgrad, but rather a direct implementation of the LiteLLM API.
# Detached from textgrad: https://github.com/zou-group/textgrad/blob/main/textgrad/engine_experimental/litellm.py

try:
    import litellm
    from litellm import supports_reasoning
except ImportError:
    raise ImportError("If you'd like to use LiteLLM, please install the litellm package by running `pip install litellm`, and set appropriate API keys for the models you want to use.")

import os
import json
import base64
import platformdirs
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union, Optional, Any, Dict

from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes

def validate_structured_output_model(model_string: str) -> bool:
    """
    Check if the model supports structured outputs.
    
    Args:
        model_string: The name of the model to check
        
    Returns:
        True if the model supports structured outputs, False otherwise
    """
    # Models that support structured outputs
    structure_output_models = [
        "gpt-4", 
        "claude-opus-4", "claude-sonnet-4", "claude-3.7-sonnet", "claude-3.5-sonnet", "claude-3-opus",
        "gemini-",
    ]
    return any(x in model_string.lower() for x in structure_output_models)

def validate_chat_model(model_string: str) -> bool:
    # 99% of LiteLLM models are chat models
    return True


def validate_reasoning_model(model_string: str) -> bool:
    """
    Check if the model is a reasoning model.
    Includes OpenAI o1/o3/o4 variants (non-pro), Claude models, and other LLMs known for reasoning.
    """
    m = model_string.lower()
    if supports_reasoning(model_string):
        return True

    # Hard ways
    if any(x in m for x in ["o1", "o3", "o4"]) and not validate_pro_reasoning_model(model_string):
        return True

    if "claude" in m and not validate_pro_reasoning_model(model_string):
        return True

    extra = ["qwen-72b", "llama-3-70b", "mistral-large", "deepseek-reasoner", "xai/grok-3", "gemini-2.5-pro"]
    if any(e in model_string.lower() for e in extra):
        return True

    return False

def validate_pro_reasoning_model(model_string: str) -> bool:
    """
    Check if the model is a pro reasoning model:
    OpenAI o1-pro, o3-pro, o4-pro, and Claude-4/Sonnet variants.
    """
    m = model_string.lower()
    if any(x in m for x in ["o1-pro", "o3-pro", "o4-pro"]):
        return True
    if any(x in m for x in ["claude-opus-4", "claude-sonnet-4", "claude-3.7-sonnet"]):
        return True
    return False

def validate_multimodal_model(model_string: str) -> bool:
    """
    Check if the model supports multimodal inputs.

    Args:
        model_string: The name of the model to check

    Returns:
        True if the model supports multimodal inputs, False otherwise
    """
    m = model_string.lower()

    # Core multimodal models
    multimodal_models = [
        "gpt-4-vision", "gpt-4o", "gpt-4.1",  # OpenAI multimodal
        "gpt-4v",                            # alias for vision-capable GPT-4
        "claude-sonnet", "claude-opus",     # Claude multimodal variants
        "gemini",                            # Base Gemini models are multimodal :contentReference[oaicite:0]{index=0}
        "gpt-4v",                            # repeats for clarity
        "llama-4",                           # reported as multimodal
        "qwen-vl", "qwen2-vl",              # Qwen vision-language models
    ]

    # Add Gemini TTS / audio-capable variants (though audio is modality)
    audio_models = ["-tts", "-flash-preview-tts", "-pro-preview-tts"]
    if any(g in m for g in multimodal_models):
        return True
    
    if "gemini" in m and any(s in m for s in audio_models):
        return True  # E.g. gemini-2.5-flash-preview-tts
    
    # Make sure we catch edge cases like "gpt-4v" or "gpt-4 vision"
    if "vision" in m or "vl" in m:
        return True

    return False

class ChatLiteLLM(EngineLM, CachedEngine):
    """
    LiteLLM implementation of the EngineLM interface.
    This allows using any model supported by LiteLLM.
    """
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "gpt-3.5-turbo",
        use_cache: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        **kwargs
    ):
        """
        Initialize the LiteLLM engine.
        
        Args:
            model_string: The name of the model to use
            use_cache: Whether to use caching
            system_prompt: The system prompt to use
            is_multimodal: Whether to enable multimodal capabilities
            **kwargs: Additional arguments to pass to the LiteLLM client
        """
        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal or validate_multimodal_model(model_string)
        self.kwargs = kwargs
        
        # Set up caching if enabled
        if self.use_cache:
            root = platformdirs.user_cache_dir("scientist")
            cache_path = os.path.join(root, f"cache_litellm_{model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)
        
        # Disable telemetry
        litellm.telemetry = False
        
        # Set model capabilities based on model name
        self.support_structured_output = validate_structured_output_model(self.model_string)
        self.is_chat_model = validate_chat_model(self.model_string)
        self.is_reasoning_model = validate_reasoning_model(self.model_string)
        self.is_pro_reasoning_model = validate_pro_reasoning_model(self.model_string)
        
        # Suppress LiteLLM debug logs
        litellm.suppress_debug_info = True
        for key in logging.Logger.manager.loggerDict.keys():
            if "litellm" in key.lower():
                logging.getLogger(key).setLevel(logging.WARNING)

    def __call__(self, prompt, **kwargs):
        """
        Handle direct calls to the instance (e.g., model(prompt)).
        Forwards the call to the generate method.
        """
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[Dict[str, Any]]:
        """
        Format content for the LiteLLM API.
        
        Args:
            content: List of content items (strings and/or image bytes)
            
        Returns:
            Formatted content for the LiteLLM API
        """
        formatted_content = []
        for item in content:
            if isinstance(item, str):
                formatted_content.append({"type": "text", "text": item})
            elif isinstance(item, bytes):
                # For images, encode as base64
                image_type = get_image_type_from_bytes(item)
                if image_type:
                    base64_image = base64.b64encode(item).decode('utf-8')
                    formatted_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{base64_image}",
                            "detail": "auto"
                        }
                    })
            elif isinstance(item, dict) and "type" in item:
                # Already formatted content
                formatted_content.append(item)
        return formatted_content

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        """
        Generate text from a prompt.
        
        Args:
            content: A string prompt or a list of strings and image bytes
            system_prompt: Optional system prompt to override the default
            **kwargs: Additional arguments to pass to the LiteLLM API
            
        Returns:
            Generated text response
        """
        try:
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            
            elif isinstance(content, list):
                has_multimodal_input = any(isinstance(item, bytes) for item in content)
                if (has_multimodal_input) and (not self.is_multimodal):
                    raise NotImplementedError(f"Multimodal generation is only supported for multimodal models. Current model: {self.model_string}")
                
                return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)
        except litellm.exceptions.BadRequestError as e:
            print(f"Bad request error: {str(e)}")
            return {
                "error": "bad_request",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except litellm.exceptions.RateLimitError as e:
            print(f"Rate limit error encountered: {str(e)}")
            return {
                "error": "rate_limit",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except litellm.exceptions.ContextWindowExceededError as e:
            print(f"Context window exceeded: {str(e)}")
            return {
                "error": "context_window_exceeded",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except litellm.exceptions.APIError as e:
            print(f"API error: {str(e)}")
            return {
                "error": "api_error",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except litellm.exceptions.APIConnectionError as e:
            print(f"API connection error: {str(e)}")
            return {
                "error": "api_connection_error",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
    
    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None, **kwargs
    ):
        """
        Generate text from a text prompt.
        
        Args:
            prompt: The text prompt
            system_prompt: Optional system prompt to override the default
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            top_p: Controls diversity via nucleus sampling
            response_format: Optional response format for structured outputs
            **kwargs: Additional arguments to pass to the LiteLLM API
            
        Returns:
            Generated text response
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": prompt},
        ]
        
        # Prepare additional parameters
        params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        
        # Add response_format if supported and provided
        if self.support_structured_output and response_format:
            params["response_format"] = response_format
            
        # Add any additional kwargs
        params.update(self.kwargs)
        params.update(kwargs)
        
        # Make the API call
        response = litellm.completion(
            model=self.model_string,
            messages=messages,
            **params
        )
        
        response_text = response.choices[0].message.content
        
        if self.use_cache:
            self._save_cache(cache_key, response_text)
        
        return response_text
    
    def _generate_multimodal(
        self, content_list, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, **kwargs
    ):
        """
        Generate text from a multimodal prompt (text and images).
        
        Args:
            content_list: List of content items (strings and/or image bytes)
            system_prompt: Optional system prompt to override the default
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            top_p: Controls diversity via nucleus sampling
            **kwargs: Additional arguments to pass to the LiteLLM API
            
        Returns:
            Generated text response
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content_list)
        
        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(str(formatted_content))
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
        
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": formatted_content},
        ]
        
        # Prepare additional parameters
        params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        
        # Add any additional kwargs
        params.update(self.kwargs)
        params.update(kwargs)
        
        # Make the API call
        response = litellm.completion(
            model=self.model_string,
            messages=messages,
            **params
        )
        
        response_text = response.choices[0].message.content
        
        if self.use_cache:
            self._save_cache(cache_key, response_text)
        
        return response_text