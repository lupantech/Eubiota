import os
import json
import base64
import platformdirs
from typing import List, Union, Dict, Any, TypeVar
from openai import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes

T = TypeVar('T', bound='BaseModel')

def validate_structured_output_model(model_string: str) -> bool:
    """Check if the model supports structured outputs."""
    # Azure OpenAI models that support structured outputs
    return any(x in model_string.lower() for x in ["gpt-4"])

def validate_chat_model(model_string: str) -> bool:
    """Check if the model is a chat model."""
    return any(x in model_string.lower() for x in ["gpt"])

def validate_reasoning_model(model_string: str) -> bool:
    """Check if the model is a reasoning model."""
    # Azure OpenAI doesn't have specific reasoning models like OpenAI
    return False

def validate_pro_reasoning_model(model_string: str) -> bool:
    """Check if the model is a pro reasoning model."""
    # Azure OpenAI doesn't have pro reasoning models
    return False

class ChatAzureOpenAI(EngineLM, CachedEngine):
    """
    Azure OpenAI API implementation of the EngineLM interface.
    """
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "gpt-4",
        use_cache: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        **kwargs
    ):
        """
        Initialize the Azure OpenAI engine.
        
        Args:
            model_string: The name of the Azure OpenAI deployment
            use_cache: Whether to use caching
            system_prompt: The system prompt to use
            is_multimodal: Whether to enable multimodal capabilities
            **kwargs: Additional arguments to pass to the AzureOpenAI client
        """
        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        # Set model capabilities
        self.support_structured_output = validate_structured_output_model(self.model_string)
        self.is_chat_model = validate_chat_model(self.model_string)
        self.is_reasoning_model = validate_reasoning_model(self.model_string)
        self.is_pro_reasoning_model = validate_pro_reasoning_model(self.model_string)

        # Set up caching if enabled
        if self.use_cache:
            root = platformdirs.user_cache_dir("scientist")
            cache_path = os.path.join(root, f"cache_azure_openai_{model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)

        # Validate required environment variables
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            raise ValueError("Please set the AZURE_OPENAI_API_KEY environment variable.")
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            raise ValueError("Please set the AZURE_OPENAI_ENDPOINT environment variable.")

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        
        # Set default kwargs
        self.default_kwargs = kwargs

    def __call__(self, prompt, **kwargs):
        """
        Handle direct calls to the instance (e.g., model(prompt)).
        Forwards the call to the generate method.
        """
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[Dict[str, Any]]:
        """Format content for the OpenAI API."""
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

    @retry(
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_attempt(5),
    )
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            elif isinstance(content, list):
                if not self.is_multimodal:
                    raise NotImplementedError(f"Multimodal generation is not supported for {self.model_string}.")
                return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)
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
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: int = 4000,
        top_p: float = 0.99,
        response_format: dict = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the Azure OpenAI API.
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        
        # Chat models with structured output format
        if self.is_chat_model and self.support_structured_output and response_format is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format=response_format,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            response = response.choices[0].message.parsed

        # Chat models without structured outputs
        elif self.is_chat_model and (not self.support_structured_output or response_format is None):
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            response = response.choices[0].message.content

        # Reasoning models: currently only supports base response
        elif self.is_reasoning_model:
            print(f"Using reasoning model: {self.model_string}")
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens,
                reasoning_effort="medium",
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            # Workaround for handling length finish reason
            if hasattr(response.choices[0], 'finish_reason') and response.choices[0].finish_reason == "length":
                response = "Token limit exceeded"
            else:
                response = response.choices[0].message.content
                
        # Fallback for other model types
        else:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            response = response.choices[0].message.content
        
        # Cache the response if caching is enabled
        if self.use_cache:
            self._add_to_cache(cache_key, response)
            
        return response

    def _generate_multimodal(
        self,
        content: List[Union[str, bytes]],
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: int = 4000,
        top_p: float = 0.99,
        response_format: dict = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from multiple input types (text and images).
        """
        if not self.is_multimodal:
            raise ValueError("Multimodal input is not supported by this model.")
            
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
        
        
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": formatted_content},
        ]
        
        # Chat models with structured output format
        if self.is_chat_model and self.support_structured_output and response_format is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_string,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format=response_format,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            response_content = response.choices[0].message.parsed
        
        # Standard chat completion
        elif self.is_chat_model and (not self.support_structured_output or response_format is None):
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            response_content = response.choices[0].message.content
            
        # Reasoning models: currently only supports base response
        elif self.is_reasoning_model:
            print(f"Using reasoning model: {self.model_string}")
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "user", "content": formatted_content},
                ],
                max_completion_tokens=max_tokens,
                reasoning_effort="medium",
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            # Workaround for handling length finish reason
            if hasattr(response.choices[0], 'finish_reason') and response.choices[0].finish_reason == "length":
                response_content = "Token limit exceeded"
            else:
                response_content = response.choices[0].message.content
                
        # Fallback for other model types
        else:
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            response_content = response.choices[0].message.content
        
        # Cache the response if caching is enabled
        if self.use_cache:
            self._add_to_cache(cache_key, response_content)
            
        return response_content
