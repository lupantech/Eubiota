# Ref: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/gemini.py
# Ref: https://ai.google.dev/gemini-api/docs/quickstart?lang=python
# Changed to use the new google-genai package May 25, 2025
# Ref: https://ai.google.dev/gemini-api/docs/migrate

try:
    # import google.generativeai as genai
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("If you'd like to use Gemini models, please install the google-genai package by running `pip install google-genai`, and add 'GOOGLE_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import base64
import json
from typing import List, Union
from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes
import io
from PIL import Image

class ChatGemini(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="gemini-pro",
        use_cache: bool=False,
        system_prompt=SYSTEM_PROMPT,
        is_multimodal: bool=False,
    ):
        self.use_cache = use_cache
        self.model_string = model_string
        self.system_prompt = system_prompt
        assert isinstance(self.system_prompt, str)
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("scientist")
            cache_path = os.path.join(root, f"cache_gemini_{model_string}.db")
            super().__init__(cache_path=cache_path)
            
        if os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable if you'd like to use Gemini models.")
        
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, str):
                return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
            
            elif isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    full_text = "\n".join(content)
                    return self._generate_from_single_prompt(full_text, system_prompt=system_prompt, **kwargs)

                elif any(isinstance(item, bytes) for item in content):
                    if not self.is_multimodal:
                        raise NotImplementedError(
                            f"Multimodal generation is only supported for {self.model_string}. "
                            "Consider using a multimodal model like 'gpt-4o'."
                        )
                    return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

                else:
                    raise ValueError("Unsupported content in list: only str or bytes are allowed.")
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }


    def _generate_from_single_prompt(
        self, prompt: str, system_prompt=None, temperature=0., max_tokens=4000, top_p=0.99, **kwargs
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_or_none = self._check_cache(sys_prompt_arg + prompt)
            if cache_or_none is not None:
                return cache_or_none

        # messages = [{'role': 'user', 'parts': [prompt]}]
        messages = [prompt]
        response = self.client.models.generate_content(
            model=self.model_string,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt_arg,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                candidate_count=1,
            )
        )
        response_text = response.text

        if self.use_cache:
            self._save_cache(sys_prompt_arg + prompt, response_text)
        return response_text

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                image_obj = Image.open(io.BytesIO(item))
                formatted_content.append(image_obj)
            elif isinstance(item, str):
                formatted_content.append(item)
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, **kwargs
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        response = self.client.models.generate_content(
            model=self.model_string,
            contents=formatted_content,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt_arg,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                candidate_count=1
            )
        )
        response_text = response.text

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text