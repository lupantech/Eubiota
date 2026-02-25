try:
    from ollama import Client
except ImportError:
    raise ImportError(
        "If you'd like to use Ollama, please install the ollama package by running `pip install ollama`, and set appropriate API keys for the models you want to use."
    )

import json
import os
from typing import List, Union

import platformdirs
from ollama import Image, Message

from .base import CachedEngine, EngineLM


class ChatOllama(EngineLM, CachedEngine):
    """
    Ollama implementation of the EngineLM interface.
    This allows using any model supported by Ollama.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="qwen2.5vl:3b",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        :param model_string:
        :param system_prompt:
        :param is_multimodal:
        """

        self.model_string = (
            model_string if ":" in model_string else f"{model_string}:latest"
        )
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("scientist")
            cache_path = os.path.join(root, f"cache_ollama_{self.model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)

        try:
            self.client = Client(
                host="http://localhost:11434",
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to Ollama server: {e}")

        models = self.client.list().models
        if len(models) == 0:
            raise ValueError(
                "No models found in the Ollama server. Please ensure the server is running and has models available."
            )
        if self.model_string not in [model.model for model in models]:
            print(
                f"Model '{self.model_string}' not found. Attempting to pull it from the Ollama registry."
            )
            try:
                self.client.pull(self.model_string)
            except Exception as e:
                raise ValueError(f"Failed to pull model '{self.model_string}': {e}")

    def generate(
        self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs
    ):
        if isinstance(content, str):
            return self._generate_text(content, system_prompt=system_prompt, **kwargs)

        elif isinstance(content, list):
            if not self.is_multimodal:
                raise NotImplementedError(
                    f"Multimodal generation is only supported for {self.model_string}."
                )

            return self._generate_multimodal(
                content, system_prompt=system_prompt, **kwargs
            )

    def _generate_text(
        self,
        prompt,
        system_prompt=None,
        temperature=0,
        max_tokens=4000,
        top_p=0.99,
        response_format=None,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        # Chat models without structured outputs
        response = self.client.chat(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            format=response_format.model_json_schema() if response_format else None,
            options={
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
        )
        response = response.message.content

        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> Message:
        """
        Formats the input content into a Message object for Ollama.
        """
        text_parts = []
        images = []
        for item in content:
            if isinstance(item, bytes):
                images.append(Image(item))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return Message(
            role="user",
            content="\n".join(text_parts) if text_parts else None,
            images=images if images else None,
        )

    def _generate_multimodal(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature=0,
        max_tokens=4000,
        top_p=0.99,
        response_format=None,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        message = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(message)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        response = self.client.chat(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {
                    "role": message.role,
                    "content": message.content,
                    "images": message.images if message.images else None,
                },
            ],
            format=response_format.model_json_schema() if response_format else None,
            options={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
        )
        response_text = response.message.content

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text
