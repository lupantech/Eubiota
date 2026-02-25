# Reference: https://help.aliyun.com/zh/dashscope/developer-reference/api-details

try:
    import dashscope
except ImportError:
    raise ImportError("If you'd like to use DashScope models, please install the dashscope package by running `pip install dashscope`, and add 'DASHSCOPE_API_KEY' to your environment variables.")

import os
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

class DefaultFormat(BaseModel):
    response: str


def validate_chat_model(model_string: str):
    return any(x in model_string for x in ["qwne", "qwen", "llama", "baichuan"])


def validate_structured_output_model(model_string: str):
    Structure_Output_Models = ["qwen-max", "qwen-plus", "llama3-70b-instruct"]
    return any(x in model_string for x in Structure_Output_Models)


class ChatDashScope(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="qwen2.5-7b-instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        use_cache: bool=True,
        **kwargs):

        self.model_string = model_string

        if model_string.startswith("dashscope-") and len(model_string) > len("dashscope-"):
            self.model_string = model_string[len("dashscope-"):]
        elif model_string == "dashscope":
            self.model_string = "qwen2.5-7b-instruct"
        else:
            raise ValueError(f"Undefined model name: '{model_string}'. Only model strings with prefix 'dashscope-' are supported.")
        
        print(f"Dashscope llm engine initialized with {self.model_string}")
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        self.support_structured_output = validate_structured_output_model(self.model_string)
        self.is_chat_model = validate_chat_model(self.model_string)

        if self.use_cache:
            root = platformdirs.user_cache_dir("scientist")
            cache_path = os.path.join(root, f"cache_dashscope_{self.model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)
        
        if os.getenv("DASHSCOPE_API_KEY") is None:
            raise ValueError("Please set the DASHSCOPE_API_KEY environment variable if you'd like to use DashScope models.")
        
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    @retry(wait=wait_random_exponential(min=1, max=7), stop=stop_after_attempt(7))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            
            elif isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    full_text = "\n".join(content)
                    return self._generate_text(full_text, system_prompt=system_prompt, **kwargs)

                elif any(isinstance(item, bytes) for item in content):
                    if not self.is_multimodal:
                        raise NotImplementedError(
                            f"Multimodal generation is only supported for {self.model_string}. "
                            "Consider using a multimodal model like 'gpt-4o'."
                        )
                    return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)

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
        
    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2048, top_p=0.99, response_format=None
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
            
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": prompt}
        ]

        request_params = {
            "model": self.model_string,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "result_format": "message"
        }
        
        if 'qwen3' in self.model_string.lower() or '8b' in self.model_string.lower() or '14b' in self.model_string.lower():
            request_params['enable_thinking'] = False

        response = dashscope.Generation.call(**request_params)

        if response.status_code == 200:
            if hasattr(response, 'output') and response.output is not None:
                if hasattr(response.output, 'choices') and response.output.choices:
                    if isinstance(response.output.choices[0], dict) and 'message' in response.output.choices[0]:
                        if 'content' in response.output.choices[0]['message']:
                            response_text = response.output.choices[0]['message']['content']
                        else:
                            raise Exception(f"Unexpected response structure: Missing 'content' field")
                    elif hasattr(response.output.choices[0], 'message') and hasattr(response.output.choices[0].message, 'content'):
                        response_text = response.output.choices[0].message.content
                    else:
                        raise Exception(f"Unexpected response structure: Missing 'message' field")
                else:
                    raise Exception(f"Unexpected response structure: 'choices' is empty or missing")
            else:
                raise Exception(f"Unexpected response structure: 'output' is None or missing")
        else:
            raise Exception(f"DashScope API error: {response.message}")

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                continue
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_multimodal(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=512, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": formatted_content}
        ]

        request_params = {
            "model": self.model_string,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "result_format": "message"
        }

        response = dashscope.Generation.call(**request_params)

        if response.status_code == 200:
            if hasattr(response, 'output') and response.output is not None:
                if hasattr(response.output, 'choices') and response.output.choices:
                    if isinstance(response.output.choices[0], dict) and 'message' in response.output.choices[0]:
                        if 'content' in response.output.choices[0]['message']:
                            response_text = response.output.choices[0]['message']['content']
                        else:
                            raise Exception(f"Unexpected response structure: Missing 'content' field")
                    elif hasattr(response.output.choices[0], 'message') and hasattr(response.output.choices[0].message, 'content'):
                        response_text = response.output.choices[0].message.content
                    else:
                        raise Exception(f"Unexpected response structure: Missing 'message' field")
                else:
                    raise Exception(f"Unexpected response structure: 'choices' is empty or missing")
            else:
                raise Exception(f"Unexpected response structure: 'output' is None or missing")
        else:
            raise Exception(f"DashScope API error: {response.message}")

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text