import os
import json
import requests
from typing import Any, Dict, List
from functools import lru_cache
from typing import Any, Mapping, Sequence
from llama_index.llms.vllm import VllmServer
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import CompletionResponse
from openai import OpenAI
from loguru import logger

DEFAULT_BASE_URL = "http://qwen.morris-chang.rocks/v1/completions"
DEFAULT_API_KEY = "EMPTY"


def custom_post_http_request(
    api_url: str, sampling_params: dict = {}, stream: bool = False
) -> requests.Response:
    # headers = {"User-Agent": "Test Client"}
    headers = {"Content-Type": "application/json"}
    sampling_params["stream"] = stream
    return requests.post(api_url, headers=headers, json=sampling_params, stream=True)


def custom_get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    return data["choices"][0]["text"]


class CustomVllmServer(VllmServer):
    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "best_of": self.best_of,
            "ignore_eos": self.ignore_eos,
            "stop": self.stop,
            "logprobs": self.logprobs,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        return {**base_kwargs}

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        # build sampling parameters
        sampling_params = dict(**params)
        sampling_params["prompt"] = prompt
        response = custom_post_http_request(self.api_url, sampling_params, stream=False)
        output = custom_get_response(response)

        return CompletionResponse(text=output)


@lru_cache(maxsize=None)
def _get_client(base_url: str, api_key: str, **kwargs: Any) -> OpenAI:
    """Create and reuse an OpenAI-compatible client for the vLLM endpoint."""
    return OpenAI(base_url=base_url, api_key=api_key, **kwargs)


def _get_vllm_client(api_url: str) -> VllmServer:

    llm = CustomVllmServer(api_url=api_url)
    return llm


def request_llm_vllm_completion(
    prompt: str,
    model: str,
    *,
    api_url: str | None = None,
    streamming: bool = False,
    enable_thinking: bool = False,
    **kwargs: Any,
):
    """Send a completion request and return the response text."""
    client = _get_vllm_client(
        api_url or os.getenv("VLLM_BASE_URL", DEFAULT_BASE_URL),
    )
    if not enable_thinking:
        prompt += " /no_think"

    if streamming:
        return [x for x in client.stream_completion(prompt)][-1]
    else:
        return client.complete(prompt, **kwargs)


def request_llm_vllm_chat(
    messages: Sequence[Mapping[str, str]],
    model: str = "Qwen/Qwen3-14B",
    *,
    api_url: str | None = None,
    streamming: bool = False,
    enable_thinking: bool = False,
    retry: int = 10,
    **kwargs: Any,
) -> str:
    """Send a chat completion request and return the response text."""
    if not messages:
        raise ValueError("messages must contain at least one item.")

    client = _get_vllm_client(
        api_url or os.getenv("VLLM_BASE_URL", DEFAULT_BASE_URL),
    )
    chat_history = []
    for message in messages:
        chat_history.append(
            ChatMessage(role=message["role"], content=message["content"])
        )
    if not enable_thinking:
        chat_history[-1].content += " /no_think"

    if streamming:
        return [x for x in client.stream_chat(chat_history)][-1]
    else:
        try:
            return client.chat(chat_history, **kwargs)
        except Exception as e:
            logger.error(f"Error while requesting chat completion: {e}")
            if retry > 0:
                return request_llm_vllm_chat(
                    messages,
                    model,
                    api_url=api_url,
                    streamming=streamming,
                    enable_thinking=enable_thinking,
                    retry=retry - 1,
                    **kwargs,
                )
            else:
                return "Error: Failed to request chat completion after retrying."
