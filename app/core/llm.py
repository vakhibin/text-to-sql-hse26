from pyexpat.errors import messages
from typing import List, Optional

from openrouter import OpenRouter
from ollama import Client

class LLMClientError(Exception):
    pass

class OpenRouterLLMClient:
    def __init__(self, api_key, model: Optional[str] = None):
        self._client = OpenRouter(api_key=api_key)
        self._model = model

    def call_llm(self, messages: List[dict]):
        if not self._client:
            raise LLMClientError("LLM client is empty or invalid")
        if not self._model:
            raise LLMClientError("You should set some model first!")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages
            )
            content = response.choices[0].message.content
        except Exception as e:
            raise LLMClientError(f"Error occured while calling LLM: {e}")

        return content

    def set_model(self, model: str):
        if not isinstance(model, str):
            raise LLMClientError("Invalid model name!")
        self._model = model


class OllamaLLMClient:
    def __init__(self,  model: Optional[str] = None):
        self._client = Client(host='http://localhost:11434')
        self._model = model

    def call_llm(self, messages: List[dict]):
        if not self._client:
            raise LLMClientError("LLM client is empty or invalid")
        if not self._model:
            raise LLMClientError("You should set some model first!")
        try:
            response = self._client.chat(model=self._model, messages=messages)
        except Exception as e:
            raise LLMClientError(f"Error occured while calling LLM: {e}")

        return response

    def set_model(self, model: str):
        if not isinstance(model, str):
            raise LLMClientError("Invalid model name!")
        self._model = model

from app.config.config import LLMConfig as llm_config

if __name__ == "__main__":
    c = OllamaLLMClient()
    c.set_model(llm_config.model)
    messages = [
        {"role": "system", "content": "Just have some funny conversation with me"},
        {"role": "user", "content": "Hello! How are you today?"}
    ]
    res = c.call_llm(messages)
    print(res)