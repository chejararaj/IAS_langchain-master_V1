from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json
from langchain.embeddings.base import Embeddings

from app.langchain.v1.config import (
    CLIENT_ID,
    CLIENT_SECRET,
    PINGFEDERATE_URL,
    IAS_OPENAI_CHAT_URL,
    IAS_OPENAI_URL,
    IAS_EMBEDDINGS_URL,
)



def federate_auth() -> str:
    """Obtains auth access token for accessing GPT endpoints"""
    try:
        payload = f"client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(PINGFEDERATE_URL, headers=headers, data=payload)
        token = response.json()["access_token"]
        return token
    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e.response.status_code, e)
    except requests.exceptions.ConnectionError as e:
        print("Connection Error:", e.response.status_code, e)
    except requests.exceptions.Timeout as e:
        print("Timeout Error:", e.response.status_code, e)
    except requests.exceptions.RequestException as e:
        print("Other Error:", e.response.status_code, e)


def ias_openai_chat_completion(
    token: str, user_message: str, engine: str, temperature: str, max_tokens: int
) -> str:
    """
    Generates a chat completion response for OpenAI model
    :param token: auth token
    :param user_message: user's prompt
    :param engine: model capable for chat completion i.e. gpt*
    :param temperature: value 0-1 that tells model to be more precise or generative
    :param max_tokens: max tokens the prompt & response should be. It depends on the model's capacity
    :return: response from OpenAI model
    """
    try:
        payload = json.dumps(
            {
                "engine": engine,
                "messages": [
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        response = requests.post(IAS_OPENAI_CHAT_URL, headers=headers, data=payload)
        response.raise_for_status()
        chat_completion = json.loads(response.json()["result"])["content"]
        return chat_completion
    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e.response.status_code, e)
    except requests.exceptions.ConnectionError as e:
        print("Connection Error:", e.response.status_code, e)
    except requests.exceptions.Timeout as e:
        print("Timeout Error:", e.response.status_code, e)
    except requests.exceptions.RequestException as e:
        print("Other Error:", e.response.status_code, e)


def ias_openai_completion(
    token: str, user_message: str, engine: str, temperature: str, max_tokens: int
) -> str:
    """
    Generates a completion response for OpenAI model
    :param token: auth token
    :param user_message: user's prompt
    :param engine: model capable for completion
    :param temperature: value 0-1 that tells model to be more precise or generative
    :param max_tokens: max tokens the prompt & response should be. It depends on the model's capacity
    :return: response from OpenAI model
    """
    try:
        payload = json.dumps(
            {
                "engine": engine,
                "prompt": user_message,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        response = requests.post(IAS_OPENAI_URL, headers=headers, data=payload)
        completion_resp = response.json()
        completion = completion_resp["result"]
        return completion
    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e.response.status_code, e)
    except requests.exceptions.ConnectionError as e:
        print("Connection Error:", e.response.status_code, e)
    except requests.exceptions.Timeout as e:
        print("Timeout Error:", e.response.status_code, e)
    except requests.exceptions.RequestException as e:
        print("Other Error:", e.response.status_code, e)


def ias_openai_embeddings(token: str, raw_text, engine: str):
    try:
        url = IAS_EMBEDDINGS_URL

        payload = json.dumps({"input": raw_text, "engine": engine})
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

        response = requests.post(url, headers=headers, data=payload)
        embeddings = json.loads(response.json()["result"])
        return embeddings
    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e.response.status_code, e)
    except requests.exceptions.ConnectionError as e:
        print("Connection Error:", e.response.status_code, e)
    except requests.exceptions.Timeout as e:
        print("Timeout Error:", e.response.status_code, e)
    except requests.exceptions.RequestException as e:
        print("Other Error:", e.response.status_code, e)


class IASOpenaiLLM(LLM):
    """Wrapper for IAS secured OpenAI completion API"""

    engine: str
    temperature: str
    max_tokens: int

    @property
    def _llm_type(self) -> str:
        return "IAS_OpenAI"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        token = federate_auth()
        response = ias_openai_completion(
            token, prompt, self.engine, self.temperature, self.max_tokens
        )
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        params = {
            "engine": self.engine,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return params


class IASOpenaiConversationalLLM(LLM):
    """Wrapper for IAS secured OpenAI chat API"""

    engine: str
    temperature: str
    max_tokens: int

    @property
    def _llm_type(self) -> str:
        return "IAS_OpenAI"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        token = federate_auth()
        response = ias_openai_chat_completion(
            token, prompt, self.engine, self.temperature, self.max_tokens
        )
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        params = {
            "engine": self.engine,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return params


class IASOpenaiEmbeddings(Embeddings):
    """Wrapper for IAS secured OpenAI embedding API"""

    engine = str

    def __init__(self, engine):
        self.engine = engine

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeddings search docs."""
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.

        token = federate_auth()
        response = ias_openai_embeddings(token, texts, self.engine)

        # Extract the embeddings
        embeddings: list[list[float]] = [data["embedding"] for data in response]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeddings  query text."""
        token = federate_auth()
        response = ias_openai_embeddings(token, text, self.engine)

        # Extract the embeddings and total tokens
        embeddings: list[list[float]] = [data["embedding"] for data in response]
        return embeddings[0]
