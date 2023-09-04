from enum import Enum
from typing import List, Optional, Union, Dict

from pydantic import BaseModel, Field


class Status(str, Enum):
    failure = "failure"
    success = "success"


class HealthCheckResponse(BaseModel):
    status: Status
    message: str


class DocumentEmbeddingsRequest(BaseModel):
    index_name: str
    engine: str


class DocumentEmbeddingsResponse(BaseModel):
    status: Status
    index_name: str
    embeddings_ids: List[str]


class QueryEmbeddingsRequest(BaseModel):
    query: str
    index_name: str
    temperature: int
    max_tokens: float
    embeddings_engine: str
    completion_engine: str


class QueryEmbeddingsResponse(BaseModel):
    status: Status
    answer: str
    citation: List
