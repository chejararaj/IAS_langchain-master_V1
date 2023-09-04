import logging
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: int = logging.WARNING


settings = Settings()

# Environment variables
AWS_OPENSEARCH_USERNAME = os.environ.get("AWS_OPENSEARCH_USERNAME", "")
AWS_OPENSEARCH_PASSWORD = os.environ.get("AWS_OPENSEARCH_PASSWORD", "")
AWS_OPENSEARCH_HOST = os.environ.get("AWS_OPENSEARCH_HOST", "")
IAS_OPENAI_URL = os.getenv("IAS_OPENAI_URL", "")
IAS_OPENAI_CHAT_URL = os.getenv("IAS_OPENAI_CHAT_URL", "")
PINGFEDERATE_URL = os.getenv("PINGFEDERATE_URL", "")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
IAS_EMBEDDINGS_URL = os.getenv("IAS_EMBEDDINGS_URL", "")
