from Configs.base_config import BaseConfig
from pydantic import Field

class QaConfig(BaseConfig):
    OPENAI_API_KEY: str = Field(default="")
    AZURE_OPENAI_API_KEY: str = Field(default="")
    AZURE_OPENAI_ENDPOINT: str = Field(default="")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = Field(default="")
    AZURE_OPENAI_API_VERSION: str = Field(default="")
    DEBUG: bool = False
    LOG_LEVEL: str = "DEBUG"