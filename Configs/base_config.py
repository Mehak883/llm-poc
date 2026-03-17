from pydantic import Field
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str = Field(default="")
    AZURE_OPENAI_API_KEY: str = Field(default="")
    AZURE_OPENAI_ENDPOINT: str = Field(default="")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = Field(default="")
    AZURE_OPENAI_API_VERSION: str = Field(default="")