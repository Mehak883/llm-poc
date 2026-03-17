from Configs.base_config import BaseConfig
from pydantic import Field

class DevConfig(BaseConfig):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    AZURE_OPENAI_API_KEY:str =""
    AZURE_OPENAI_ENDPOINT:str =""
    AZURE_OPENAI_DEPLOYMENT_NAME:str ="gpt-4o-mini"
    AZURE_OPENAI_API_VERSION:str ="2025-01-01-preview"