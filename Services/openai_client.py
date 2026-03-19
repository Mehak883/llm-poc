import os
from openai import AzureOpenAI
from Configs import config

class AzureOpenAIClient:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            api_key = config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not set in environment variables")
            cls._client = AzureOpenAI(api_key=config.AZURE_OPENAI_API_KEY, 
                                      azure_endpoint=config.AZURE_OPENAI_ENDPOINT, 
                                      api_version=config.AZURE_OPENAI_API_VERSION)
        return cls._client