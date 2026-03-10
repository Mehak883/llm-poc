import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment variables")
            cls._client = OpenAI(api_key=api_key)
        return cls._client