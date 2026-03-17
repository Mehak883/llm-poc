import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()


def get_config():
    env = os.getenv("ENV_TYPE","DEV")
    if env is None:
        logger.exception("ENV_TYPE environment variable is not set or found.")
        raise ValueError("ENV_TYPE environment variable is not set or found.")
    env = env.lower()
    logger.info(f"Environment type of SaraAI: {env}")

    if env == "prod":
        from Configs.prod_config import ProdConfig
        return ProdConfig()
    elif env == "qa":
        from Configs.qa_config import QaConfig
        return QaConfig()
    else:
        from Configs.dev_config import DevConfig
        return DevConfig()

config = get_config()
