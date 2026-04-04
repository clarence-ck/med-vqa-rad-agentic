import logging
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vqa-rad')
DATA_DIR = DEFAULT_DATA_DIR
BACKEND_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "BACKEND_CORS_ORIGINS",
        "",
    ).split(",")
    if origin.strip()
]

_cfg_log = logging.getLogger("medvqa.config")

if not OPENAI_API_KEY:
    _cfg_log.warning("OPENAI_API_KEY not set — set it in the .env file")

if not OPENAI_MODEL:
    _cfg_log.warning("OPENAI_MODEL not set — defaulting to 'gpt-5.4-mini-2026-03-17'")
    OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"

if not VISION_MODEL:
    _cfg_log.warning("VISION_MODEL not set — defaulting to 'gpt-5.4-mini-2026-03-17'")
    VISION_MODEL = "gpt-5.4-mini-2026-03-17"
