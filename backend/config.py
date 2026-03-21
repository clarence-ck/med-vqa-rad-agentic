import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-120b:fastest")
TEXT_MODEL_BASE_URL = os.getenv("TEXT_MODEL_BASE_URL", "https://router.huggingface.co/v1")
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

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Set it in .env file.")

if not OPENAI_MODEL:
    print("WARNING: OPENAI_MODEL not set. Defaulting to 'gpt-5.4-mini-2026-03-17'.")
    OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. Text agents (gpt-oss-120b) will fail. Set it in .env file.")
