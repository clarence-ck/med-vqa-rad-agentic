import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
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
