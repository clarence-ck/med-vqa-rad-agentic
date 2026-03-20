import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
LLAVA_API_BASE_URL = os.getenv(
    "LLAVA_API_BASE_URL",
    "https://api-inference.huggingface.co/models/microsoft/llava-med-v1.5-mistral-7b",
)
LLAVA_API_KEY = os.getenv("LLAVA_API_KEY", os.getenv("HF_TOKEN", ""))
LLAVA_MODEL_ID = os.getenv("LLAVA_MODEL_ID", "microsoft/llava-med-v1.5-mistral-7b")
LLAVA_TIMEOUT_SECONDS = float(os.getenv("LLAVA_TIMEOUT_SECONDS", "120"))
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
    print("WARNING: OPENAI_MODEL not set. Defaulting to 'gpt-5-mini-2025-08-07'.")
    OPENAI_MODEL = "gpt-5-mini-2025-08-07"

if not LLAVA_API_BASE_URL:
    print("WARNING: LLAVA_API_BASE_URL not set. Defaulting to the Hugging Face Llava endpoint.")
