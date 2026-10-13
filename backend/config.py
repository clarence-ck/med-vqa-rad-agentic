import logging
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

# ---------------------------------------------------------------------------
# Per-role model configuration.
#
# Each role reads MODEL_<ROLE> and REASONING_EFFORT_<ROLE> from the environment.
# Roles that need image input (look, answer, baseline) must use a vision-capable
# model. The baseline should match the `answer` model so the comparison measures
# architecture rather than model choice.
# ---------------------------------------------------------------------------
_ROLE_DEFAULTS = {
    #  role        model            effort    used by
    # Budget profile: Luna everywhere except the verifier and judge, which use a
    # different model (Sol) so checking is not done by the model that answered.
    "controller": ("gpt-6-luna",   "none"),   # Haystack Agent controller (chat-completions tool calls reject other efforts)
    "look":       ("gpt-6-luna",   "medium"), # ImageInterpreter
    "analyze":    ("gpt-6-luna",   "low"),    # QuestionAnalyzer, DifficultyRouter
    "answer":     ("gpt-6-luna",   "medium"), # ReasoningAgent
    "check":      ("gpt-6.1-sol",  "low"),    # VerificationAgent
    "baseline":   ("gpt-6-luna",   "medium"), # single-shot baseline
    "judge":      ("gpt-6.1-sol",  "low"),    # evaluation judge for borderline answers
}

# Reasoning models count hidden reasoning tokens toward this limit, so keep it generous.
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "8000"))

MODELS = {
    role: os.getenv(f"MODEL_{role.upper()}", model)
    for role, (model, _) in _ROLE_DEFAULTS.items()
}
REASONING_EFFORTS = {
    role: os.getenv(f"REASONING_EFFORT_{role.upper()}", effort).strip().lower()
    for role, (_, effort) in _ROLE_DEFAULTS.items()
}

for _legacy in ("VISION_MODEL", "OPENAI_MODEL"):
    if os.getenv(_legacy):
        _cfg_log.warning("%s is no longer used; set MODEL_<ROLE> variables instead (see .env.example)", _legacy)


def generation_kwargs(role: str, max_completion_tokens: int | None = None) -> dict:
    """Chat-completion kwargs for a role: token limit plus reasoning effort ('' to omit)."""
    kwargs = {"max_completion_tokens": max_completion_tokens or MAX_COMPLETION_TOKENS}
    effort = REASONING_EFFORTS.get(role, "")
    if effort:
        kwargs["reasoning_effort"] = effort
    return kwargs


def model_manifest() -> dict:
    """Models and reasoning efforts in effect, stored with cached results for traceability."""
    return {
        role: {"model": MODELS[role], "reasoning_effort": REASONING_EFFORTS[role] or None}
        for role in _ROLE_DEFAULTS
    }
