"""
Shared utility for stripping markdown code fences from LLM JSON responses.
"""
import re


def strip_json_fences(raw: str) -> str:
    """
    Remove markdown code fences that LLMs sometimes wrap around JSON.

    Handles:
        ```json
        { ... }
        ```

        ```
        { ... }
        ```

        { ... }   (already clean)
    """
    raw = raw.strip()
    # Remove opening fence with optional language tag (case-insensitive)
    raw = re.sub(r"^```[a-zA-Z]*\s*\n?", "", raw)
    # Remove closing fence
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()
