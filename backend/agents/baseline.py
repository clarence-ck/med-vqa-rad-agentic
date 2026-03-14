import base64
import json
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def baseline_answer(image_bytes: bytes, question: str,
                    mime_type: str = "image/png") -> dict:
    """
    Baseline: Single-shot answer
    Sends image + question in one prompt to get a direct answer.
    Used for comparison against the agentic pipeline.
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""You are a medical visual question answering system. Look at this medical image and answer the following question.

Question: "{question}"

Return a JSON object with:
{{
  "answer": "Direct answer to the question",
  "explanation": "Brief explanation (1-2 sentences)",
  "confidence": "high / medium / low"
}}

Return ONLY the JSON object, no additional text."""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=3000,

        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        result = json.loads(raw)
        result["status"] = "success"
        return result

    except json.JSONDecodeError:
        return {
            "status": "error",
            "answer": "Baseline failed to produce an answer.",
            "explanation": "",
            "confidence": "low"
        }
    except Exception as e:
        return {
            "status": "error",
            "answer": "Baseline error.",
            "explanation": str(e),
            "confidence": "low"
        }
