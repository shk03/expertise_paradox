# DeepSeek R1 provider utilities.
import os
import re
from typing import Optional

from openai import OpenAI

from llm_pipeline.schemas import DeepSeekDiagnosis

DEFAULT_MODEL = os.getenv(
    "DEEPSEEK_MODEL", "accounts/fireworks/models/deepseek-r1-0528"
)
DEFAULT_BASE_URL = os.getenv(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)


def get_client():
    return OpenAI(api_key=os.getenv("FIREWORKS_API_KEY"), base_url=DEFAULT_BASE_URL)


def generate_diagnosis(
    case_description: str, client: OpenAI, model: Optional[str] = None
) -> DeepSeekDiagnosis:
    prompt = f"""You are an experienced neuroradiologist.

Below is a case presentation. Based on this information, provide your top three differential diagnoses ranked in order of likelihood.

CASE:
{case_description}

First, show your chain of thought enclosed in <think>…</think>.  
Then output exactly one JSON object matching this schema:
{DeepSeekDiagnosis.model_json_schema()}

Do NOT include any extra keys or markdown."""

    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Think first, then output JSON matching the schema.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=6000,
    )
    raw = resp.choices[0].message.content

    json_match = re.search(r"</think>\s*(\{.*\})", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Could not find valid JSON after <think> tag in response.")

    return DeepSeekDiagnosis.parse_raw(json_match.group(1))
