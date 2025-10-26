# Gemini provider utilities.
import json
import os
import re
from typing import Optional

import google.generativeai as genai

from llm_pipeline.schemas import GeminiDiagnosis

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-exp")


def get_client(model_name: Optional[str] = None):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name or DEFAULT_MODEL)


def generate_diagnosis(
    case_description: str, client: genai.GenerativeModel
) -> GeminiDiagnosis:
    prompt = f"""You are an expert neuroradiologist.

Below is a case presentation including patient demographics, relevant clinical history, and brain MRI findings. Based on this information, provide your top three differential diagnoses ranked in order of likelihood.

CASE:
{case_description}

For each differential diagnosis, briefly explain your reasoning.

Output your response ONLY as a single JSON object matching this schema. Do not include any extra text or markdown formatting like ```json ... ```.

JSON Schema:
{GeminiDiagnosis.model_json_schema()}
"""

    config = genai.types.GenerationConfig(temperature=0, max_output_tokens=6000)
    response = client.generate_content(prompt, generation_config=config)

    raw_text = response.text.strip()
    try:
        # Handle both raw JSON and JSON in a markdown block
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = raw_text

        return GeminiDiagnosis.parse_raw(json_str)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from Gemini response.")
