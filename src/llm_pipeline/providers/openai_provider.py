# OpenAI provider utilities.
import os
from typing import Optional

from openai import OpenAI

from llm_pipeline.schemas import OpenAIDiagnosis

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")


def get_client() -> OpenAI:
    """Initializes and returns the standard OpenAI client."""
    # The client automatically picks up the OPENAI_API_KEY from the .env file
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return client
    except Exception as e:
        raise ValueError(f"Failed to initialize OpenAI client: {e}") from e


def generate_diagnosis(
    case_description: str, client: OpenAI, model: Optional[str] = None
) -> OpenAIDiagnosis:
    """
    Generates a structured neuroradiology diagnosis using the OpenAI API.
    """
    model_name = model or DEFAULT_MODEL
    prompt = f"""You are an experienced neuroradiologist.

    Below is a case presentation including patient demographics, relevant clinical history, and brain MRI findings. Based on this information, provide your top three differential diagnoses ranked in order of likelihood.

    CASE: {case_description}

    For each differential diagnosis, briefly explain your reasoning.
    """
    try:
        response = client.beta.chat.completions.parse(  # type: ignore[attr-defined]
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You produce structured JSON output only with the following keys: first_diagnosis, second_diagnosis, third_diagnosis, and rationale.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format=OpenAIDiagnosis,
        )
        return response.choices[0].message.parsed
    except Exception as e:
        # Re-raise the exception to be caught by the batch processor
        raise RuntimeError(f"OpenAI API call failed: {e}") from e
