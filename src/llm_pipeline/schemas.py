# src/schemas.py

from pydantic import BaseModel, Field


class OpenAIDiagnosis(BaseModel):
    """Pydantic model for OpenAI structured output."""

    first_diagnosis: str = Field(..., description="The most likely diagnosis")
    second_diagnosis: str = Field(..., description="The second most likely diagnosis")
    third_diagnosis: str = Field(..., description="The third most likely diagnosis")
    rationale: str = Field(..., description="Detailed explanation for the diagnoses")


class DeepSeekDiagnosis(BaseModel):
    """Pydantic model for DeepSeek (via Fireworks) structured output."""

    r1_first_diagnosis: str = Field(..., description="The most likely diagnosis")
    r1_second_diagnosis: str = Field(
        ..., description="The second most likely diagnosis"
    )
    r1_third_diagnosis: str = Field(..., description="The third most likely diagnosis")
    r1_rationale: str = Field(..., description="Detailed explanation for the diagnoses")


class GeminiDiagnosis(BaseModel):
    """Pydantic model for Google Gemini structured output."""

    gemini_first_diagnosis: str = Field(..., description="The most likely diagnosis")
    gemini_second_diagnosis: str = Field(
        ..., description="The second most likely diagnosis"
    )
    gemini_third_diagnosis: str = Field(
        ..., description="The third most likely diagnosis"
    )
    gemini_rationale: str = Field(
        ..., description="Detailed explanation for the diagnoses"
    )

