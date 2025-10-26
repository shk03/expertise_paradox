"""Batch generation workflow across different LLM providers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from llm_pipeline.batch import run_batch_inference
from llm_pipeline.providers import get_provider, list_providers

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import pandas as pd

logger = logging.getLogger(__name__)


class GenerationConfigError(ValueError):
    """Raised when the provided generation configuration is invalid."""


def run_generation(
    provider_name: str,
    input_path: str,
    output_path: str,
    temp_output_path: str,
    case_column: str = "case_description",
    client_kwargs: Optional[Dict[str, object]] = None,
) -> "pd.DataFrame":
    """
    Execute the neuroradiology generation workflow for a specific provider.

    Parameters
    ----------
    provider_name:
        The LLM provider identifier (``openai``, ``deepseek``, or ``gemini``).
    input_path:
        Path to the CSV containing at least the case description column.
    output_path:
        Path where the merged input+generation results will be written.
    temp_output_path:
        Path used to persist incremental results for resumability.
    case_column:
        Name of the column that holds the free-text case description.
    client_kwargs:
        Optional keyword arguments forwarded to the provider's ``get_client``.

    Returns
    -------
    pandas.DataFrame
        The merged DataFrame ready for downstream analysis.
    """
    provider_name = provider_name.lower()
    available = list(list_providers())
    if provider_name not in available:
        raise GenerationConfigError(
            f"Unknown provider '{provider_name}'. Available: {sorted(available)}"
        )

    provider = get_provider(provider_name)
    logger.info("Loading input data from %s", input_path)
    input_df = _load_input_dataframe(input_path)

    if case_column not in input_df.columns:
        raise GenerationConfigError(
            f"Column '{case_column}' not found in input data. "
            f"Available columns: {list(input_df.columns)}"
        )

    Path(temp_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising %s client", provider_name)
    client_factory = provider.client
    client = client_factory(**(client_kwargs or {}))

    logger.info("Starting batch inference with %s", provider_name)
    results_df = run_batch_inference(
        input_df=input_df,
        case_column=case_column,
        output_csv_path=temp_output_path,
        generation_function=provider.generator,
        api_client=client,
    )

    rename_map = provider.rename_map or {}
    if rename_map:
        results_df = results_df.rename(columns=rename_map)

    merged_df = input_df.merge(
        results_df, left_index=True, right_index=True, how="left"
    )
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(
        "Completed generation with %s. Saved merged results to %s",
        provider_name,
        output_path,
    )
    return merged_df


def _load_input_dataframe(path: str) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(path)
    if "original_index" in df.columns:
        df = df.set_index("original_index")
    else:
        df = df.reset_index().rename(columns={"index": "original_index"})
        df = df.set_index("original_index")
    return df
