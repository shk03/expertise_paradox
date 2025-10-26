# Batch tooling shared across LLM batch generation commands.
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel
from tqdm import tqdm

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import pandas as pd

logger = logging.getLogger(__name__)


def run_batch_inference(
    input_df: "pd.DataFrame",
    case_column: str,
    output_csv_path: str,
    generation_function: Callable[[str, Any], BaseModel],
    api_client: Any,
    max_retries: int = 5,
) -> "pd.DataFrame":
    """
    Processes a DataFrame of cases using a given generation function.
    This is resumable and saves intermediate results to disk.
    """
    import pandas as pd

    index_name = input_df.index.name or "index"

    if os.path.exists(output_csv_path):
        processed_df = _read_results(output_csv_path, index_name=index_name)
        processed_ids = set(processed_df.index)
        logger.info(
            f"Loaded {len(processed_ids)} existing results from {output_csv_path}."
        )
    else:
        processed_ids = set()

    indices_to_process = sorted(list(set(input_df.index) - processed_ids))

    if not indices_to_process:
        logger.info("No new cases to process.")
        return _read_results(output_csv_path, index_name=index_name)

    success_count, fail_count = 0, 0
    for idx in tqdm(indices_to_process, desc="Processing cases", unit="case"):
        for attempt in range(max_retries):
            try:
                case_text = input_df.at[idx, case_column]
                diagnosis = generation_function(case_text, api_client)

                payload = diagnosis.model_dump()
                payload[index_name] = idx

                temp_df = pd.DataFrame([payload]).set_index(index_name)
                write_header = not os.path.exists(output_csv_path)
                temp_df.to_csv(output_csv_path, mode="a", header=write_header)

                success_count += 1
                break
            except Exception as e:
                logger.warning(
                    f"Error on index {idx} (attempt {attempt+1}/{max_retries}): {e}"
                )
                time.sleep(2**attempt)
        else:
            fail_count += 1
            logger.error(f"Failed to process index {idx} after {max_retries} attempts.")

    logger.info(
        "Batch complete. New successes: %s, Failures: %s",
        success_count,
        fail_count,
    )
    return _read_results(output_csv_path, index_name=index_name)


def _read_results(path: str, index_name: str) -> "pd.DataFrame":
    """Helper to load previously persisted generation results."""
    import pandas as pd

    df = pd.read_csv(path)
    if index_name not in df.columns:
        raise ValueError(
            f"Expected column '{index_name}' in {path}, found columns: {df.columns}"
        )
    return df.set_index(index_name)
