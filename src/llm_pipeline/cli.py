"""Command-line interface for running batch LLM generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from llm_pipeline.providers import list_providers


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="llm-pipeline",
        description="Generate differential diagnoses with supported LLM providers.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Run batch generation.")
    provider_choices = sorted(list(list_providers()))
    gen_parser.add_argument(
        "--provider",
        required=True,
        choices=provider_choices,
        help="LLM provider to use.",
    )
    gen_parser.add_argument(
        "--input", required=True, help="Path to input CSV containing case text."
    )
    gen_parser.add_argument(
        "--output",
        required=True,
        help="Path to write merged CSV with generation outputs.",
    )
    gen_parser.add_argument(
        "--temp-output",
        required=True,
        help="Path used for resumable intermediate results.",
    )
    gen_parser.add_argument(
        "--column",
        default="case_description",
        help="Column containing the case description text.",
    )

    args = parser.parse_args(argv)

    load_dotenv()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("llm_pipeline.cli")

    if args.command == "generate":
        from llm_pipeline import run_generation

        run_generation(
            provider_name=args.provider,
            input_path=args.input,
            output_path=args.output,
            temp_output_path=args.temp_output,
            case_column=args.column,
        )
    else:
        parser.error(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main()
