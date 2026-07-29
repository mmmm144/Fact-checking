#!/usr/bin/env python3
"""Download the Vietnamese fact-checking dataset from Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError


LOGGER = logging.getLogger("hf_dataset_downloader")
DEFAULT_REPO_ID = "aiMy144/viet-fact-checking"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "vie" / "raw" / "viet-fact-checking"
ENV_PATH = PROJECT_ROOT / ".env"
TOKEN_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_TOKEN")


def configure_console_encoding() -> None:
    """Use UTF-8 for console output when the active Windows code page is limited."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_hf_token() -> str | None:
    """Load a Hugging Face token from the environment or project-root .env."""
    for variable_name in TOKEN_VARIABLES:
        token = os.getenv(variable_name)
        if token:
            return token.strip()

    if not ENV_PATH.is_file():
        return None

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
            continue

        key, value = stripped_line.split("=", 1)
        if key.strip() in TOKEN_VARIABLES:
            token = value.strip().strip('"').strip("'")
            if token:
                return token

    return None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download aiMy144/viet-fact-checking from Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch, tag, or commit to download (default: main).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files again instead of reusing the local cache.",
    )
    return parser


def download_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    revision: str = "main",
    force_download: bool = False,
) -> Path:
    """Download a dataset repository and return its local directory.

    ``snapshot_download`` verifies cached files and only downloads missing or
    changed files. Public datasets do not require a token. For private/gated
    repositories, set ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``.
    """
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    token = load_hf_token()
    LOGGER.info("Downloading dataset %s to %s", repo_id, destination)

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        token=token,
        force_download=force_download,
    )

    result = Path(downloaded_path).resolve()
    LOGGER.info("Dataset download completed: %s", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset downloader."""
    configure_console_encoding()
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        download_dataset(
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            revision=args.revision,
            force_download=args.force,
        )
    except HfHubHTTPError as exc:
        LOGGER.error("Hugging Face returned an HTTP error: %s", exc)
        return 1
    except OSError as exc:
        LOGGER.error("Cannot save the dataset: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
