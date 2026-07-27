#!/usr/bin/env python3
"""Uploader script for pushing Vietnamese Evidence Corpus v1.0 to Hugging Face Datasets."""

import logging
from pathlib import Path
from huggingface_hub import HfApi

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("hf_uploader")

REPO_ID = "aiMy144/viet-fact-checking"
CORPUS_PATH = Path("src/clean_normalize/output/corpus_v1.json")
ENV_PATH = Path(".env")


def load_hf_token() -> str | None:
    """Read Hugging Face write token from local .env file."""
    if not ENV_PATH.exists():
        LOGGER.error(".env file not found in project root.")
        return None
        
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if "HUGGING_FACE" in line:
                # Extract token from HUGGING_FACE="..."
                parts = line.split("=")
                if len(parts) >= 2:
                    token = parts[1].strip().strip('"').strip("'")
                    return token
    return None


def generate_dataset_card() -> str:
    """Create a structured YAML-frontmatter README for Hugging Face."""
    return """---
language:
- vi
- en
license: cc-by-4.0
size_categories:
- 10K<n<100K
task_categories:
- text-retrieval
- question-answering
tags:
- fact-checking
- rag
- evidence-corpus
- vietnamese
pretty_name: Vietnamese Evidence Corpus v1.0
---

# Vietnamese Evidence Corpus for Fact-Checking & RAG (v1.0)

This dataset is a clean, standardized, and unified **Vietnamese Evidence Corpus (v1.0)** built for research in **Information Retrieval, Retrieval-Augmented Generation (RAG), and Fact-Checking / Claim Verification**.

## Dataset Statistics

- **Total Documents**: 13,572 (frozen unique records, duplicates filtered out)
- **Languages**: ~70% Vietnamese (`vi`), ~30% English (`en`)
- **Size**: 115.33 MB

### Documents by Source

| Source Name | Source Type | Domain | Doc Type | Count |
| :--- | :--- | :--- | :--- | :--- |
| **GSO** | Government | Economy | Statistics | 2,393 |
| **WHO** | International Org | Health | Report | 2,400 |
| **MOH** | Government | Health | Guideline | 1,799 |
| **World Bank** | International Org | Economy | Report | 1,397 |
| **Báo Chính phủ** | Government | Government & Policy | News | 2,500 |
| **VnExpress** | News Agency | General News | News | 2,600 |
| **VAFC** | Fact Checking Portal | Fact Checking | Debunking | 483 |

---

## Schema Reference

Every record contains the following standard structure:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `doc_id` | `string` | Unique identifier (e.g., `WHO_a1b2c3d4`) |
| `title` | `string` | Cleansed claim or headline of the article |
| `text` | `string` | Cleaned body/justification text (free of HTML markup & boilerplate) |
| `summary` | `string` or `null` | Sapo/abstract summary of the document |
| `source` | `string` | Source publisher name |
| `source_type` | `string` | Category: `Government`, `International Organization`, `News Agency`, `Fact Checking Portal` |
| `domain` | `string` | Domain: `Health`, `Economy`, `Government & Policy`, `Fact Checking`, `General News` |
| `document_type` | `string` | Genre: `News`, `Report`, `Guideline`, `Press Release`, `Statistics`, `Debunking` |
| `language` | `string` | ISO 639-1 code (`vi` or `en`) |
| `country` | `string` | Focus country/context (`Vietnam` or `Global`) |
| `publish_date` | `string` or `null` | Normalized ISO date (`YYYY-MM-DD`) |
| `crawl_date` | `string` | Date of retrieval |
| `url` | `string` | Source canonical link |
| `author` | `string` or `null` | Author (if available) |
| `metadata` | `object` | Metric stats: `word_count`, `char_count`, `sentence_count`, `reading_time` |
| `quality` | `object` | Preprocessing audit flags: `cleaned`, `html_removed`, `duplicate`, `language_verified` |

---

## Usage

You can download `corpus_v1.json` directly from the repository Files tab or load it via the `datasets` library.
"""


def main():
    token = load_hf_token()
    if not token:
        LOGGER.error("Hugging Face write token is missing in .env. Exiting.")
        return
        
    if not CORPUS_PATH.exists():
        LOGGER.error("Corpus file not found at: %s. Run the cleaning script first.", CORPUS_PATH)
        return
        
    api = HfApi()
    
    # 1. Create dataset repo if it does not exist
    LOGGER.info("Verifying/creating Hugging Face dataset repository: %s", REPO_ID)
    try:
        api.create_repo(
            repo_id=REPO_ID,
            token=token,
            repo_type="dataset",
            exist_ok=True
        )
    except Exception as e:
        LOGGER.error("Failed to create/access repo %s: %s", REPO_ID, e)
        return
        
    # 2. Upload corpus_v1.json
    LOGGER.info("Uploading %s to Hugging Face...", CORPUS_PATH.name)
    try:
        api.upload_file(
            path_or_fileobj=str(CORPUS_PATH),
            path_in_repo="corpus_v1.json",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token
        )
        LOGGER.info("Successfully uploaded corpus_v1.json to %s!", REPO_ID)
    except Exception as e:
        LOGGER.error("Failed to upload corpus_v1.json: %s", e)
        return
        
    # 3. Create and upload Dataset Card (README.md)
    readme_content = generate_dataset_card()
    readme_temp_path = Path("src/clean_normalize/README_temp.md")
    with open(readme_temp_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    LOGGER.info("Uploading Dataset Card (README.md) to Hugging Face...")
    try:
        api.upload_file(
            path_or_fileobj=str(readme_temp_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token
        )
        LOGGER.info("Successfully uploaded README.md (Dataset Card) to %s!", REPO_ID)
    except Exception as e:
        LOGGER.error("Failed to upload README.md: %s", e)
    finally:
        if readme_temp_path.exists():
            readme_temp_path.unlink()
            
    LOGGER.info("Dataset upload workflow completed successfully!")


if __name__ == "__main__":
    main()
