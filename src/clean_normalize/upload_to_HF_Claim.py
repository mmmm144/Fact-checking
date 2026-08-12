#!/usr/bin/env python3
'''Upload the generated claim-verification dataset to Hugging Face.'''

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLAIM_FILE = PROJECT_ROOT / 'generation_claim' / 'claim_01.json'
DEFAULT_REPO_ID = 'Loctran123/vietnamese-fact-checking-claims'
DEFAULT_PATH_IN_REPO = 'data/claim_01.json'
ENV_FILE = PROJECT_ROOT / '.env'
TOKEN_VARIABLE = 'HUGGING_FACE_HUB_TOKEN'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
LOGGER = logging.getLogger('hf_claim_uploader')


def load_hf_token() -> str | None:
    '''Load a Hugging Face write token without logging its value.'''

    token = os.getenv(TOKEN_VARIABLE)
    if token:
        return token.strip()

    if not ENV_FILE.is_file():
        return None

    for line in ENV_FILE.read_text(encoding='utf-8-sig').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, value = stripped.split('=', 1)
        if key.strip() == TOKEN_VARIABLE:
            token = value.strip().strip(chr(34)).strip(chr(39))
            if token:
                return token
    return None


def dataset_card(repo_id: str, path_in_repo: str) -> str:
    '''Return the dataset card for the generated claim dataset.'''

    return f'''---
language:
- vi
- en
license: other
size_categories:
- 10K<n<100K
task_categories:
- text-classification
- question-answering
tags:
- fact-checking
- claim-verification
- natural-language-inference
- vietnamese
pretty_name: Vietnamese Fact-Checking Claims
---

# Vietnamese Fact-Checking Claims

Generated claim-verification data derived from the Vietnamese Evidence Corpus.
Each article contains claims labeled as supported, refuted, or not having enough
information, together with evidence and a short rationale.

## Statistics

- 12,238 source articles
- 73,454 generated claims
- 24,476 `SUPPORTED` claims
- 24,502 `REFUTED` claims
- 24,476 `NOT_ENOUGH_INFO` claims

## Main fields

- Article: `id`, `date_iso`, `full_text`, `claims`
- Claim: `claim`, `label`, `evidence`, `reason`
- Evidence: `type`, `quote`, `article_id`, `url`

The `claims` field groups claim records under `SUPPORTED`, `REFUTED`, and
`NOT_ENOUGH_INFO`.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    '{repo_id}',
    data_files='{path_in_repo}',
    split='train',
)
```

This dataset contains generated annotations. Users should validate labels and
evidence before using the data in high-stakes settings. The repository does not
grant additional rights over the original source articles.
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Upload the generated claim dataset to Hugging Face.'
    )
    parser.add_argument('--repo-id', default=DEFAULT_REPO_ID)
    parser.add_argument('--claim-file', type=Path, default=DEFAULT_CLAIM_FILE)
    parser.add_argument('--path-in-repo', default=DEFAULT_PATH_IN_REPO)
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create a private dataset repository (default: public).',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    claim_path = args.claim_file.resolve()
    if not claim_path.is_file():
        LOGGER.error('Claim file not found: %s', claim_path)
        return 2

    token = load_hf_token()
    if not token:
        LOGGER.error(
            '%s was not found in the environment or %s',
            TOKEN_VARIABLE,
            ENV_FILE,
        )
        return 2

    api = HfApi(token=token)
    try:
        account = api.whoami()
        LOGGER.info('Authenticated as %s', account.get('name', '<unknown>'))
        repo_url = api.create_repo(
            repo_id=args.repo_id,
            repo_type='dataset',
            private=args.private,
            exist_ok=True,
        )
        LOGGER.info('Dataset repository ready: %s', repo_url)

        LOGGER.info(
            'Uploading %s (%.2f MiB) as %s',
            claim_path,
            claim_path.stat().st_size / (1024 * 1024),
            args.path_in_repo,
        )
        api.upload_file(
            path_or_fileobj=claim_path,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            repo_type='dataset',
            commit_message='Upload generated claim dataset',
        )

        api.upload_file(
            path_or_fileobj=dataset_card(
                args.repo_id, args.path_in_repo
            ).encode('utf-8'),
            path_in_repo='README.md',
            repo_id=args.repo_id,
            repo_type='dataset',
            commit_message='Add claim dataset card',
        )
    except Exception:
        LOGGER.exception('Hugging Face upload failed')
        return 1

    LOGGER.info('Upload complete: https://huggingface.co/datasets/%s', args.repo_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
