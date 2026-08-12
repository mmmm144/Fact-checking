'''Check duplicate article IDs within and across two JSON files.'''

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_FILE_1 = ROOT_DIR / 'generation_claim' / 'claim_01_part1.json'
DEFAULT_FILE_2 = ROOT_DIR / 'generation_claim' / 'claim_01_part2.json'


def load_id_counts(path: Path) -> tuple[Counter, int, int]:
    '''Return ID frequencies, total articles, and articles without an ID.'''
    with path.open('r', encoding='utf-8') as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f'{path} must contain a JSON array, not {type(data).__name__}.'
        )

    ids = []
    missing_id_count = 0
    for index, article in enumerate(data):
        if not isinstance(article, dict):
            raise ValueError(f'Item {index} in {path} is not a JSON object.')

        article_id = article.get('id')
        if article_id is None or article_id == '':
            missing_id_count += 1
            continue
        if not isinstance(article_id, (str, int)) or isinstance(article_id, bool):
            raise ValueError(f'ID at item {index} in {path} must be a string or integer.')
        ids.append(article_id)

    return Counter(ids), len(data), missing_id_count


def get_duplicates(id_counts: Counter) -> dict:
    '''Return IDs that occur more than once.'''
    return {article_id: count for article_id, count in id_counts.items() if count > 1}


def print_file_report(path: Path, id_counts: Counter, total: int, missing: int) -> None:
    duplicates = get_duplicates(id_counts)
    duplicate_articles = sum(duplicates.values())
    extra_articles = sum(count - 1 for count in duplicates.values())

    print(f'\nFILE: {path}')
    print(f'  Total articles: {total:,}')
    print(f'  Unique IDs: {len(id_counts):,}')
    print(f'  Articles without ID: {missing:,}')
    print(f'  Distinct duplicated IDs: {len(duplicates):,}')
    print(f'  Articles whose ID is duplicated: {duplicate_articles:,}')
    print(f'  Extra duplicate articles (keeping one per ID): {extra_articles:,}')

    if duplicates:
        print('  Duplicate details (ID: occurrence count):')
        for article_id, count in sorted(duplicates.items(), key=lambda item: str(item[0])):
            print(f'    {article_id}: {count}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Count duplicate article IDs within and across two JSON files.'
    )
    parser.add_argument('file1', nargs='?', type=Path, default=DEFAULT_FILE_1)
    parser.add_argument('file2', nargs='?', type=Path, default=DEFAULT_FILE_2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        counts1, total1, missing1 = load_id_counts(args.file1)
        counts2, total2, missing2 = load_id_counts(args.file2)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1

    print_file_report(args.file1, counts1, total1, missing1)
    print_file_report(args.file2, counts2, total2, missing2)

    common_ids = sorted(counts1.keys() & counts2.keys(), key=str)
    print('\nCROSS-FILE COMPARISON')
    print(f'  IDs appearing in both files: {len(common_ids):,}')
    if common_ids:
        print('  Shared ID details (count in file 1, count in file 2):')
        for article_id in common_ids:
            print(f'    {article_id}: {counts1[article_id]}, {counts2[article_id]}')
    else:
        print('  The two files have no article IDs in common.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
