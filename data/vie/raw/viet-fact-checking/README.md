---
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
