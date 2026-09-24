# Chunk v3 data-quality audit

Artifact audited:
`Loctran123/vietnamese-evidence-corpus-chunked-e5-v3` at revision
`d669d204dbb4d245e980aa18aa1bd7caf314504e`.

## Results

### Published v3 before repair

| Check | Result |
|---|---:|
| Source chunks | 53,267 |
| Source documents | 13,607 |
| Unique normalized passage (`title + text`) | 53,114 |
| Duplicate passage/content-hash rows | 153 |
| Unique normalized chunk text | 52,533 |
| Duplicate text rows, ignoring title | 734 |
| Chunks containing `Tham khảo thêm` | 2,073 |
| Strict Báo Chính phủ related-suffix markers | 2,071 |
| Chunks containing a WHO `Related link(s)` heading | 70 |
| Chunks containing `Xem thêm` | 3 |

The embedding pipeline correctly collapses the 153 exact duplicate passages,
which explains the reduction from 53,267 source chunks to 53,114 embedding
rows. The larger text-only duplicate count includes passages with different
titles and must not be deleted automatically.

The three `Xem thêm` matches are legitimate prose or event information rather
than navigation. In contrast, the 2,071 strict `Tham khảo thêm` markers are
related-article suffixes appended by Báo Chính phủ. Chunking v3.1 removes that
suffix only for this source and preserves lowercase phrases used inside normal
sentences.

WHO `Related link(s)` sections are reported separately and are not removed
automatically because some pages interleave them with substantive assembly
updates. They require either DOM-level extraction rules or a manual sample audit
before deletion.

The published chunk v3 is not derived from the frozen 13,572-document corpus:
it contains 61 document IDs absent from the frozen corpus and omits 26 frozen
document IDs. All 26 omitted documents are referenced by the generated claims.
This makes the published v3 unsuitable as the final retrieval corpus.

### Rebuilt local v3.1

The corpus was rebuilt from `aiMy144/viet-fact-checking` revision
`61ec51be1ca8270af4ec7ecb06d4e88a1431c488` after adding source-specific suffix
cleaning for Báo Chính phủ and VnExpress.

| Check | Result |
|---|---:|
| Source documents | 13,572 |
| Output chunks | 53,861 |
| Unique normalized passage (`title + text`) | 53,747 |
| Duplicate passage/content-hash rows | 114 |
| Unique normalized chunk text | 53,074 |
| Duplicate text rows, ignoring title | 787 |
| Strict Báo Chính phủ related-suffix markers | 0 |
| VnExpress Google tutorial blocks | 0 |
| Natural-language `Tham khảo thêm` matches | 2 |
| Natural-language `Xem thêm` matches | 4 |
| WHO `Related link(s)` headings retained for review | 70 |

The remaining Vietnamese matches are normal prose and must not be deleted.
Exact duplicate passages will be collapsed by the embedding pipeline while
retaining provenance.

The claim audit found 35 claims without a valid evidence document ID: 33
evidence entries omitted `article_id`, and seven non-empty IDs contained typos
or corrupted text. All 40 affected evidence entries have now been repaired to
their valid parent document IDs. The resulting 73,454 claims contain no missing
or unindexable evidence IDs and no empty evidence quotes. Retrieval evaluation
also retains an explicit parent-document fallback for defensive compatibility
with older claim revisions.

## Reproduce

```bash
python chunking/audit_chunk_corpus.py /path/to/corpus_v1_chunked_e5_v3.json
```
