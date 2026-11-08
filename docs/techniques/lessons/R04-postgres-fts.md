# R04: PostgreSQL FTS

## Intuition

PostgreSQL full-text search provides lexical matching and ranking through PostgreSQL’s text-search engine. It is a separate baseline with analyzer behavior and score semantics of its own.

## Mechanism

`FtsRetriever` is implemented in `packages/indexing/src/rag_foundry_indexing/fts.py`. It uses PostgreSQL’s `simple` text-search configuration, `websearch_to_tsquery`, `ts_rank_cd` with normalization `0`, and a generated `tsvector` column indexed by GIN. The SQL applies the scope filter with the search so results remain scoped.

This is PostgreSQL FTS, not BM25. The `simple` configuration lowercases but does not stem, remove stop words, or normalize Arabic diacritics, tatweel, or alef forms. `and` is the default query operator; the configurable `or` mode can retrieve partial query matches. There is no offline demo because this technique requires PostgreSQL.

## Worked example

The real testdb run used the command below. The observed results, including the exact identifier and English query, are copied from [lexical observations](../../evidence/P5.1/lexical_observations.md).

```text
PYTHONIOENCODING=utf-8 uv run pytest tests/indexing/test_fts_db.py -m db -q -s
1 passed
```

For `AB-1234`, R04 in `and` mode ranked the matching Arabic part-number line first at `0.1` (`ts_rank_cd` `0.1`). For `How often does the Port Sallow observatory record sea levels?`, R04 `and` mode returned no hits; in `or` mode its top three ranks and scores were `1 / 0.6`, `2 / 0.3`, and `3 / 0.2`. The evidence table gives the corresponding chunk beginnings and the R03 results; these formulas produce scores that are not comparable.

The observed R03/R04 table from the evidence file:

| Technique | Rank | Score | Components | Chunk starts with |
|---|---:|---:|---|---|
| R03 | 1 | 8.4872 | `bm25_term` ab 3.4316, 1234 2.5278, ab-1234 2.5278 | Arabic part-number line |
| R04 | 1 | 0.1 | `ts_rank_cd` 0.1 | Arabic part-number line with AB-1234 |

| Technique | Rank | Score | Components | Chunk starts with |
|---|---:|---:|---|---|
| R03 | 1 | 13.4446 | `bm25_term` the 1.1436, port 2.4602, sallow 2.4602, observatory 2.4602, sea 2.4602, levels 2.4602 | `# Foundry field notes` / tidal observatory |
| R03 | 2 | 1.7373 | `bm25_term` the 1.7373 | Farmers bring apples and honey |
| R03 | 3 | 1.5658 | `bm25_term` the 1.5658 | Volunteers repaint the wooden bridge |
| R04 | 1 | 0.6 | `ts_rank_cd` 0.6 | `# Foundry field notes` / tidal observatory |
| R04 | 2 | 0.3 | `ts_rank_cd` 0.3 | Farmers bring apples and honey |
| R04 | 3 | 0.2 | `ts_rank_cd` 0.2 | The mountain railway climbs |

Scores use different formulas and are not comparable.

## Failure case

The testdb run diacritized Arabic query `\u0627\u0644\u0645\u064e\u062e\u0652\u0632\u064e\u0646` returned no hits in R04 `and` or `or` mode; R03 returned the part-number hit. This is the observed losing case for `simple`, not an inferred behavior.

The probe also records the identifier parsing: `to_tsvector('simple', 'AB-1234')` yields `'ab':1` and `'-1234':2`. Thus the hyphenated form is parsed as `ab` plus `-1234`; the complete identifier is not kept as one lexeme. The full probe table, including the Arabic diacritics, tatweel, Arabic-Indic digits, and the `AB 1234` and `AB-1243` rows, is quoted in the evidence file.

The evidence file's observed `simple` probe table:

| Input | `to_tsvector('simple', ...)` |
|---|---|
| `\u0643\u0650\u062a\u064e\u0640\u0627\u0628` (diacritics + tatweel) | `'\u0643\u0650\u062a\u064e\u0640\u0627\u0628':1` |
| `\u0643\u062a\u0627\u0628` | `'\u0643\u062a\u0627\u0628':1` |
| `\u0642-\u0667\u0667\u0662` (Arabic-Indic digits) | `'\u0642':2 '\u0642-\u0667\u0667\u0662':1 '\u0667\u0667\u0662':3` |
| `AB-1234` | `'-1234':2 'ab':1` |
| `AB 1234` | `'1234':2 'ab':1` |
| `AB-1243` | `'-1243':2 'ab':1` |
| `\u0625\u062f\u0627\u0631\u0629` | `'\u0625\u062f\u0627\u0631\u0629':1` |


## Limitations

The observed PostgreSQL version was `PostgreSQL 18.6 (Debian 18.6-1.pgdg13+2)` with UTF8 server encoding and `en_US.utf8` database `datcollate` / `datctype`. Analyzer behavior is specific to this configuration and environment. The `simple` analyzer does not normalize Arabic forms, and `and` can be restrictive for longer natural-language queries. Scores are not comparable across queries or with the R03 formula.

## Baseline comparison

See the two exact evidence tables in [lexical observations](../../evidence/P5.1/lexical_observations.md): the R03/R04 result table and the `simple` probe table. They show the matching `AB-1234` row, the English query's `and`/`or` behavior, diacritized Arabic miss, and tokenization output without adding interpretation beyond observed output.

## Status

Implemented; not validated or benchmarked.
