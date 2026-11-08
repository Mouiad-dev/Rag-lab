# P5.1 lexical retrieval observations

Real output, 2026-11-08, on a fresh compose testdb. This is a small fixture check, not a benchmark.

Server: `PostgreSQL 18.6 (Debian 18.6-1.pgdg13+2)`, `server_encoding` UTF8, database `datcollate` / `datctype` `en_US.utf8`.

## R03 BM25 vs R04 PostgreSQL FTS on the same index

Command: `PYTHONIOENCODING=utf-8 uv run pytest tests/indexing/test_fts_db.py -m db -q -s` (1 passed).
The index holds the `tests/indexing/_fixture.py` Markdown plus two Arabic lines that name the part numbers `AB-1234` and `AB-1243`. Top 3 hits per technique; scores are from different formulas and are not comparable.

Query `AB-1234` (R04 in `and` mode):

| Technique | Rank | Score | Components | Chunk starts with |
|---|---:|---:|---|---|
| R03 | 1 | 8.4872 | `bm25_term` ab 3.4316, 1234 2.5278, ab-1234 2.5278 | Arabic part-number line |
| R04 | 1 | 0.1 | `ts_rank_cd` 0.1 | Arabic part-number line with AB-1234 |

Query `How often does the Port Sallow observatory record sea levels?` (R04 in `or` mode; in `and` mode it returns no hits):

| Technique | Rank | Score | Components | Chunk starts with |
|---|---:|---:|---|---|
| R03 | 1 | 13.4446 | `bm25_term` the 1.1436, port 2.4602, sallow 2.4602, observatory 2.4602, sea 2.4602, levels 2.4602 | `# Foundry field notes` / tidal observatory |
| R03 | 2 | 1.7373 | `bm25_term` the 1.7373 | Farmers bring apples and honey |
| R03 | 3 | 1.5658 | `bm25_term` the 1.5658 | Volunteers repaint the wooden bridge |
| R04 | 1 | 0.6 | `ts_rank_cd` 0.6 | `# Foundry field notes` / tidal observatory |
| R04 | 2 | 0.3 | `ts_rank_cd` 0.3 | Farmers bring apples and honey |
| R04 | 3 | 0.2 | `ts_rank_cd` 0.2 | The mountain railway climbs |

Diacritized Arabic query `المَخْزَن` (the text has `المخزن`): R04 `and` and `or` return no hit with the part number; R03 returns it (the test asserts all three).

## What the `simple` configuration does with Arabic and identifiers

Run in the testdb with `psql` (input written as `E'\u...'` escapes):

| Input | `to_tsvector('simple', ...)` |
|---|---|
| `كِتَـاب` (diacritics + tatweel) | `'كِتَـاب':1` |
| `كتاب` | `'كتاب':1` |
| `ق-٧٧٢` (Arabic-Indic digits) | `'ق':2 'ق-٧٧٢':1 '٧٧٢':3` |
| `AB-1234` | `'-1234':2 'ab':1` |
| `AB 1234` | `'1234':2 'ab':1` |
| `AB-1243` | `'-1243':2 'ab':1` |
| `إدارة` | `'إدارة':1` |

`websearch_to_tsquery('simple', 'كِتَاب')` gives `'كِتَاب'`.

Observed: `simple` keeps diacritics and tatweel inside the lexeme and does not fold alef forms, so a vowelled query and plain text do not meet. Arabic-Indic digits stay as they are. `AB-1234` becomes the word `ab` and the signed number `-1234`, so it does not match `AB-1243` (`-1243`) but also does not match the spaced form `AB 1234` (`1234`).

R03 (`lab-lexical-v1`) folds diacritics, tatweel, alef forms and Arabic-Indic digits before scoring, and keeps the whole identifier `ab-1234` as an extra token next to `ab` and `1234`.
