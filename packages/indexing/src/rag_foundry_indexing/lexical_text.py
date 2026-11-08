"""Arabic-aware lexical tokenization for the in-process BM25 retriever.

The tokenizer folds alef variants, removes bidi controls, tatweel, and harakat,
folds Arabic-Indic digits, and casefolds text. It does not remove stopwords,
stem words, or fold teh-marbuta and yeh variants.
"""

import re

from rag_foundry_chunking.multilingual import normalize_with_map

TOKENIZER_ID = "lab-lexical-v1"
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_IDENTIFIERS = re.compile(r"[^\W_]+(?:[-_./:][^\W_]+)+", re.UNICODE)
_DIGITS = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9",
    "01234567890123456789",
)


def lexical_tokens(text: str) -> tuple[str, ...]:
    normalized, _ = normalize_with_map(text)
    normalized = normalized.translate(_DIGITS).casefold()
    tokens = [match.group(0) for match in _WORDS.finditer(normalized)]
    for match in _IDENTIFIERS.finditer(normalized):
        token = match.group(0)
        if any(char.isdigit() for char in token):
            tokens.append(token)
    return tuple(tokens)
