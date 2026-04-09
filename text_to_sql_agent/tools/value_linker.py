"""Pre-generation value linking: find actual DB values matching question entities."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "and", "or", "but", "not", "no", "nor", "if", "then", "else", "than",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "about", "against", "each", "every", "all", "both",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "so", "very", "just", "also", "too",
    "find", "show", "list", "give", "get", "display", "return", "tell",
    "name", "names", "number", "total", "count", "average", "maximum",
    "minimum", "many", "much", "least", "most", "highest", "lowest",
    "different", "distinct", "unique", "order", "sort", "group",
    "ascending", "descending", "asc", "desc",
})

_SQL_KEYWORDS = frozenset({
    "select", "from", "where", "join", "inner", "left", "right", "outer",
    "on", "and", "or", "not", "in", "between", "like", "is", "null",
    "group", "by", "having", "order", "asc", "desc", "limit", "offset",
    "union", "intersect", "except", "insert", "update", "delete", "create",
    "drop", "alter", "table", "index", "view", "as", "distinct", "count",
    "sum", "avg", "min", "max", "case", "when", "then", "end", "exists",
})

_NUMERIC_COL_TYPES = frozenset({"number", "integer", "real", "float", "numeric", "int"})

_MAX_CANDIDATES = 25
_MAX_NGRAM = 4
_MAX_HINTS = 15


@dataclass
class ValueHint:
    entity: str
    table: str
    column: str
    db_value: str

    def as_dict(self) -> dict[str, str]:
        return {
            "entity": self.entity,
            "table": self.table,
            "column": self.column,
            "db_value": self.db_value,
        }


def _extract_quoted_strings(question: str) -> list[str]:
    """Extract explicitly quoted strings from the question."""
    return re.findall(r"""['"]([^'"]{2,})['"]""", question)


def _extract_candidate_entities(question: str) -> list[str]:
    """Extract candidate entities from question via named-entity heuristics + n-grams.

    Priority: quoted strings > capitalized words > multi-word n-grams.
    """
    quoted = _extract_quoted_strings(question)

    cleaned = re.sub(r"""['"][^'"]*['"]""", " ", question)
    cleaned = re.sub(r"[?!.,;:()\[\]{}]", " ", cleaned)
    tokens = cleaned.split()

    candidates: list[str] = list(quoted)
    seen_lower: set[str] = {q.lower() for q in quoted}

    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower in seen_lower or tok_lower in _STOPWORDS or tok_lower in _SQL_KEYWORDS:
            continue
        if len(tok) < 2:
            continue
        if tok[0].isupper() or re.fullmatch(r"\d{4}", tok):
            candidates.append(tok)
            seen_lower.add(tok_lower)

    for n in range(_MAX_NGRAM, 1, -1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            gram_lower = gram.lower()

            if gram_lower in seen_lower:
                continue
            if all(t.lower() in _STOPWORDS or t.lower() in _SQL_KEYWORDS for t in gram.split()):
                continue

            has_content = any(
                t.lower() not in _STOPWORDS and t.lower() not in _SQL_KEYWORDS
                for t in gram.split()
            )
            if has_content:
                candidates.append(gram)
                seen_lower.add(gram_lower)

            if len(candidates) >= _MAX_CANDIDATES:
                break
        if len(candidates) >= _MAX_CANDIDATES:
            break

    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower in seen_lower or tok_lower in _STOPWORDS or tok_lower in _SQL_KEYWORDS:
            continue
        if len(tok) < 2:
            continue
        candidates.append(tok)
        seen_lower.add(tok_lower)
        if len(candidates) >= _MAX_CANDIDATES:
            break

    return candidates


def _is_text_column(col: dict[str, Any]) -> bool:
    col_type = str(col.get("type", "")).lower()
    return col_type not in _NUMERIC_COL_TYPES


def _search_value_in_column(
    db_path: str,
    table: str,
    column: str,
    entity: str,
    *,
    allow_like: bool = False,
) -> str | None:
    """Search for entity in a specific column. Returns the actual DB value or None.

    ``allow_like`` enables substring LIKE matching — use only for
    entities that were explicitly quoted in the question.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            row = conn.execute(
                f"SELECT [{column}] FROM [{table}] "
                f"WHERE [{column}] = ? COLLATE NOCASE LIMIT 1",
                (entity,),
            ).fetchone()
            if row:
                return str(row[0])

            if allow_like and len(entity) >= 3:
                row = conn.execute(
                    f"SELECT [{column}] FROM [{table}] "
                    f"WHERE [{column}] LIKE ? COLLATE NOCASE LIMIT 1",
                    (f"%{entity}%",),
                ).fetchone()
                if row:
                    return str(row[0])
    except Exception:
        pass
    return None


def link_values(
    question: str,
    full_schema: dict[str, Any],
    selected_tables: list[str] | None = None,
) -> list[ValueHint]:
    """Find actual DB values matching question entities.

    Searches text columns in selected tables (or all tables if none specified).
    LIKE substring matching is only enabled for explicitly quoted strings.
    Returns up to _MAX_HINTS verified hints.
    """
    db_path = str(full_schema.get("db_path", "")).strip()
    if not db_path or not question.strip():
        return []

    quoted_set = {s.lower() for s in _extract_quoted_strings(question)}
    candidates = _extract_candidate_entities(question)
    if not candidates:
        return []

    tables = full_schema.get("tables", [])
    if selected_tables:
        selected_set = {t.lower() for t in selected_tables}
        tables = [t for t in tables if str(t.get("name", "")).lower() in selected_set]

    hints: list[ValueHint] = []
    seen: set[tuple[str, str, str]] = set()

    for entity in candidates:
        allow_like = entity.lower() in quoted_set
        for table in tables:
            tname = str(table.get("name", ""))
            for col in table.get("columns", []):
                if not _is_text_column(col):
                    continue
                cname = str(col.get("name", ""))
                key = (entity.lower(), tname.lower(), cname.lower())
                if key in seen:
                    continue
                seen.add(key)

                db_value = _search_value_in_column(
                    db_path, tname, cname, entity, allow_like=allow_like,
                )
                if db_value is not None:
                    hints.append(ValueHint(
                        entity=entity,
                        table=tname,
                        column=cname,
                        db_value=db_value,
                    ))
                    if len(hints) >= _MAX_HINTS:
                        return hints

    return hints


def format_value_hints(hints: list[ValueHint] | list[dict]) -> str:
    """Format value hints as a text block for prompts."""
    if not hints:
        return ""
    lines = []
    for h in hints:
        if isinstance(h, ValueHint):
            lines.append(f'- "{h.entity}" found in {h.table}.{h.column} as: \'{h.db_value}\'')
        else:
            lines.append(
                f'- "{h.get("entity", "")}" found in '
                f'{h.get("table", "")}.{h.get("column", "")} as: \'{h.get("db_value", "")}\''
            )
    return "Value hints (verified from database):\n" + "\n".join(lines)
