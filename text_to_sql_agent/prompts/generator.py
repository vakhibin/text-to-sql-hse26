"""Prompt templates for SQL candidate generation."""

GENERATOR_PROMPT = """
Generate one SQL query for the question using provided mSchema.
Output only SQL.
""".strip()


def build_generator_prompt(
    *,
    question: str,
    evidence: str | None,
    filtered_schema: str,
    query_sketch_text: str,
    few_shot_examples: list[dict[str, str]],
    value_hints_text: str = "",
    column_hints_text: str = "",
) -> str:
    """Build SQL generation prompt with optional few-shot examples."""
    few_shot_block = ""
    if few_shot_examples:
        chunks = []
        for idx, ex in enumerate(few_shot_examples, start=1):
            chunks.append(
                f"Example {idx}\n"
                f"Question: {ex.get('question', '')}\n"
                f"SQL: {ex.get('sql', '')}"
            )
        few_shot_block = "Few-shot examples:\n" + "\n\n".join(chunks) + "\n\n"

    query_sketch_block = query_sketch_text.strip() if query_sketch_text.strip() else "- (none)"
    evidence_block = evidence.strip() if evidence and evidence.strip() else "- (none)"
    linking_block_parts = []
    if value_hints_text:
        linking_block_parts.append(value_hints_text)
    if column_hints_text:
        linking_block_parts.append(column_hints_text)
    linking_block = "\n" + "\n\n".join(linking_block_parts) + "\n" if linking_block_parts else ""

    return f"""
You are an expert SQLite SQL generator.
Return ONLY SQL without markdown, explanations, or comments.

Question:
{question}

Evidence:
{evidence_block}

Query sketch:
{query_sketch_block}

mSchema:
{filtered_schema}
{linking_block}
{few_shot_block}Rules:
1) Use ONLY the tables and columns listed in the mSchema above. Do NOT invent or assume tables/columns that are not explicitly listed.
2) Use SQLite-compatible SQL.
3) Prefer explicit JOINs with ON clauses when a join is actually needed.
4) Include all required filters from the question.
5) If the question uses synonyms or informal wording, map them to the closest schema items that actually exist. Never rename schema identifiers to match the wording of the question.
6) Do not invent missing derived columns or guessed identifiers. If a column/table is not explicitly present in the schema, do not use it.
7) When uncertain between natural-language wording and schema naming, trust the schema naming.
8) Treat the query sketch as a planning scaffold: follow it when it is compatible with the schema and question, but trust the schema over the sketch if they conflict.
9) Before deciding joins, first determine the exact output columns needed by the question. The final SELECT list must answer the question directly.
10) SELECT column order MUST follow the order in which concepts appear in the question. "Find X and Y" → SELECT X, Y. "What is the Y and X?" → SELECT Y, X. Never reorder columns for aesthetic reasons.
11) Match the output shape implied by the question exactly. Do not add extra projected columns, and do not drop requested columns.
12) If the question asks for a name / title / description / text field, prefer the corresponding human-readable schema column, not an id/code/key column.
13) If the question asks for an id / code / number, do not substitute a name/description column instead.
14) If ALL requested columns exist in a single table, SELECT directly from that table. Do NOT join other tables just to "enrich" the output or because a foreign key exists.
15) Do not join extra tables only to make the SQL look more relational. A JOIN is only justified when the question asks for data that lives in different tables.
16) Use the exact schema identifiers verbatim. Do not shorten, naturalize, singularize/pluralize, or semantically substitute identifiers such as replacing `cost_of_treatment` with `cost`, `treatment_type_description` with `description`, or `cell_number` with `cell_phone`.
17) Do not introduce CAST, TRIM, LOWER, UPPER, substrings, or other value transformations unless the question or evidence explicitly requires that transformation.
18) Preserve the intended aggregation/comparison semantics. Do not replace COUNT with SUM, replace NOT IN with IN, or otherwise change the meaning just because it seems more natural.
19) When joins can duplicate entity rows but the question asks for unique entities/names/descriptions rather than events/records, use DISTINCT to avoid duplicate answers.
20) When the question asks for "all information about X" or "all details of X" and X maps to a single table, use SELECT * FROM that_table with appropriate WHERE filters. Do not enumerate columns manually or join related tables.
21) Preserve literal values faithfully. If the question or evidence provides an exact string/code/value, keep that value instead of normalizing or paraphrasing it.
22) When value hints are provided, use the exact `db_value` spelling in WHERE/HAVING clauses instead of paraphrasing the question wording. These values have been verified against the actual database.
23) When column hints are provided, use the exact `table.column` identifiers listed there. Do not shorten, rename, or move columns to different tables.
24) End query with semicolon.

SQL:
""".strip()
