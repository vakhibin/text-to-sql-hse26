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
    complexity: str,
    sub_questions: list[str],
    decomposition_risk_flags: dict[str, object],
    query_sketch_text: str,
    few_shot_examples: list[dict[str, str]],
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

    sub_questions_block = ""
    if sub_questions:
        sub_questions_block = "\n".join(f"- {item}" for item in sub_questions)
    else:
        sub_questions_block = "- (none)"

    query_sketch_block = query_sketch_text.strip() if query_sketch_text.strip() else "- (none)"
    evidence_block = evidence.strip() if evidence and evidence.strip() else "- (none)"
    risk_flags_block = ", ".join(
        f"{key}={value}" for key, value in decomposition_risk_flags.items()
    ) or "- (none)"

    return f"""
You are an expert SQLite SQL generator.
Return ONLY SQL without markdown, explanations, or comments.

Question:
{question}

Complexity:
{complexity}

Decomposition hints:
{sub_questions_block}

Evidence:
{evidence_block}

Routing risk flags:
{risk_flags_block}

Query sketch:
{query_sketch_block}

mSchema:
{filtered_schema}

{few_shot_block}Rules:
1) Use ONLY the tables and columns listed in the mSchema above. Do NOT invent or assume tables/columns that are not explicitly listed.
2) Use SQLite-compatible SQL.
3) Prefer explicit JOINs with ON clauses when a join is actually needed.
4) Include all required filters from the question.
5) If the question uses synonyms or informal wording, map them to the closest schema items that actually exist. Never rename schema identifiers to match the wording of the question.
6) Do not invent missing derived columns or guessed identifiers. If a column/table is not explicitly present in the schema, do not use it.
7) When uncertain between natural-language wording and schema naming, trust the schema naming.
8) Treat the query sketch as a planning scaffold: follow it when it is compatible with the schema and question, but trust the schema over the sketch if they conflict.
9) SELECT column order MUST follow the order in which concepts appear in the question. "Find X and Y" → SELECT X, Y. "What is the Y and X?" → SELECT Y, X. Never reorder columns for aesthetic reasons.
10) Match the output shape implied by the question. Do not add extra projected columns, and do not drop requested columns.
11) If ALL requested columns exist in a single table, SELECT directly from that table. Do NOT join other tables just to "enrich" the output or because a foreign key exists.
12) Do not join extra tables only to make the SQL look more relational. A JOIN is only justified when the question asks for data that lives in different tables.
13) When the question asks for "all information about X" or "all details of X" and X maps to a single table, use SELECT * FROM that_table with appropriate WHERE filters. Do not enumerate columns manually or join related tables.
14) Preserve literal values faithfully. If the question or evidence provides an exact string/code/value, keep that value instead of normalizing or paraphrasing it.
15) When `literal_filter_risk=true`, be extra careful with literal spelling, spacing, casing, and code values.
16) End query with semicolon.

SQL:
""".strip()

