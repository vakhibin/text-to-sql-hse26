"""Prompt templates for SQL candidate generation."""

GENERATOR_PROMPT = """
Generate one SQL query for the question using provided mSchema.
Output only SQL.
""".strip()


def build_generator_prompt(
    *,
    question: str,
    filtered_schema: str,
    complexity: str,
    sub_questions: list[str],
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

    return f"""
You are an expert SQLite SQL generator.
Return ONLY SQL without markdown, explanations, or comments.

Question:
{question}

Complexity:
{complexity}

Decomposition hints:
{sub_questions_block}

Query sketch:
{query_sketch_block}

mSchema:
{filtered_schema}

{few_shot_block}Rules:
1) Use ONLY the tables and columns listed in the mSchema above. Do NOT invent or assume tables/columns that are not explicitly listed.
2) Use SQLite-compatible SQL.
3) Prefer explicit JOINs with ON clauses.
4) Include all required filters from the question.
5) If the question uses synonyms or informal wording, map them to the closest schema items that actually exist. Never rename schema identifiers to match the wording of the question.
6) Do not invent missing derived columns or guessed identifiers. If a column/table is not explicitly present in the schema, do not use it.
7) When uncertain between natural-language wording and schema naming, trust the schema naming.
8) Treat the query sketch as a planning scaffold: follow it when it is compatible with the schema and question, but trust the schema over the sketch if they conflict.
9) End query with semicolon.

SQL:
""".strip()

