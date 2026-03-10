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

    return f"""
You are an expert SQLite SQL generator.
Return ONLY SQL without markdown, explanations, or comments.

Question:
{question}

Complexity:
{complexity}

Decomposition hints:
{sub_questions_block}

mSchema:
{filtered_schema}

{few_shot_block}Rules:
1) Use ONLY the tables and columns listed in the mSchema above. Do NOT invent or assume tables/columns that are not explicitly listed.
2) Use SQLite-compatible SQL.
3) Prefer explicit JOINs with ON clauses.
4) Include all required filters from the question.
5) End query with semicolon.

SQL:
""".strip()

