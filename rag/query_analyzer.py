from __future__ import annotations

from pydantic import BaseModel, Field

from .cancer_mapping import canonical_cancer_family
from .llm import parse_json_object


class QueryAnalysis(BaseModel):
    cancer_type: str | None = None
    cancer_family: str | None = None
    clinical_topics: list[str] = Field(default_factory=list)
    claims_to_check: list[str] = Field(default_factory=list)
    explicit_question: str
    search_queries: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """
You are a query analyzer for a cancer-information retrieval system.

The user's question may contain incorrect, overgeneralized, or unsupported assumptions.
Your job is NOT to decide whether those assumptions are true.
Your job is to extract them as claims that should be checked against NCI PDQ evidence.

Return JSON only with:
{
  "cancer_type": string or null,
  "clinical_topics": [strings],
  "claims_to_check": [standalone factual claims],
  "explicit_question": "the user's underlying information need",
  "search_queries": [3-6 short retrieval queries]
}

Requirements:
- Preserve clinically important terms such as stage, treatment names, biomarkers, procedures, and age.
- Include search queries that test the embedded assumptions, not only the final request.
- Do not answer the medical question.
"""


def analyze_question(llm, question: str, cancer_hint: str | None = None) -> QueryAnalysis:
    hint = f"\nDataset cancer label: {cancer_hint}" if cancer_hint else ""
    content = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}{hint}",
            },
        ],
        json_mode=True,
    )

    data = parse_json_object(content)
    analysis = QueryAnalysis(**data)

    if cancer_hint and not analysis.cancer_type:
        analysis.cancer_type = cancer_hint

    analysis.cancer_family = canonical_cancer_family(
        analysis.cancer_type or cancer_hint
    )

    # Ensure the explicit question and original query are always retrievable.
    queries = [question]
    queries.extend(analysis.search_queries)
    queries.extend(analysis.claims_to_check)
    queries.append(analysis.explicit_question)

    deduped = []
    seen = set()
    for q in queries:
        q = q.strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            deduped.append(q)

    analysis.search_queries = deduped[:8]
    return analysis
