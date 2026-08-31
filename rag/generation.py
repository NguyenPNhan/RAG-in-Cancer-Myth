from __future__ import annotations

import json

from .llm import parse_json_object


SYSTEM_PROMPT = """
You answer cancer questions using ONLY the supplied NCI/PDQ evidence.

The question may contain false, overgeneralized, or unsupported assumptions.

Rules:
1. Do not accept a premise merely because the user states it.
2. Evaluate important assumptions against the supplied evidence.
3. For each assumption, use one of:
   - supported
   - contradicted
   - partially_supported
   - insufficient_evidence
4. If an assumption is contradicted or only partly supported, correct it before
   answering the user's secondary question.
5. Do not invent treatment recommendations, prognosis, survival statistics,
   screening schedules, or contraindications absent from the evidence.
6. Do not make individualized treatment decisions.
7. If the evidence does not answer something, explicitly say so.
8. Cite evidence with chunk IDs in square brackets, for example [abc123].
9. Prefer health-professional PDQ for precise factual claims; patient PDQ can
   support clearer wording.

Return JSON only:
{
  "premise_assessment": [
    {
      "claim": "...",
      "status": "supported|contradicted|partially_supported|insufficient_evidence",
      "reason": "...",
      "evidence_ids": ["..."]
    }
  ],
  "answer": "...",
  "evidence_ids": ["..."],
  "confidence": "high|moderate|low"
}
"""


def _format_evidence(evidence: list[dict]) -> str:
    blocks = []
    for i, e in enumerate(evidence, start=1):
        blocks.append(
            "\n".join([
                f"EVIDENCE {i}",
                f"chunk_id: {e.get('chunk_id')}",
                f"title: {e.get('title')}",
                f"cancer_type: {e.get('cancer_type')}",
                f"topic: {e.get('topic')}",
                f"audience: {e.get('audience')}",
                f"section: {e.get('section')}",
                f"url: {e.get('url')}",
                f"text: {e.get('text')}",
            ])
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(llm, question: str, analysis, evidence: list[dict]) -> dict:
    context = _format_evidence(evidence)

    user_prompt = f"""
QUESTION
{question}

EXTRACTED CLAIMS TO CHECK
{json.dumps(analysis.claims_to_check, ensure_ascii=False)}

UNDERLYING QUESTION
{analysis.explicit_question}

NCI/PDQ EVIDENCE
{context}
"""

    content = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        json_mode=True,
    )
    return parse_json_object(content)
