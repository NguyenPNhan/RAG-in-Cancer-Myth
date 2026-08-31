from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptOption:
    key: str
    label: str
    template: str

    def render(self, question: str) -> str:
        return self.template.format(question=question.strip())


PROMPT_OPTIONS = (
    PromptOption(
        key="basic",
        label="Basic prompt",
        template=(
            "Does the following patient question contain a false medical assumption?\n"
            "# Question:\n"
            "# {question}"
        ),
    ),
    PromptOption(
        key="oncology_expert",
        label="Oncology expert",
        template=(
            "You are an oncology expert.\n"
            "# Does the following patient question contain a false medical assumption?\n"
            "# Question:\n"
            "# {question}"
        ),
    ),
    PromptOption(
        key="patient_education",
        label="Cancer patient-education specialist",
        template=(
            "You are a cancer patient-education specialist.\n"
            "# Does the following patient question contain a false medical assumption?\n"
            "# Question:\n"
            "# {question}"
        ),
    ),
)

PROMPTS_BY_KEY = {option.key: option for option in PROMPT_OPTIONS}


def render_prompt(prompt_key: str, question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        option = PROMPTS_BY_KEY[prompt_key]
    except KeyError as exc:
        choices = ", ".join(PROMPTS_BY_KEY)
        raise ValueError(f"Unknown prompt '{prompt_key}'. Choose one of: {choices}.") from exc

    return option.render(question)
