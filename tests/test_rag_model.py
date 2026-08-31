from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag.rag_model.config import LLMSettings
from rag.rag_model.corpus import load_pdq_chunks
from rag.rag_model.llm import SYSTEM_MESSAGE, parse_boolean_response
from rag.rag_model.prompts import PROMPT_OPTIONS, render_prompt
from rag.rag_model.retriever import BM25Retriever, format_evidence


class PromptTests(unittest.TestCase):
    def test_all_three_prompts_include_question(self) -> None:
        question = "Is surgery the only treatment for bladder cancer?"
        self.assertEqual(len(PROMPT_OPTIONS), 3)
        for option in PROMPT_OPTIONS:
            rendered = render_prompt(option.key, question)
            self.assertIn(question, rendered)
            self.assertIn("false medical assumption", rendered)


class BooleanParserTests(unittest.TestCase):
    def test_accepts_only_boolean_tokens(self) -> None:
        self.assertIs(parse_boolean_response("true"), True)
        self.assertIs(parse_boolean_response(" FALSE\n"), False)

    def test_rejects_explanations(self) -> None:
        with self.assertRaises(ValueError):
            parse_boolean_response("true, because the assumption is incorrect")

    def test_accepts_qwen_empty_thinking_wrapper(self) -> None:
        self.assertIs(parse_boolean_response("<think>\n</think>\ntrue"), True)

    def test_rejects_nonempty_thinking_wrapper(self) -> None:
        with self.assertRaises(ValueError):
            parse_boolean_response("<think>reasoning</think>true")


class QwenConfigurationTests(unittest.TestCase):
    def test_qwen3_8b_is_default_model(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = LLMSettings.from_environment()
        self.assertEqual(settings.model, "qwen3:8b")

    def test_qwen_thinking_is_disabled(self) -> None:
        self.assertIn("/no_think", SYSTEM_MESSAGE)


class CorpusAndRetrievalTests(unittest.TestCase):
    def test_loads_and_retrieves_pdq_section(self) -> None:
        raw = {
            "cancer_type": "Bladder Cancer",
            "pages": [
                {
                    "title": "Bladder Cancer Treatment",
                    "topic": "adult_treatment",
                    "audience": "patient",
                    "url": "https://example.test/bladder",
                    "sections": [
                        {
                            "section": "Treatment options",
                            "section_path": ["Treatment", "Options"],
                            "is_boilerplate": False,
                            "text": "Bladder cancer may be treated with surgery, radiation, and chemotherapy.",
                        },
                        {
                            "section": "About PDQ",
                            "is_boilerplate": True,
                            "text": "This text should not be indexed.",
                        },
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bladder.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            chunks = load_pdq_chunks((temp_dir,))

        self.assertEqual(len(chunks), 1)
        retriever = BM25Retriever(chunks)
        results = retriever.search("bladder chemotherapy treatment", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("chemotherapy", format_evidence(results))


if __name__ == "__main__":
    unittest.main()
