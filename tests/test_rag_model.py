from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rag.rag_model.config import LLMSettings
from rag.rag_model.corpus import load_pdq_chunks
from rag.rag_model.llm import OpenAICompatibleBooleanClient, parse_boolean_response
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
    def test_accepts_schema_boolean_values(self) -> None:
        self.assertIs(parse_boolean_response('{"value": true}'), True)
        self.assertIs(parse_boolean_response('{"value": false}'), False)

    def test_rejects_non_json_and_non_boolean_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_boolean_response("true")
        with self.assertRaises(ValueError):
            parse_boolean_response('{"value": "true"}')

    def test_rejects_missing_or_additional_fields(self) -> None:
        with self.assertRaises(ValueError):
            parse_boolean_response("{}")
        with self.assertRaises(ValueError):
            parse_boolean_response('{"value": true, "explanation": "..."}')


class OpenAIConfigurationTests(unittest.TestCase):
    def test_openai_is_the_default_provider(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = LLMSettings.from_environment()
        self.assertEqual(settings.base_url, "https://api.openai.com/v1")
        self.assertEqual(settings.model, "gpt-5.6-luna")

    def test_openai_api_key_is_read_from_environment(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "openai-test-key"}, clear=True):
            settings = LLMSettings.from_environment()
        self.assertEqual(settings.api_key, "openai-test-key")

    def test_generic_api_key_override_takes_precedence(self) -> None:
        environment = {
            "OPENAI_API_KEY": "openai-test-key",
            "LLM_API_KEY": "generic-test-key",
        }
        with patch.dict("os.environ", environment, clear=True):
            settings = LLMSettings.from_environment()
        self.assertEqual(settings.api_key, "generic-test-key")


class OpenAIClientTests(unittest.TestCase):
    @patch("rag.rag_model.llm.httpx.Client")
    def test_gpt_5_6_luna_request_uses_openai_endpoint(self, client_class: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"value": true}'}}],
        }
        client_class.return_value.__enter__.return_value.post.return_value = response
        settings = LLMSettings(
            base_url="https://api.openai.com/v1",
            model="gpt-5.6-luna",
            api_key="openai-test-key",
        )

        content = OpenAICompatibleBooleanClient(settings)._chat(
            [{"role": "user", "content": "Classify this."}],
        )

        self.assertEqual(content, '{"value": true}')
        request = client_class.return_value.__enter__.return_value.post.call_args
        self.assertEqual(
            request.args[0],
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(request.kwargs["headers"]["Authorization"], "Bearer openai-test-key")
        self.assertEqual(request.kwargs["json"]["model"], "gpt-5.6-luna")
        self.assertEqual(
            request.kwargs["json"]["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "medical_premise_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "boolean"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        self.assertNotIn("max_tokens", request.kwargs["json"])
        self.assertEqual(request.kwargs["json"]["max_completion_tokens"], 32)
        self.assertEqual(request.kwargs["json"]["reasoning_effort"], "none")
        self.assertNotIn("include_reasoning", request.kwargs["json"])

    @patch.object(OpenAICompatibleBooleanClient, "_chat")
    def test_classifier_uses_no_system_message(self, chat: MagicMock) -> None:
        chat.return_value = '{"value": false}'
        client = OpenAICompatibleBooleanClient(
            LLMSettings(base_url="https://api.openai.com/v1", model="gpt-5.6-luna")
        )

        result = client.classify("Classify the question.", "Evidence text")

        self.assertIs(result, False)
        messages = chat.call_args.args[0]
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("# NCI PDQ Evidence:", messages[0]["content"])

    @patch.object(OpenAICompatibleBooleanClient, "_chat")
    def test_classifier_can_run_without_rag_evidence(self, chat: MagicMock) -> None:
        chat.return_value = '{"value": true}'
        client = OpenAICompatibleBooleanClient(
            LLMSettings(base_url="https://api.openai.com/v1", model="gpt-5.6-luna")
        )

        result = client.classify("Classify the question.")

        self.assertIs(result, True)
        message = chat.call_args.args[0][0]
        self.assertEqual(message["role"], "user")
        self.assertNotIn("NCI PDQ", message["content"])

    @patch("rag.rag_model.llm.httpx.Client")
    def test_model_refusal_has_a_clear_error(self, client_class: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": None, "refusal": "Cannot classify."}}],
        }
        client_class.return_value.__enter__.return_value.post.return_value = response
        client = OpenAICompatibleBooleanClient(
            LLMSettings(base_url="https://api.openai.com/v1", model="gpt-5.6-luna")
        )

        with self.assertRaisesRegex(RuntimeError, "refused.*Cannot classify"):
            client._chat([{"role": "user", "content": "Classify this."}])


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
