from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERRA_EXPERIMENT_ROOT = PROJECT_ROOT / "rag-terra"
if str(TERRA_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TERRA_EXPERIMENT_ROOT))

from rag_terra import TERRA_MODEL, terra_settings


class TerraConfigurationTests(unittest.TestCase):
    def test_terra_is_pinned_when_llm_model_is_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = terra_settings()
        self.assertEqual(settings.model, "gpt-5.6-terra")
        self.assertEqual(TERRA_MODEL, settings.model)

    def test_terra_overrides_generic_model_environment_setting(self) -> None:
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-5.6-luna"}, clear=True):
            settings = terra_settings()
        self.assertEqual(settings.model, "gpt-5.6-terra")

    def test_connection_environment_settings_are_preserved(self) -> None:
        environment = {
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_API_KEY": "terra-test-key",
            "LLM_TIMEOUT_SECONDS": "45",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = terra_settings()
        self.assertEqual(settings.base_url, "https://example.test/v1")
        self.assertEqual(settings.api_key, "terra-test-key")
        self.assertEqual(settings.timeout_seconds, 45.0)


if __name__ == "__main__":
    unittest.main()
