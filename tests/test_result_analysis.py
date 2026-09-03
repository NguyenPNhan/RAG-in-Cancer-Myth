from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "result-analysis" / "analyze.py"
SPEC = importlib.util.spec_from_file_location("result_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class ResultAnalysisTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        dataset = {
            "1": {"correct_answer": True},
            "2": {"correct_answer": True},
            "3": {"correct_answer": False},
            "4": {"correct_answer": False},
        }
        predictions = {"1": True, "2": False, "3": True, "4": False}
        metrics = analysis.calculate_metrics(predictions, dataset, dataset)

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_exact_mcnemar_is_symmetric(self) -> None:
        self.assertEqual(analysis.exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(analysis.exact_mcnemar_p(3, 7), analysis.exact_mcnemar_p(7, 3))
        self.assertAlmostEqual(analysis.exact_mcnemar_p(0, 5), 0.0625)

    def test_undefined_class_metric_is_nan(self) -> None:
        dataset = {"1": {"correct_answer": True}}
        metrics = analysis.calculate_metrics({"1": True}, dataset, dataset)
        self.assertTrue(math.isnan(metrics["specificity"]))
        self.assertTrue(math.isnan(metrics["balanced_accuracy"]))

    def test_prediction_loader_rejects_duplicate_ids(self) -> None:
        dataset = {"1": {"correct_answer": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.csv"
            path.write_text("question_id,answer\n1,true\n1,false\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate question_id"):
                analysis.load_predictions(path, dataset)

    def test_jupyter_notebook_is_valid_and_runs_analyzer(self) -> None:
        notebook_path = MODULE_PATH.with_name("result_analysis.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertTrue(any("analysis.main" in source for source in code_cells))
        for index, source in enumerate(code_cells):
            compile(source, f"result_analysis.ipynb cell {index}", "exec")


if __name__ == "__main__":
    unittest.main()
