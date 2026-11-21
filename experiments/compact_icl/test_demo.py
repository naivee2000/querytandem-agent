"""Verify recorded demo behavior without model calls or benchmark databases."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from .export_demo import OUTPUT, REPORT, ROOT, execute_readonly, read_json, sha256, validate


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "example.sqlite"
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE numbers (value INTEGER)")
            conn.executemany("INSERT INTO numbers VALUES (?)", [(x,) for x in range(13)])

    def test_complete_results_determine_correctness_before_preview_truncation(self):
        first, first_rows = execute_readonly(self.path, "SELECT value FROM numbers")
        other, other_rows = execute_readonly(self.path, "SELECT value FROM numbers WHERE value != 9")
        self.assertEqual(first["rows_preview"], other["rows_preview"])
        self.assertNotEqual(first_rows, other_rows)
        self.assertEqual((first["row_count"], other["row_count"]), (13, 12))
        self.assertTrue(first["truncated"])

    def test_empty_results_are_distinct_from_errors_and_limits(self):
        empty, rows = execute_readonly(self.path, "SELECT value FROM numbers WHERE 0")
        self.assertEqual((empty["status"], rows), ("empty", frozenset()))
        for sql, options, expected in (
            ("invalid", {}, "sql_error"),
            ("SELECT value FROM numbers", {"timeout": 0}, "timeout"),
            ("SELECT value FROM numbers", {"max_rows": 5}, "row_limit"),
            (None, {}, "missing_sql"),
        ):
            result, rows = execute_readonly(self.path, sql, **options)
            self.assertEqual(result["status"], expected)
            self.assertIsNone(rows)
            self.assertIsNone(result["row_count"])

    def test_sql_cannot_modify_database(self):
        result, _ = execute_readonly(self.path, "DELETE FROM numbers RETURNING value")
        self.assertEqual(result["status"], "sql_error")
        _, rows = execute_readonly(self.path, "SELECT count(*) FROM numbers")
        self.assertEqual(rows, frozenset({(13,)}))

    def test_bundle_matches_published_metrics_and_preserves_regression(self):
        bundle = read_json(OUTPUT)
        validate(bundle)
        report = read_json(ROOT / REPORT)
        self.assertEqual(bundle["source"]["artifact_sha256"][REPORT], sha256(ROOT / REPORT))
        for key, counts in bundle["benchmark"]["variants"].items():
            self.assertEqual(counts, {field: report[key][field] for field in counts})
        success, regression, failure = bundle["examples"]
        self.assertLess(success["variants"]["tuned-compact"]["tokens"]["local_generator"]["total_tokens"],
                        success["variants"]["baseline"]["tokens"]["local_generator"]["total_tokens"])
        self.assertGreater(success["variants"]["tuned-compact"]["tokens"]["pipeline"]["total_tokens"],
                           success["variants"]["baseline"]["tokens"]["pipeline"]["total_tokens"])
        self.assertFalse(regression["variants"]["tuned-compact"]["correct"])
        self.assertEqual(failure["reference_result"]["row_count"], 67)
        self.assertEqual(failure["variants"]["tuned-compact"]["result"]["row_count"], 72)

    def test_modified_metrics_outcomes_and_preview_are_rejected(self):
        bundle = read_json(OUTPUT)
        for modification in (
            lambda b: b["benchmark"].update(pipeline_token_savings_fraction=.7),
            lambda b: b["examples"][1]["variants"]["tuned-compact"].update(correct=True),
            lambda b: b["examples"][2]["reference_result"].update(truncated=False),
        ):
            modified = deepcopy(bundle)
            modification(modified)
            with self.assertRaises(ValueError):
                validate(modified)

    def test_portable_validation_needs_only_two_files_and_standard_library(self):
        folder = Path(self.temporary.name)
        script = folder / "export_demo.py"
        data = folder / "demo-data.json"
        shutil.copyfile(ROOT / "experiments/compact_icl/export_demo.py", script)
        shutil.copyfile(OUTPUT, data)
        result = subprocess.run([sys.executable, "-I", str(script), "--verify", "--output", str(data)],
                                cwd=folder, capture_output=True, text=True, check=True)
        self.assertIn("Verified three recorded examples", result.stdout)


if __name__ == "__main__":
    unittest.main()
