import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.dataset import DataItem
from app.pipeline.sql_generation.fast_path import accepts_sql
from app.pipeline.sql_generation.sql_generation import SQLGenerationRunner
from app.pipeline.sql_revision.sql_revision import SQLRevisionRunner
from app.pipeline.sql_selection.sql_selection import SQLSelectionRunner
from experiments.compact_icl.routing import summarize


class RoutingTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = str(Path(temporary.name) / "db.sqlite")
        with sqlite3.connect(self.path) as connection:
            connection.executescript("CREATE TABLE customer(id); CREATE TABLE sale(customer);")
        self.sql = "SELECT id FROM customer"

    def test_screen(self):
        single = lambda sql, question="List customers", **kwargs: accepts_sql(
            question, sql, self.path, "single_table", **kwargs
        )
        self.assertTrue(single(self.sql))
        self.assertTrue(single("SELECT 'SELECT; JOIN' FROM customer; -- comment"))
        for sql in (
            "SELECT unknown FROM customer",
            "SELECT * FROM customer AS a, customer AS b",
            "SELECT * FROM (SELECT * FROM customer)",
            "SELECT id FROM customer GROUP BY id",
            "SELECT id FROM customer; SELECT id FROM customer",
            "DELETE FROM customer",
        ):
            with self.subTest(sql=sql):
                self.assertFalse(single(sql))
        self.assertFalse(single(self.sql, "Rank customers"))
        self.assertFalse(single(self.sql, timeout=0))
        join = "SELECT customer.id FROM customer JOIN sale ON customer.id = sale.customer"
        self.assertFalse(single(join))
        self.assertTrue(accepts_sql("List customers with sales", join, self.path, "one_join"))

    def test_shortcut_and_fallback(self):
        tokens = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        generation = SQLGenerationRunner.__new__(SQLGenerationRunner)
        revision = SQLRevisionRunner.__new__(SQLRevisionRunner)
        selection = SQLSelectionRunner.__new__(SQLSelectionRunner)
        generation._stage_config = SimpleNamespace(icl_fast_path="single_table", icl_sampling_budget=1,
                                                   dc_sampling_budget=2, skeleton_sampling_budget=2)
        generation._llm = generation._icl_llm = None
        generation._icl_generator = Mock()
        generation._icl_generator.generate.return_value = ([self.sql], tokens)
        generation._dc_generator = Mock()
        generation._dc_generator.generate.return_value = (["dc"], tokens)
        generation._skeleton_generator = Mock()
        generation._skeleton_generator.generate.return_value = (["skeleton"], tokens)
        item = DataItem(question_id=1, question="List customers", gold_sql=self.sql, database_id="db",
                        database_path=self.path, database_schema={}, total_time=0, total_llm_cost=dict.fromkeys(tokens, 0))
        with ThreadPoolExecutor() as pool, patch.dict("os.environ", {}, clear=True):
            generation._inner_thread_pool_executor = pool
            generation._generate_sql(item)
            generation._dc_generator.generate.assert_not_called()
            generation._skeleton_generator.generate.assert_not_called()
            generation._icl_generator.generate.assert_called_once()

            restored = item.model_copy()
            restored.apply_stage_artifact("sql_generation", item.get_stage_artifact("sql_generation").model_dump())
            with patch.object(revision, "_revise_one_candidate", side_effect=AssertionError), \
                    patch.object(selection, "_get_top_k_sql_candidates", side_effect=AssertionError):
                revision._revise_sql(restored)
                selection._select_best_sql(restored)
            self.assertEqual(restored.final_selected_sql, self.sql)
            self.assertEqual(restored.total_llm_cost, tokens)

            generation._icl_generator.generate.reset_mock()
            deferred = item.model_copy(update={"question": "Rank customers", "fast_path_sql": None})
            generation._generate_sql(deferred)
            generation._icl_generator.generate.assert_called_once()
            self.assertEqual(deferred.sql_candidates, ["dc", "skeleton", self.sql])

            generation._stage_config.icl_fast_path = "first"
            generation._icl_generator.generate.return_value = (None, tokens)
            generation._dc_generator.generate.reset_mock()
            generation._generate_sql(deferred)
            generation._dc_generator.generate.assert_not_called()
            revision._revise_sql(deferred)
            selection._select_best_sql(deferred)
            self.assertEqual(deferred.final_selected_sql, "Error")

    def test_attempt_accounting(self):
        result = {"id": "1", "database": "db", "full": 1, "first": 1, "single_table": True}
        requests = [
            {"stage": "run_schema_linking", "provider": "together", "ok": True, "prompt_tokens": 100, "completion_tokens": 0},
            {"stage": "run_sql_generation", "provider": "local", "ok": True, "item_id": "1", "prompt_tokens": 20, "completion_tokens": 0},
            {"stage": "run_sql_generation", "provider": "together", "ok": True, "item_id": "1", "prompt_tokens": 50, "completion_tokens": 0},
            {"stage": "run_sql_generation", "provider": "together", "ok": True, "item_id": "1", "prompt_tokens": 25, "completion_tokens": 0},
            {"stage": "run_sql_revision", "provider": "together", "item_id": "1", "reserve": 45_000},
        ]
        report = summarize([result], requests, ("full", "first", "single_table"))
        self.assertEqual(report["full"]["tokens"], 195)
        self.assertEqual(report["single_table"]["tokens"], 120)
        self.assertEqual(report["full"]["unknown_usage_requests"], 1)
        self.assertEqual(report["single_table"]["unknown_usage_requests"], 0)


if __name__ == "__main__":
    unittest.main()
