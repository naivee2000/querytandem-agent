"""Local tests for SQL execution, scoring, and selection fallback."""

import math
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.pipeline.sql_selection.reranker import Answer, FEATURES, SQLReranker, group_answers, majority_vote, read_answer
from app.pipeline.sql_selection.sql_selection import SQLSelectionRunner
from experiments.compact_icl.rerank_data import split
from experiments.compact_icl.rerank_experiment import fit, labeled, snapshot_complete
from experiments.compact_icl.augment_reranker import load_inputs, parse_sql
from experiments.compact_icl.refine_reranker import forecast, frozen_json, prompt_records
from experiments.compact_icl.routing import request_rows
from experiments.compact_icl.rerank_experiment import usage


class SQLRerankerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = str(Path(temporary.name) / "example.sqlite")
        with sqlite3.connect(self.path) as database:
            database.executescript("CREATE TABLE names (name TEXT); INSERT INTO names VALUES ('Mel'), ('Lee');")

    def test_groups_equivalent_queries_and_keeps_empty_answers(self):
        first = "SELECT name FROM names"
        equivalent = "SELECT name FROM names ORDER BY name"
        empty = "SELECT name FROM names WHERE 0"
        item = SimpleNamespace(database_path=self.path, question="List the names", evidence="",
                               sql_candidates_after_revision=[first, equivalent, empty, "garbage"])
        agreed, empty_group = group_answers(item)
        self.assertEqual(agreed.sqls, (first, equivalent))
        self.assertEqual(empty_group.answer.rows, frozenset())
        self.assertEqual(majority_vote([agreed, empty_group], empty), first)
        weights = tuple(3 if label == "empty" else 0 for label in FEATURES)
        self.assertEqual(SQLReranker(weights, .9).choose([agreed, empty_group], first), empty)

    def test_read_only_deadline_and_missing_score(self):
        self.assertIsNone(read_answer(self.path, "DELETE FROM names RETURNING name", time.monotonic() + 1))
        self.assertIsNone(read_answer(self.path, "SELECT * FROM names", time.monotonic()))
        current = "SELECT * FROM names"
        self.assertEqual(SQLReranker((math.nan,) * len(FEATURES), .6).choose([], current), current)

    def test_offline_timeout_retains_other_results_and_runtime_falls_back(self):
        clock = [0.]
        deadlines = []
        def execute(_path, sql, deadline, *, diagnostics=None):
            deadlines.append(deadline)
            clock[0] += 2 if sql == "slow" else .01
            if diagnostics is not None:
                diagnostics.update(status="timeout" if sql == "slow" else "ok", seconds=0.)
            return None if sql == "slow" else Answer(frozenset({(sql,)}), 1, 1)
        item = SimpleNamespace(database_path=self.path, question="names", evidence="",
                               sql_candidates_after_revision=["first", "slow", "last", "first"])
        with patch("app.pipeline.sql_selection.reranker.time.monotonic", side_effect=lambda: clock[0]), patch(
                "app.pipeline.sql_selection.reranker.read_answer", side_effect=execute):
            diagnostics = []
            groups = group_answers(item, timeout=1., per_query_timeout=True, diagnostics=diagnostics)
            self.assertEqual(len(groups), 2)
            self.assertEqual(len(groups[0].sqls), 2)
            self.assertEqual([row["status"] for row in diagnostics], ["ok", "timeout", "ok"])
            self.assertGreater(deadlines[-1], deadlines[0] + 2)
            clock[0] = 0.
            self.assertEqual(group_answers(item, timeout=1.), [])

    def test_execution_diagnostics_distinguish_errors_and_timeouts(self):
        for sql, remaining, expected in [("SELECT 1", 1, "ok"), ("garbage", 1, "sql_error"),
                                         ("SELECT 1", -1, "timeout"), ("", 1, "missing_sql")]:
            diagnostics = {}
            read_answer(self.path, sql, time.monotonic() + remaining, diagnostics=diagnostics)
            self.assertEqual(diagnostics["status"], expected)

    def test_offline_labels_are_correct_and_report_invalid_candidates(self):
        item = SimpleNamespace(database_path=self.path, question="List names", evidence="", database_id="names",
                               get_item_id=lambda: "1", gold_sql="SELECT name FROM names",
                               sql_candidates_after_revision=["SELECT name FROM names", "SELECT name FROM names WHERE 0", "bad"])
        diagnostics = {}
        good, bad = labeled(item, diagnostics=diagnostics)
        self.assertEqual((len(good), len(bad)), (1, 1))
        self.assertEqual(diagnostics["gold"]["status"], "ok")
        self.assertEqual(diagnostics["candidates"][-1]["status"], "sql_error")

    def test_incomplete_checkpoint_is_not_treated_as_finished(self):
        item = SimpleNamespace(is_stage_complete=lambda stage: False)
        with patch("experiments.compact_icl.rerank_experiment.load_dataset", return_value=[item]):
            self.assertFalse(snapshot_complete(self.path, "sql_selection"))

    def test_augmentation_input_and_prompt_never_include_gold(self):
        class Item:
            database_schema_after_schema_linking = {"schema": "cached inference schema"}
            question = "List names"
            evidence = ""
            database_id = "names"
            def get_item_id(self):
                return "1"
            @property
            def gold_sql(self):
                raise AssertionError("Gold SQL accessed during prompt construction")
        with patch("experiments.compact_icl.refine_reranker.get_schema_service") as service:
            service.return_value.build_schema_profile.return_value = "CREATE TABLE names(name TEXT)"
            rows = prompt_records([Item()])
        path = Path(self.path).with_name("prompts.json")
        path.write_text(json.dumps({"items": rows}))
        self.assertEqual(load_inputs(path)["items"], rows)
        rows[0]["gold_sql"] = "SELECT name FROM names"
        path.write_text(json.dumps({"items": rows}))
        with self.assertRaises(ValueError):
            load_inputs(path)

    def test_augmentation_parser_requires_complete_output_format(self):
        self.assertIsNone(parse_sql("<result>SELECT 1"))
        self.assertEqual(parse_sql("<result>```sql\nSELECT 1\n```</result>"), "SELECT 1")

    def test_changed_cached_inputs_are_rejected(self):
        path = Path(self.path).with_name("manifest.json")
        frozen_json(path, {"prompt": "first"})
        frozen_json(path, {"prompt": "first"})
        with self.assertRaises(ValueError):
            frozen_json(path, {"prompt": "different"})

    def test_budget_reserves_fresh_evaluation_and_accounts_for_prior_spend(self):
        promising = forecast(40, 12, .53, 8., .53)
        self.assertTrue(promising["feasible"])
        self.assertGreaterEqual(promising["fresh_reserve_usd"], 3.2)
        self.assertLessEqual(promising["projected_total_usd"], 8.)
        self.assertFalse(forecast(40, 5, .53, 8., .53)["feasible"])
        self.assertFalse(forecast(40, 12, .53, 8., 4.)["feasible"])

    def test_disabled_runner_uses_standard_selection(self):
        runner = SQLSelectionRunner.__new__(SQLSelectionRunner)
        runner._reranker = None
        item = SimpleNamespace()
        with patch.object(runner, "_select_best_sql") as baseline, patch(
            "app.pipeline.sql_selection.sql_selection.group_answers", side_effect=AssertionError("reranker called")
        ):
            runner._select_with_optional_reranker(item)
        baseline.assert_called_once_with(item)

    def test_enabled_runner_overrides_without_changing_token_usage(self):
        first = "SELECT name FROM names"
        empty = "SELECT name FROM names WHERE 0"
        runner = SQLSelectionRunner.__new__(SQLSelectionRunner)
        runner._reranker = SQLReranker(tuple(3 if label == "empty" else 0 for label in FEATURES), .9)
        item = SimpleNamespace(database_path=self.path, question="List the names", evidence=None,
                               sql_candidates_after_revision=[first, empty], fast_path_sql=None,
                               final_selected_sql=first, sql_selection_time=0., total_time=0.,
                               sql_selection_llm_cost={"total_tokens": 123})
        with patch.object(runner, "_select_best_sql") as baseline:
            runner._select_with_optional_reranker(item)
        baseline.assert_called_once_with(item)
        self.assertEqual(item.final_selected_sql, empty)
        self.assertEqual(item.sql_selection_llm_cost, {"total_tokens": 123})
        self.assertGreater(item.sql_selection_time, 0.)

    def test_pairwise_fit_learns_correct_direction_and_has_minimum_count(self):
        correct = (1., *([0.] * (len(FEATURES) - 1)))
        incorrect = (0.,) * len(FEATURES)
        self.assertIsNone(fit([([correct], [incorrect])] * 59))
        self.assertGreater(fit([([correct], [incorrect])] * 60)[0], 0)

    def test_saved_ranker_loads_and_rejects_invalid_weights(self):
        path = Path(self.path).with_name("ranker.json")
        artifact = dict(version=1, features=list(FEATURES), weights=[0.] * len(FEATURES), threshold=.75)
        path.write_text(json.dumps(artifact))
        model = SQLReranker.load(path)
        self.assertEqual(model.choose([], "SELECT 1"), "SELECT 1")
        artifact["weights"][0] = float("nan")
        path.write_text(json.dumps(artifact))
        with self.assertRaises(ValueError):
            SQLReranker.load(path)

    def test_legacy_selection_configuration_remains_disabled(self):
        from app.config.config import LLMConfig, SQLSelectionConfig

        config = SQLSelectionConfig(llm=LLMConfig(model="test", base_url="http://localhost/v1", api_key="test"))
        self.assertIsNone(config.reranker_path)
        self.assertEqual(config.filter_top_k_sql, 2)

    def test_retry_usage_keeps_unknown_charge_without_double_counting(self):
        path = Path(self.path).with_name("usage.jsonl")
        rows = [
            dict(request=1, provider="together", reserve=44821),
            dict(request=1, provider="together", ok=False, prompt_tokens=None, completion_tokens=None),
            dict(request=2, provider="together", reserve=44821),
            dict(request=2, provider="together", ok=True, prompt_tokens=10, completion_tokens=5),
            dict(request=3, provider="local", reserve=0),
            dict(request=3, provider="local", ok=True, prompt_tokens=20, completion_tokens=3),
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = usage(request_rows(path))
        self.assertEqual((result["requests"], result["tokens"], result["unknown_usage_requests"]), (3, 38, 1))
        self.assertAlmostEqual(result["api_usd_with_reservations"], .044821 + .00000295)

    @unittest.skipUnless(Path("data/bird/downloads/train.jsonl").is_file(), "BIRD cache is optional")
    def test_training_split_is_disjoint(self):
        pilot, _ = split("pilot")
        train, _ = split("train")
        fresh, _ = split("fresh")
        self.assertEqual((len(pilot), len(train), len(fresh)), (40, 360, 200))
        self.assertEqual(len({idx for idx, _ in pilot + train}), 400)
        self.assertFalse({row["db_id"] for _, row in pilot + train} & {row["db_id"] for _, row in fresh})
        shards = [entry for number in range(1, 10) for entry in split(f"refine-{number:02d}")[0]]
        self.assertEqual(shards, train)
        with self.assertRaises(ValueError):
            split("refine-10")


if __name__ == "__main__":
    unittest.main()
